"""固定幅の上限が公式 engine の到達範囲を覆っているかを検証する。

port は ``jit`` の静的形状要求のため、engine が可変長で持つものを固定幅の
配列に置き換えている:

    hands          -> MAX_UNITS スロット (index 0 は farmer)
    market orders  -> MAX_MARKET_ORDERS スロット
    1 注文の数量   -> MAX_ORDER_UNITS でループを打ち切り

これらが engine の実際の到達範囲より小さいと、**上限付近だけ静かに挙動が
ずれる**。無作為ファズは上限に到達しないため見つけられない (実際 hire は
1 ターン数件しか出ず、上限に迫らなかった)。ここは意図的に上限を叩く。

``MAX_UNITS`` は当初 16 だったが、engine は開始資金 $3,000 だけで 16 人
雇えるのに port 側は farmer を含めて 15 人分しか席が無く、
このテストで不一致が出た。
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX is an optional dependency")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional dependency")

from kaggle_environments.envs.kaggriculture import (  # noqa: E402
    kaggriculture as engine,
)

from simulate.jaxenv import env as E  # noqa: E402
from simulate.jaxenv.state import MAX_UNITS, initial_state  # noqa: E402

from .test_jaxenv_equivalence import (  # noqa: E402
    DETERMINISTIC_STEP,
    _diff,
    _engine_snapshot,
    _make_engine,
    _port_snapshot,
)


def test_hire_cost_table_matches_engine() -> None:
    """port の fib テーブルが engine の ``_hire_cost`` と一致する。"""
    for n in range(MAX_UNITS):
        assert int(E.HIRE_COST[n]) == engine._hire_cost(n), f"hire #{n}"


def test_max_units_covers_engine_reachable_hands() -> None:
    """開始資金で engine が雇える人数を port が収容できる。

    fib の累積コストから理論値を出し、席数と突き合わせる。実 pod を使わず
    純粋な計算で確かめられるので、回帰検出用に軽い。
    """
    budget = 3000  # startingMoney の既定
    total = 0
    reachable = 0
    while True:
        cost = engine._hire_cost(reachable)
        if total + cost > budget:
            break
        total += cost
        reachable += 1

    assert reachable == 16, f"engine の到達 hand 数が変わった: {reachable}"
    # index 0 は farmer なので、hand 用の席は MAX_UNITS - 1。
    assert MAX_UNITS - 1 >= reachable, (
        f"MAX_UNITS={MAX_UNITS} では hand 席が {MAX_UNITS - 1} しかないが、"
        f"engine は開始資金だけで {reachable} 人雇える"
    )


def test_mass_hire_matches_engine() -> None:
    """1 ターン 10 件の HIRE を連発し、上限付近まで engine と一致するか。

    2 ターン目で engine は 16 人に到達する。MAX_UNITS が不足していると
    ここで hands 数と carried の長さがずれる。
    """
    steps = 24
    env = _make_engine(steps, seed=3)
    eng_state = env.state
    port = initial_state(1, seed=3)

    orders_per_turn = E.MAX_MARKET_ORDERS
    unit_shape = (1, 2, MAX_UNITS)
    market_shape = (1, 2, E.MAX_MARKET_ORDERS)
    zeros_u = jnp.zeros(unit_shape, dtype=jnp.int32)
    zeros_m = jnp.zeros(market_shape, dtype=jnp.int32)
    hire_ops = zeros_m.at[:, :, :orders_per_turn].set(E.MARKET_HIRE)

    peak = 0
    for n in range(steps):
        for p in (0, 1):
            n_hands = len(eng_state[0].observation.farms[p]["hands"])
            eng_state[p].action = {
                "farmer": ["PASS"],
                "hands": [["PASS"]] * n_hands,
                "market": [["HIRE"] for _ in range(orders_per_turn)],
            }
        engine.interpreter(eng_state, env)
        eng_state[0].observation.step = n + 1

        port = DETERMINISTIC_STEP(
            port, zeros_u, zeros_u, zeros_u, hire_ops, zeros_m, zeros_m
        )

        peak = max(peak, len(eng_state[0].observation.farms[0]["hands"]))
        diffs = _diff(_engine_snapshot(eng_state), _port_snapshot(port))
        assert not diffs, (
            f"step {n} で不一致 (engine hands="
            f"{len(eng_state[0].observation.farms[0]['hands'])}):\n  "
            + "\n  ".join(diffs[:8])
        )

    # 上限に迫っていなければテストとして無意味なので、到達を主張する。
    assert peak >= 16, f"hand 数が {peak} までしか伸びておらず上限を検証できていない"


def test_full_market_order_slots_match_engine() -> None:
    """market スロットを全部埋めても engine と一致する。

    ``maxMarketOrdersPerTurn`` の既定は 10 で、port の MAX_MARKET_ORDERS と
    同じ。engine は 11 件目以降を捨てるので、境界がずれていないか見る。
    """
    steps = 48
    env = _make_engine(steps, seed=5)
    eng_state = env.state
    port = initial_state(1, seed=5)

    n_orders = E.MAX_MARKET_ORDERS
    unit_shape = (1, 2, MAX_UNITS)
    market_shape = (1, 2, E.MAX_MARKET_ORDERS)
    zeros_u = jnp.zeros(unit_shape, dtype=jnp.int32)
    zeros_m = jnp.zeros(market_shape, dtype=jnp.int32)
    # 全スロットで WHEAT の種を買う。資金が尽きると engine は失敗させるので、
    # 「途中から通らなくなる」境界も併せて検証できる。
    buy_ops = zeros_m.at[:, :, :n_orders].set(E.MARKET_BUY_SEED)
    buy_qty = zeros_m.at[:, :, :n_orders].set(3)

    for n in range(steps):
        for p in (0, 1):
            eng_state[p].action = {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 3] for _ in range(n_orders)],
            }
        engine.interpreter(eng_state, env)
        eng_state[0].observation.step = n + 1

        port = DETERMINISTIC_STEP(
            port, zeros_u, zeros_u, zeros_u, buy_ops, zeros_m, buy_qty
        )

        diffs = _diff(_engine_snapshot(eng_state), _port_snapshot(port))
        assert not diffs, f"step {n} で不一致:\n  " + "\n  ".join(diffs[:8])


def test_large_single_order_matches_engine() -> None:
    """1 注文で大量に売る。``MAX_ORDER_UNITS`` の打ち切りが効いていないか。

    shed 容量は 100 なので、100 個売る注文が port の 100 トリップに収まる。
    ここが不足していると売れ残りが出て資金がずれる。
    """
    steps = 30
    env = _make_engine(steps, seed=9)
    eng_state = env.state
    port = initial_state(1, seed=9)

    unit_shape = (1, 2, MAX_UNITS)
    market_shape = (1, 2, E.MAX_MARKET_ORDERS)
    zeros_u = jnp.zeros(unit_shape, dtype=jnp.int32)
    zeros_m = jnp.zeros(market_shape, dtype=jnp.int32)

    for n in range(steps):
        # 前半で WHEAT を大量に仕込み、後半で 1 注文 100 個を売る。
        if n < 15:
            m_engine = [["BUY_PRODUCT", "WHEAT", 8]]
            m_op = zeros_m.at[:, :, 0].set(E.MARKET_BUY_PRODUCT)
            m_qty = zeros_m.at[:, :, 0].set(8)
        else:
            m_engine = [["SELL", "WHEAT", 100]]
            m_op = zeros_m.at[:, :, 0].set(E.MARKET_SELL)
            m_qty = zeros_m.at[:, :, 0].set(100)

        for p in (0, 1):
            eng_state[p].action = {
                "farmer": ["PASS"], "hands": [], "market": m_engine,
            }
        engine.interpreter(eng_state, env)
        eng_state[0].observation.step = n + 1

        port = DETERMINISTIC_STEP(
            port, zeros_u, zeros_u, zeros_u, m_op, zeros_m, m_qty
        )

        diffs = _diff(_engine_snapshot(eng_state), _port_snapshot(port))
        assert not diffs, f"step {n} で不一致:\n  " + "\n  ".join(diffs[:8])
