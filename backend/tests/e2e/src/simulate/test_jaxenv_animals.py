"""家畜のライフサイクルを台本で駆動し、公式 engine と突き合わせる。

``test_jaxenv_equivalence.py`` の無作為ファズは全 op を生成するが、家畜は
**一度も盤上に配置されない**。engine の PLACE は「acting unit の所持品に
家畜がある」ことを要求し、それには

    BUY_ANIMAL (shed に入る) → shed tile に立つ → PICKUP → 構造物まで歩く → PLACE

という 5 手の連鎖が要る。無作為行動でこれが揃う確率は事実上ゼロで、実際
30 日 × 複数 seed のファズでも ``animals: {}`` (配置ゼロ) だった。

つまりファズだけでは家畜の産出・給餌・世話ボーナス・脱走・肥料が
**全て未検証**のまま通ってしまう。ここを台本で埋める。

検証対象:
  - BUILD_COOP / BUILD_PASTURE で構造物を建てる
  - BUY_ANIMAL → PICKUP → PLACE で家畜を配置する
  - FEED (WHEAT を消費)、CARE (世話ボーナスを貯める)
  - 産出間隔ごとの yield と max_held の上限
  - COLLECT_FERTILIZER
  - 給餌を止めたときの脱走 (構造物は残る)
"""

from __future__ import annotations

from typing import Any

import pytest

jax = pytest.importorskip("jax", reason="JAX is an optional dependency")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional dependency")

from kaggle_environments.envs.kaggriculture import (  # noqa: E402
    kaggriculture as engine,
)

from simulate.jaxenv import env as E  # noqa: E402
from simulate.jaxenv.state import MAX_UNITS, N_PRODUCTS, initial_state  # noqa: E402

from .test_jaxenv_equivalence import (  # noqa: E402
    DETERMINISTIC_STEP,
    _diff,
    _engine_snapshot,
    _make_engine,
    _port_snapshot,
)

#: 家畜は product 域の後ろに続く番号で指定する (port の PICKUP/PLACE 規約)。
GOOSE_ITEM = N_PRODUCTS + 0
COW_ITEM = N_PRODUCTS + 1

#: 盤上の座標。(4,4) は shed access かつ NW quadrant。
SHED_X, SHED_Y = 4, 4
#: COOP を建てる場所。shed から 1 歩北。
COOP_X, COOP_Y = 4, 3


def _run_script(
    script: list[tuple[str, int, int, int, int, int]], seed: int = 7
) -> None:
    """台本を両エンジンに流し、毎ターン全状態を突き合わせる。

    script の各要素は ``(unit_op, arg, qty, market_op, market_item, market_qty)``
    を表す 6 要素。player 0 のみ動かし、player 1 は PASS で固定する
    (家畜の挙動を見たいので、対戦相手の撹乱を排除する)。
    """
    steps = len(script)
    env = _make_engine(steps, seed)
    eng_state = env.state
    port = initial_state(1, seed=seed)

    for n, (op_name, arg, qty, mop, mitem, mqty) in enumerate(script):
        op_code = E.OP_NAMES.index(op_name)

        # --- engine 側 ---
        unit_action: list[Any] = [op_name]
        if op_name == "PLANT":
            from simulate.jaxenv.state import CROP_NAMES

            unit_action = ["PLANT", CROP_NAMES[arg]]
        elif op_name in ("PICKUP", "PLACE"):
            from simulate.jaxenv.state import ANIMAL_NAMES, PRODUCT_NAMES

            name = (
                PRODUCT_NAMES[arg]
                if arg < N_PRODUCTS
                else ANIMAL_NAMES[arg - N_PRODUCTS]
            )
            unit_action = [op_name, name, qty]

        market: list[list[Any]] = []
        if mop != E.MARKET_NONE:
            from simulate.jaxenv.state import ANIMAL_NAMES, CROP_NAMES, PRODUCT_NAMES

            if mop == E.MARKET_BUY_ANIMAL:
                market = [["BUY_ANIMAL", ANIMAL_NAMES[mitem], mqty]]
            elif mop == E.MARKET_BUY_SEED:
                market = [["BUY_SEED", CROP_NAMES[mitem], mqty]]
            elif mop == E.MARKET_BUY_PRODUCT:
                market = [["BUY_PRODUCT", PRODUCT_NAMES[mitem], mqty]]
            elif mop == E.MARKET_SELL:
                market = [["SELL", PRODUCT_NAMES[mitem], mqty]]

        eng_state[0].action = {"farmer": unit_action, "hands": [], "market": market}
        eng_state[1].action = {"farmer": ["PASS"], "hands": [], "market": []}
        engine.interpreter(eng_state, env)
        eng_state[0].observation.step = n + 1

        # --- port 側 ---
        shape = (1, 2, MAX_UNITS)
        mshape = (1, 2, E.MAX_MARKET_ORDERS)
        u_op = jnp.zeros(shape, dtype=jnp.int32).at[0, 0, 0].set(op_code)
        u_arg = jnp.zeros(shape, dtype=jnp.int32).at[0, 0, 0].set(arg)
        u_qty = jnp.zeros(shape, dtype=jnp.int32).at[0, 0, 0].set(qty)
        m_op = jnp.zeros(mshape, dtype=jnp.int32).at[0, 0, 0].set(mop)
        m_item = jnp.zeros(mshape, dtype=jnp.int32).at[0, 0, 0].set(mitem)
        m_qty = jnp.zeros(mshape, dtype=jnp.int32).at[0, 0, 0].set(mqty)

        port = DETERMINISTIC_STEP(port, u_op, u_arg, u_qty, m_op, m_item, m_qty)

        diffs = _diff(_engine_snapshot(eng_state), _port_snapshot(port))
        assert not diffs, (
            f"step {n} ({op_name}) で不一致 "
            f"(day {n // 24}, hour {n % 24}):\n  " + "\n  ".join(diffs[:12])
        )


def _pass(n: int) -> list[tuple[str, int, int, int, int, int]]:
    return [("PASS", 0, 0, E.MARKET_NONE, 0, 0)] * n


def test_goose_full_lifecycle_matches_engine() -> None:
    """COOP 建設 → 購入 → PICKUP → PLACE → 給餌 → 産出 → 収穫。

    GOOSE は first_yield_day=4 / interval=1 なので、短い台本で産出まで届く。
    """
    s: list[tuple[str, int, int, int, int, int]] = []
    # day0: shed 上で COOP 用に北へ出て建設、戻って GOOSE を買う
    s.append(("PASS", 0, 0, E.MARKET_BUY_ANIMAL, 0, 1))  # GOOSE を shed へ
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))  # (4,3)
    s.append(("BUILD_COOP", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("SOUTH", 0, 0, E.MARKET_NONE, 0, 0))  # (4,4) shed access
    s.append(("PICKUP", GOOSE_ITEM, 1, E.MARKET_NONE, 0, 0))  # 所持品へ
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PLACE", GOOSE_ITEM, 1, E.MARKET_NONE, 0, 0))  # COOP へ配置
    # 給餌には WHEAT が要る。買って拾って戻る。
    s.append(("SOUTH", 0, 0, E.MARKET_BUY_PRODUCT, 0, 3))
    s.append(("PICKUP", 0, 3, E.MARKET_NONE, 0, 0))  # WHEAT x3
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("FEED", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("CARE", 0, 0, E.MARKET_NONE, 0, 0))
    s += _pass(24 - len(s))  # day0 を閉じる

    # day1..5: 毎日 feed + care し、産出を待つ。位置は毎朝 spawn に戻る。
    for _ in range(5):
        d: list[tuple[str, int, int, int, int, int]] = []
        d.append(("PICKUP", 0, 1, E.MARKET_BUY_PRODUCT, 0, 1))
        d.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("FEED", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("CARE", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("COLLECT_FERTILIZER", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("HARVEST", 0, 0, E.MARKET_NONE, 0, 0))
        d += _pass(24 - len(d))
        s += d

    _run_script(s)


def test_animal_escapes_when_unfed_matches_engine() -> None:
    """2 日続けて給餌しないと家畜は逃げ、構造物だけが残る。"""
    s: list[tuple[str, int, int, int, int, int]] = []
    s.append(("PASS", 0, 0, E.MARKET_BUY_ANIMAL, 0, 1))
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("BUILD_COOP", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("SOUTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PICKUP", GOOSE_ITEM, 1, E.MARKET_NONE, 0, 0))
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PLACE", GOOSE_ITEM, 1, E.MARKET_NONE, 0, 0))
    s += _pass(24 - len(s))
    # 給餌せず 3 日放置 → 2 日目の日跨ぎで脱走するはず
    s += _pass(24 * 3)
    _run_script(s)


def test_cow_on_pasture_matches_engine() -> None:
    """PASTURE + COW。GOOSE と structure / interval が異なる経路を通す。"""
    s: list[tuple[str, int, int, int, int, int]] = []
    s.append(("PASS", 0, 0, E.MARKET_BUY_ANIMAL, 1, 1))  # COW
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("BUILD_PASTURE", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("SOUTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PICKUP", COW_ITEM, 1, E.MARKET_NONE, 0, 0))
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PLACE", COW_ITEM, 1, E.MARKET_NONE, 0, 0))
    s += _pass(24 - len(s))
    for _ in range(4):
        d: list[tuple[str, int, int, int, int, int]] = []
        d.append(("PICKUP", 0, 1, E.MARKET_BUY_PRODUCT, 0, 1))
        d.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("FEED", 0, 0, E.MARKET_NONE, 0, 0))
        d.append(("HARVEST", 0, 0, E.MARKET_NONE, 0, 0))
        d += _pass(24 - len(d))
        s += d
    _run_script(s)


def test_wrong_structure_rejects_animal_matches_engine() -> None:
    """COW を COOP に置こうとしても失敗し、shed drop に落ちる。"""
    s: list[tuple[str, int, int, int, int, int]] = []
    s.append(("PASS", 0, 0, E.MARKET_BUY_ANIMAL, 1, 1))  # COW
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("BUILD_COOP", 0, 0, E.MARKET_NONE, 0, 0))  # COOP は COW 用ではない
    s.append(("SOUTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PICKUP", COW_ITEM, 1, E.MARKET_NONE, 0, 0))
    s.append(("NORTH", 0, 0, E.MARKET_NONE, 0, 0))
    s.append(("PLACE", COW_ITEM, 1, E.MARKET_NONE, 0, 0))  # 失敗するはず
    s += _pass(24 - len(s))
    s += _pass(24)
    _run_script(s)
