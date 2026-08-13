"""GPU 上での忠実性検証とスループット計測。

CPU 側では公式 engine との等価性を seed 100本 × 最大30日で確認済みだが、
それは CPU の話であって GPU で同じ結果になる保証はない。float32 の丸め、
scatter の実装差、XLA の最適化などで値がずれる可能性がある。

そこで2つを続けて行う:

1. **忠実性** — 固定 action 列を15日流し、12個の状態配列の SHA256 を
   CPU で得た期待値と突き合わせる。1つでも不一致なら GPU の結果は信用できない。
2. **スループット** — batch を振って env-steps/s を測る。

``python -m simulate.jaxenv.gpu_bench`` で実行する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

import jax
import jax.numpy as jnp

from . import env as E
from .state import MAX_UNITS, initial_state

#: CPU で公式 engine と一致することを確認済みの状態ダイジェスト。
#: ローカル (M2 Max, CPU backend) で再現を確認している。
EXPECTED_DIGEST: dict[str, str] = {
    "animal": "e202c0cf84a7cd41",
    "animal_shed": "3158570784fcab36",
    "carried": "2d07a41ae9927700",
    "crop": "ee506e29a41b3a8b",
    "kind": "5c22a5e65fe57d08",
    "lands_bought": "3110ab1939d07217",
    "market_inv": "7cf5da2bd2566757",
    "money": "2b4ff02479a20654",
    "seeds": "546dba8e52359fd4",
    "shed": "460cb091b241731c",
    "unit_active": "2435b3bb50cad942",
    "yield_units": "41cc89efb05c356d",
}

#: 忠実性チェックは RNG を無効化した決定的な設定で行う (CPU 側と同条件)。
_DETERMINISTIC = dict(
    weed_chance=0.0,
    shop_sell_interval=4,
    center_sell_interval=24,
    shop_unlock_interval=10**6,
)

#: 公式 env.run のスループット (ローカル実測)。速度比の分母。
OFFICIAL_STEPS_PER_SEC = 768


def _fidelity_state(step_fn: Any, batch: int = 8, days: int = 15) -> Any:
    """固定 action 列を流した後の状態を返す。CPU 側と完全に同一の手順。"""
    rng = jax.random.PRNGKey(0)
    state = initial_state(batch, seed=0)
    unit_shape = (batch, 2, MAX_UNITS)
    market_shape = (batch, 2, E.MAX_MARKET_ORDERS)

    for n in range(days * 24):
        key = jax.random.fold_in(rng, n)
        hour = n % 24
        # unit 0 は wheat loop を回し、残りは無作為に全 op を叩く。
        loop_op = {
            1: E.OP_PLANT, 2: E.OP_WATER, 3: E.OP_WATER,
            5: E.OP_HARVEST, 7: E.OP_DROP,
        }.get(hour, E.OP_PASS)
        op = jax.random.randint(
            jax.random.fold_in(key, 1), unit_shape, 0, E.N_OPS
        ).at[:, :, 0].set(loop_op)
        arg = jax.random.randint(
            jax.random.fold_in(key, 2), unit_shape, 0, 12
        ).at[:, :, 0].set(0)
        qty = jax.random.randint(jax.random.fold_in(key, 3), unit_shape, 0, 4)

        mop = jax.random.randint(jax.random.fold_in(key, 4), market_shape, 0, 7)
        mop = mop.at[:, :, 0].set(
            E.MARKET_BUY_SEED if hour == 0
            else (E.MARKET_SELL if hour == 8 else E.MARKET_NONE)
        )
        mitem = jax.random.randint(jax.random.fold_in(key, 5), market_shape, 0, 9)
        mqty = jax.random.randint(jax.random.fold_in(key, 6), market_shape, 0, 4)

        state = step_fn(state, op, arg, qty, mop, mitem, mqty)
    return state


def check_fidelity(step_fn: Any) -> tuple[bool, dict[str, str]]:
    """GPU 上の状態が CPU 検証済みの値と一致するか。"""
    state = _fidelity_state(step_fn)
    got: dict[str, str] = {}
    for name in EXPECTED_DIGEST:
        arr = jax.device_get(getattr(state, name))
        got[name] = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return got == EXPECTED_DIGEST, got


def _rollout_fn(step_fn: Any, batch: int) -> Any:
    """1 シーズン分を 1 つの fori_loop に畳んだ関数を返す。

    Python 側の per-step dispatch を測ってしまわないよう、ループごと jit する。
    """

    def body(i: jnp.ndarray, state: Any) -> Any:
        hour = jnp.remainder(i, 24)
        op_code = jnp.where(
            hour == 1, E.OP_PLANT,
            jnp.where((hour == 2) | (hour == 3), E.OP_WATER,
                      jnp.where(hour == 5, E.OP_HARVEST,
                                jnp.where(hour == 7, E.OP_DROP, E.OP_PASS))))
        mop_code = jnp.where(
            hour == 0, E.MARKET_BUY_SEED,
            jnp.where(hour == 8, E.MARKET_SELL, E.MARKET_NONE))
        qty_val = jnp.where(hour == 0, 1, jnp.where(hour == 8, 5, 0))

        unit_shape = (batch, 2, MAX_UNITS)
        market_shape = (batch, 2, E.MAX_MARKET_ORDERS)
        zeros_u = jnp.zeros(unit_shape, dtype=jnp.int32)
        zeros_m = jnp.zeros(market_shape, dtype=jnp.int32)
        return step_fn(
            state,
            zeros_u.at[:, :, 0].set(op_code), zeros_u, zeros_u,
            zeros_m.at[:, :, 0].set(mop_code), zeros_m,
            zeros_m.at[:, :, 0].set(qty_val),
        )

    return jax.jit(lambda s, n: jax.lax.fori_loop(0, n, body, s))


def bench(step_fn: Any, batch: int, steps: int) -> dict[str, float]:
    """1 構成の計測。compile は 1 回目、計測は 2 回目の wall clock。"""
    rollout = _rollout_fn(step_fn, batch)
    state = initial_state(batch, seed=0)
    n = jnp.int32(steps)

    t0 = time.perf_counter()
    jax.block_until_ready(rollout(state, n))
    compile_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    jax.block_until_ready(rollout(state, n))
    wall = time.perf_counter() - t0

    sps = batch * steps / wall
    return {
        "batch": batch, "wall_sec": wall, "compile_sec": compile_sec,
        "steps_per_sec": sps, "vs_official": sps / OFFICIAL_STEPS_PER_SEC,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=720, help="1 シーズン = 720")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 64, 1024, 8192, 65536])
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    device = jax.devices()[0]
    print(f"device: {device} ({device.platform})", flush=True)
    print(f"jax: {jax.__version__}", flush=True)

    step_fn = E.make_step(**_DETERMINISTIC)

    print("\n=== 忠実性 (CPU 検証済みダイジェストとの突合) ===", flush=True)
    ok, got = check_fidelity(step_fn)
    for name, want in EXPECTED_DIGEST.items():
        mark = "OK " if got[name] == want else "MISMATCH"
        print(f"  {mark} {name:16s} {got[name]}", flush=True)
    print(f"  --> {'ALL 12 MATCH' if ok else 'MISMATCH あり'}", flush=True)

    print("\n=== スループット ===", flush=True)
    header = f"{'batch':>7} {'wall(s)':>9} {'compile(s)':>11} {'env-steps/s':>14} {'vs official':>12}"
    print(header, flush=True)
    rows = []
    for b in args.batches:
        try:
            r = bench(step_fn, b, args.steps)
            rows.append(r)
            print(f"{b:>7} {r['wall_sec']:>9.3f} {r['compile_sec']:>11.1f} "
                  f"{r['steps_per_sec']:>14,.0f} {r['vs_official']:>11,.0f}x",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{b:>7}  FAILED {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            break

    payload = {
        "device": str(device), "platform": device.platform,
        "jax": jax.__version__, "fidelity_ok": ok, "digest": got, "bench": rows,
    }
    print("\nRESULT_JSON " + json.dumps(payload), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
