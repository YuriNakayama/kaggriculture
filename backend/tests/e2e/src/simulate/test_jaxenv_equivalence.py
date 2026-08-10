"""The JAX port must agree with the official engine on its supported subset.

Unlike :mod:`simulate.fast`, which reuses the engine's own interpreter, this is
a reimplementation — so equivalence is not structural and has to be measured.

Comparison runs with ``weedSpawnChance=0`` and shops pushed past the episode.
Both are driven by a Python ``random.Random`` stream whose draw count depends on
how many tiles are empty (see ``jaxenv/town.py``), so reproducing them on the
device is not practical. Disabling them **on both sides** isolates the
deterministic core: crops, watering, harvest, market pricing, town-centre
demand, and end-of-day accounting.

Anything outside that subset (hands, animals, land, fertilizer) is not
implemented and not asserted here.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX is an optional dependency")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional dependency")

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import (  # noqa: E402
    kaggriculture as engine,
)

from simulate.jaxenv import env as E  # noqa: E402
from simulate.jaxenv.state import initial_state, market_price  # noqa: E402

#: Shops never unlock inside an episode at this interval.
NO_SHOPS = 10**6

_OP_TO_NAME = {
    E.OP_PASS: "PASS",
    E.OP_PLANT: "PLANT",
    E.OP_WATER: "WATER",
    E.OP_HARVEST: "HARVEST",
    E.OP_DIG: "DIG",
}


def _script(step: int) -> tuple[int, int, int]:
    """A wheat loop keyed on turn-of-day: (unit op, market op, quantity)."""
    hour = step % 24
    if hour == 0:
        return E.OP_PASS, E.MARKET_BUY_SEED, 1
    if hour == 1:
        return E.OP_PLANT, E.MARKET_NONE, 0
    if hour in (2, 3):
        return E.OP_WATER, E.MARKET_NONE, 0
    if hour == 5:
        return E.OP_HARVEST, E.MARKET_NONE, 0
    if hour == 6:
        return E.OP_PASS, E.MARKET_SELL, 5
    return E.OP_PASS, E.MARKET_NONE, 0


def _run_official(steps: int, seed: int) -> list[tuple[float, int, int, int]]:
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": steps + 2,
            "seed": seed,
            "weedSpawnChance": 0.0,
            "townShopUnlockInterval": NO_SHOPS,
        },
        debug=False,
    )
    state = env.state
    trace = []

    for n in range(steps):
        op, mop, qty = _script(n)
        name = _OP_TO_NAME[op]
        for player in (0, 1):
            action: dict[str, object] = {
                "farmer": ["PLANT", "WHEAT"] if name == "PLANT" else [name],
                "hands": [],
                "market": [],
            }
            if mop == E.MARKET_BUY_SEED:
                action["market"] = [["BUY_SEED", "WHEAT", qty]]
            elif mop == E.MARKET_SELL:
                action["market"] = [["SELL", "WHEAT", qty]]
            state[player].action = action

        engine.interpreter(state, env)
        # The framework owns observation.step; the interpreter only reads it.
        state[0].observation.step = n + 1

        obs = state[0].observation
        trace.append(
            (
                float(obs.farms[0]["money"]),
                int(obs.private["seeds"]["WHEAT"]),
                int(obs.private["shed"].get("WHEAT", 0)),
                int(obs.market["inventory"]["WHEAT"]),
            )
        )
    return trace


def _run_jax(steps: int) -> list[tuple[float, int, int, int]]:
    state = initial_state(1)
    trace = []
    for n in range(steps):
        op, mop, qty = _script(n)
        shape = (1, 2)
        state = E.step(
            state,
            jnp.full(shape, op, dtype=jnp.int32),
            jnp.zeros(shape, dtype=jnp.int32),
            jnp.full(shape, mop, dtype=jnp.int32),
            jnp.zeros(shape, dtype=jnp.int32),
            jnp.full(shape, qty, dtype=jnp.int32),
        )
        trace.append(
            (
                float(state.money[0, 0]),
                int(state.seeds[0, 0, 0]),
                int(state.shed[0, 0, 0]),
                int(state.market_inv[0, 0]),
            )
        )
    return trace


@pytest.mark.parametrize("days", [3, 8, 15])
def test_wheat_loop_matches_official_engine(days: int) -> None:
    steps = 24 * days

    official = _run_official(steps, seed=1)
    ported = _run_jax(steps)

    assert len(official) == len(ported) == steps
    mismatches = [
        (i, o, j)
        for i, (o, j) in enumerate(zip(official, ported, strict=True))
        if o != j
    ]
    assert not mismatches, f"first divergence: {mismatches[0]}"


def test_market_price_matches_official_across_inventory_range() -> None:
    from simulate.jaxenv.state import PRODUCT_NAMES

    inventories = [0, 1, 5_000, 9_999, 10_000, 10_001, 15_000, 30_000, 100_000]

    for inv in inventories:
        got = market_price(jnp.array([inv] * len(PRODUCT_NAMES), dtype=jnp.int32))
        for i, name in enumerate(PRODUCT_NAMES):
            assert int(got[i]) == engine.market_price(name, inv), (
                f"{name} at inventory {inv}"
            )


def test_batched_environments_are_independent() -> None:
    # Two environments given different actions must not leak into each other.
    state = initial_state(2)
    op = jnp.array([[E.OP_PASS, E.OP_PASS], [E.OP_PASS, E.OP_PASS]], dtype=jnp.int32)
    mop = jnp.array(
        [[E.MARKET_BUY_SEED, E.MARKET_NONE], [E.MARKET_NONE, E.MARKET_NONE]],
        dtype=jnp.int32,
    )
    qty = jnp.array([[3, 0], [0, 0]], dtype=jnp.int32)
    zeros = jnp.zeros((2, 2), dtype=jnp.int32)

    out = E.step(state, op, zeros, mop, zeros, qty)

    assert int(out.seeds[0, 0, 0]) == 3
    assert int(out.seeds[1, 0, 0]) == 0
    assert float(out.money[0, 0]) < float(out.money[1, 0])


def test_batching_does_not_change_dynamics() -> None:
    # The same scripted episode must give the same result at any batch size.
    steps = 24 * 4
    single = _run_jax(steps)

    state = initial_state(16)
    for n in range(steps):
        op, mop, qty = _script(n)
        shape = (16, 2)
        state = E.step(
            state,
            jnp.full(shape, op, dtype=jnp.int32),
            jnp.zeros(shape, dtype=jnp.int32),
            jnp.full(shape, mop, dtype=jnp.int32),
            jnp.zeros(shape, dtype=jnp.int32),
            jnp.full(shape, qty, dtype=jnp.int32),
        )

    final_money = float(single[-1][0])
    for b in range(16):
        assert float(state.money[b, 0]) == final_money
