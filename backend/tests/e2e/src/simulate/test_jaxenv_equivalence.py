"""The JAX port must agree with the official engine across the full rule set.

Unlike :mod:`simulate.fast`, which reuses the engine's own interpreter, this is
a reimplementation — so equivalence is not structural and has to be measured.

The comparison runs with ``weedSpawnChance=0`` and shops pushed past the
episode. Both are RNG-driven, and the engine's stream depends on how many tiles
happen to be empty (it draws once per empty tile per day), so reproducing the
exact sequence on-device is not practical. Disabling them **on both sides**
isolates everything deterministic: crops, watering, fertilizer, harvest,
animals, hired hands, land purchase, shed logistics, market pricing and
lockstep settlement, town-centre demand, and end-of-day accounting.

The RNG paths that are excluded here are covered by
``test_jaxenv_stochastic.py``, which asserts their distributions instead.

The headline test is :func:`test_random_action_streams_match`, a differential
fuzz over the whole op surface: it drives both engines with the same randomly
generated actions and compares full state every turn, so a divergence in any
rule shows up as a first-mismatch turn rather than a silent score difference.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

jax = pytest.importorskip("jax", reason="JAX is an optional dependency")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional dependency")

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import (  # noqa: E402
    kaggriculture as engine,
)

from simulate.jaxenv import env as E  # noqa: E402
from simulate.jaxenv.state import (  # noqa: E402
    ANIMAL_NAMES,
    CROP_NAMES,
    MAX_UNITS,
    PRODUCT_NAMES,
    initial_state,
    market_price,
)

#: Shops never unlock inside an episode at this interval.
NO_SHOPS = 10**6

#: A step compiled with the RNG paths disabled, matching the engine config below.
DETERMINISTIC_STEP = E.make_step(
    weed_chance=0.0,
    shop_sell_interval=4,
    center_sell_interval=24,
    shop_unlock_interval=NO_SHOPS,
)

_OP_TO_NAME = dict(enumerate(E.OP_NAMES))


def _make_engine(steps: int, seed: int) -> Any:
    return make(
        "kaggriculture",
        configuration={
            "episodeSteps": steps + 2,
            "seed": seed,
            "weedSpawnChance": 0.0,
            "townShopUnlockInterval": NO_SHOPS,
        },
        debug=False,
    )


def _unit_action(op: int, arg: int, qty: int) -> list[Any]:
    """Render one unit op as the engine's list form."""
    name = _OP_TO_NAME[op]
    if name == "PLANT":
        return ["PLANT", CROP_NAMES[arg]]
    if name == "PLACE":
        # Same index encoding as PICKUP: products first, then livestock. The
        # engine treats a non-animal item (or an animal that cannot be placed)
        # as a shed drop, so both halves need coverage.
        if arg < len(PRODUCT_NAMES):
            return ["PLACE", PRODUCT_NAMES[arg], qty]
        return ["PLACE", ANIMAL_NAMES[arg - len(PRODUCT_NAMES)], qty]
    if name == "PICKUP":
        # arg addresses products first, then livestock — the port's encoding.
        if arg < len(PRODUCT_NAMES):
            return ["PICKUP", PRODUCT_NAMES[arg], qty]
        return ["PICKUP", ANIMAL_NAMES[arg - len(PRODUCT_NAMES)], qty]
    return [name]


def _market_action(op: int, item: int, qty: int) -> list[Any] | None:
    """Render one market order as the engine's list form."""
    if op == E.MARKET_NONE:
        return None
    if op == E.MARKET_HIRE:
        return ["HIRE"]
    if op == E.MARKET_BUY_LAND:
        return ["BUY_LAND"]
    if op == E.MARKET_SELL:
        return ["SELL", PRODUCT_NAMES[item], qty]
    if op == E.MARKET_BUY_PRODUCT:
        return ["BUY_PRODUCT", PRODUCT_NAMES[item], qty]
    if op == E.MARKET_BUY_SEED:
        return ["BUY_SEED", CROP_NAMES[item], qty]
    if op == E.MARKET_BUY_ANIMAL:
        return ["BUY_ANIMAL", ANIMAL_NAMES[item], qty]
    raise AssertionError(f"unhandled market op {op}")


def _engine_snapshot(state: Any) -> dict[str, Any]:
    """Everything observable, flattened for comparison against the port."""
    obs0 = state[0].observation
    out: dict[str, Any] = {
        "market_inv": {k: obs0.market["inventory"][k] for k in PRODUCT_NAMES},
    }
    for p in (0, 1):
        farm = obs0.farms[p]
        priv = state[p].observation.private
        out[f"money{p}"] = float(farm["money"])
        out[f"seeds{p}"] = {c: priv["seeds"].get(c, 0) for c in CROP_NAMES}
        out[f"shed{p}"] = {i: priv["shed"].get(i, 0) for i in PRODUCT_NAMES}
        out[f"livestock{p}"] = {a: priv["shed"].get(a, 0) for a in ANIMAL_NAMES}
        out[f"hands{p}"] = len(farm["hands"])
        out[f"quads{p}"] = len(farm["unlocked_quadrants"])
        out[f"pos{p}"] = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        # Per-unit carried inventory, products and livestock together.
        carried = []
        for inv in priv["inventories"]:
            carried.append({k: v for k, v in sorted(inv.items()) if v})
        out[f"carried{p}"] = carried

        tiles: list[str] = []
        for row in farm["tiles"]:
            for tile in row:
                tiles.append(_render_tile(tile))
        out[f"tiles{p}"] = tiles
    return out


def _render_tile(tile: Any) -> str:
    """A tile's full comparable content as a string.

    Compares every field the port models, so a divergence in e.g.
    ``consecutive_unwatered`` fails the test rather than lurking until it
    changes a yield several days later.
    """
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    kind = tile.get("kind")
    if kind == "WEED":
        return "WEED"
    if kind == "PLANT":
        return (
            f"PLANT:{tile['crop']}:pd={tile['planted_day']}:"
            f"w={int(tile['watered_today'])}:cu={tile['consecutive_unwatered']}:"
            f"y={tile['yield_units']}:mls={tile['max_lifespan_step']}:"
            f"f={tile.get('fertilized_until_day', -1)}"
        )
    if "animal" in tile:
        return (
            f"ANIMAL:{tile['animal']}:pd={tile['placed_day']}:"
            f"y={tile['yield_units']}:cuf={tile['consecutive_unfed']}:"
            f"fed={int(tile['fed_today'])}:cared={int(tile['cared_today'])}:"
            f"fa={int(tile['fertilizer_available'])}:"
            f"pcb={tile.get('pending_care_bonus', 0)}"
        )
    return str(kind)


def _port_snapshot(state: Any, b: int = 0) -> dict[str, Any]:
    """The same view, read out of the batched arrays."""
    out: dict[str, Any] = {
        "market_inv": {
            name: int(state.market_inv[b, i]) for i, name in enumerate(PRODUCT_NAMES)
        },
    }
    for p in (0, 1):
        out[f"money{p}"] = float(state.money[b, p])
        out[f"seeds{p}"] = {
            c: int(state.seeds[b, p, i]) for i, c in enumerate(CROP_NAMES)
        }
        out[f"shed{p}"] = {
            i_name: int(state.shed[b, p, i]) for i, i_name in enumerate(PRODUCT_NAMES)
        }
        out[f"livestock{p}"] = {
            a: int(state.animal_shed[b, p, i]) for i, a in enumerate(ANIMAL_NAMES)
        }
        # Slot 0 is the main farmer, so active hands is one less.
        n_units = int(state.unit_active[b, p].sum())
        out[f"hands{p}"] = n_units - 1
        out[f"quads{p}"] = int(state.lands_bought[b, p]) + 1
        out[f"pos{p}"] = [
            (int(state.unit_x[b, p, u]), int(state.unit_y[b, p, u]))
            for u in range(n_units)
        ]
        carried = []
        for u in range(n_units):
            held = {
                name: int(state.carried[b, p, u, i])
                for i, name in enumerate(PRODUCT_NAMES)
                if int(state.carried[b, p, u, i])
            }
            held.update(
                {
                    name: int(state.carried_animals[b, p, u, i])
                    for i, name in enumerate(ANIMAL_NAMES)
                    if int(state.carried_animals[b, p, u, i])
                }
            )
            carried.append(dict(sorted(held.items())))
        out[f"carried{p}"] = carried

        tiles: list[str] = []
        for y in range(10):
            for x in range(10):
                tiles.append(_render_port_tile(state, b, p, y, x))
        out[f"tiles{p}"] = tiles
    return out


def _render_port_tile(state: Any, b: int, p: int, y: int, x: int) -> str:
    from simulate.jaxenv import state as S

    kind = int(state.kind[b, p, y, x])
    animal = int(state.animal[b, p, y, x])
    if animal != S.ANIMAL_NONE:
        return (
            f"ANIMAL:{ANIMAL_NAMES[animal]}:pd={int(state.placed_day[b, p, y, x])}:"
            f"y={int(state.yield_units[b, p, y, x])}:"
            f"cuf={int(state.consecutive_unfed[b, p, y, x])}:"
            f"fed={int(state.fed_today[b, p, y, x])}:"
            f"cared={int(state.cared_today[b, p, y, x])}:"
            f"fa={int(state.fertilizer_available[b, p, y, x])}:"
            f"pcb={int(state.pending_care_bonus[b, p, y, x])}"
        )
    if kind == S.KIND_EMPTY:
        return "EMPTY"
    if kind == S.KIND_LOCKED:
        return "LOCKED"
    if kind == S.KIND_WEED:
        return "WEED"
    if kind == S.KIND_COOP:
        return "COOP"
    if kind == S.KIND_PASTURE:
        return "PASTURE"
    crop = int(state.crop[b, p, y, x])
    return (
        f"PLANT:{CROP_NAMES[crop]}:pd={int(state.planted_day[b, p, y, x])}:"
        f"w={int(state.watered_today[b, p, y, x])}:"
        f"cu={int(state.consecutive_unwatered[b, p, y, x])}:"
        f"y={int(state.yield_units[b, p, y, x])}:"
        f"mls={int(state.max_lifespan_step[b, p, y, x])}:"
        f"f={int(state.fertilized_until_day[b, p, y, x])}"
    )


def _diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    out = []
    for key in sorted(a):
        if a[key] != b[key]:
            if key.startswith("tiles"):
                for i, (t1, t2) in enumerate(zip(a[key], b[key], strict=True)):
                    if t1 != t2:
                        out.append(f"{key}[y={i // 10},x={i % 10}]: {t1!r} != {t2!r}")
            else:
                out.append(f"{key}: {a[key]!r} != {b[key]!r}")
    return out


class ActionStream:
    """Generates the same action for both engines, in both representations."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def unit_ops(self, n_units: int) -> list[tuple[int, int, int]]:
        """(op, arg, qty) per unit slot."""
        rng = self.rng
        out = []
        for _ in range(n_units):
            op = rng.choice(
                # Weighted toward the ops that move the simulation forward,
                # with every op represented so nothing goes untested.
                [E.OP_PASS] * 2
                + [E.OP_NORTH, E.OP_SOUTH, E.OP_EAST, E.OP_WEST] * 2
                + [E.OP_PLANT] * 3
                + [E.OP_WATER] * 3
                + [E.OP_HARVEST] * 2
                + [E.OP_DIG]
                + [E.OP_BUILD_COOP, E.OP_BUILD_PASTURE]
                + [E.OP_PLACE] * 2
                + [E.OP_FEED, E.OP_CARE, E.OP_COLLECT_FERTILIZER]
                + [E.OP_FERTILIZE]
                + [E.OP_DROP] * 2
                + [E.OP_PICKUP] * 2
            )
            arg = 0
            qty = 0
            if op == E.OP_PLANT:
                arg = rng.randrange(len(CROP_NAMES))
            elif op == E.OP_PLACE:
                arg = rng.randrange(len(PRODUCT_NAMES) + len(ANIMAL_NAMES))
                qty = rng.randint(1, 3)
            elif op == E.OP_PICKUP:
                arg = rng.randrange(len(PRODUCT_NAMES) + len(ANIMAL_NAMES))
                qty = rng.randint(1, 3)
            out.append((op, arg, qty))
        return out

    def market_orders(self) -> list[tuple[int, int, int]]:
        rng = self.rng
        n = rng.randint(0, 3)
        out = []
        for _ in range(n):
            op = rng.choice(
                [E.MARKET_SELL] * 3
                + [E.MARKET_BUY_SEED] * 3
                + [E.MARKET_BUY_PRODUCT] * 2
                + [E.MARKET_BUY_ANIMAL]
                + [E.MARKET_HIRE] * 2
                + [E.MARKET_BUY_LAND]
            )
            item, qty = 0, 1
            if op == E.MARKET_SELL:
                item = rng.randrange(len(PRODUCT_NAMES))
                qty = rng.randint(1, 5)
            elif op == E.MARKET_BUY_SEED:
                item = rng.randrange(len(CROP_NAMES))
                qty = rng.randint(1, 3)
            elif op == E.MARKET_BUY_PRODUCT:
                # The engine only sells WHEAT and FERTILIZER.
                item = rng.choice(
                    [PRODUCT_NAMES.index("WHEAT"), PRODUCT_NAMES.index("FERTILIZER")]
                )
                qty = rng.randint(1, 3)
            elif op == E.MARKET_BUY_ANIMAL:
                item = rng.randrange(len(ANIMAL_NAMES))
                qty = 1
            out.append((op, item, qty))
        while len(out) < E.MAX_MARKET_ORDERS:
            out.append((E.MARKET_NONE, 0, 0))
        return out


def _run_differential(steps: int, seed: int) -> None:
    """Drive both engines with one random action stream; compare every turn."""
    rng = random.Random(seed)
    stream = ActionStream(rng)

    env = _make_engine(steps, seed)
    eng_state = env.state
    port = initial_state(1, seed=seed)

    for n in range(steps):
        # The engine's hand count is authoritative for how many unit actions
        # each player owes this turn; the port keeps the same count in
        # `unit_active`, which the previous turn's comparison already verified.
        per_player = []
        for p in (0, 1):
            n_hands = len(eng_state[0].observation.farms[p]["hands"])
            per_player.append(stream.unit_ops(1 + n_hands))
        markets = [stream.market_orders() for _ in (0, 1)]

        for p in (0, 1):
            units = per_player[p]
            orders = [
                a for a in (_market_action(*o) for o in markets[p]) if a is not None
            ]
            eng_state[p].action = {
                "farmer": _unit_action(*units[0]),
                "hands": [_unit_action(*u) for u in units[1:]],
                "market": orders,
            }

        engine.interpreter(eng_state, env)
        # The framework owns observation.step; the interpreter only reads it.
        eng_state[0].observation.step = n + 1

        op = jnp.zeros((1, 2, MAX_UNITS), dtype=jnp.int32)
        arg = jnp.zeros((1, 2, MAX_UNITS), dtype=jnp.int32)
        qty = jnp.zeros((1, 2, MAX_UNITS), dtype=jnp.int32)
        for p in (0, 1):
            for u, (o, a, q) in enumerate(per_player[p]):
                op = op.at[0, p, u].set(o)
                arg = arg.at[0, p, u].set(a)
                qty = qty.at[0, p, u].set(q)

        mop = jnp.array([[[o[0] for o in markets[p]] for p in (0, 1)]], dtype=jnp.int32)
        mitem = jnp.array(
            [[[o[1] for o in markets[p]] for p in (0, 1)]], dtype=jnp.int32
        )
        mqty = jnp.array(
            [[[o[2] for o in markets[p]] for p in (0, 1)]], dtype=jnp.int32
        )

        port = DETERMINISTIC_STEP(port, op, arg, qty, mop, mitem, mqty)

        got = _port_snapshot(port)
        want = _engine_snapshot(eng_state)
        diffs = _diff(want, got)
        assert not diffs, (
            f"divergence at step {n} (day {n // 24}, hour {n % 24}) "
            f"with seed {seed}:\n  " + "\n  ".join(diffs[:20])
        )


@pytest.mark.parametrize("seed", range(1, 11))
def test_random_action_streams_match(seed: int) -> None:
    """Differential fuzz over the full op surface, three days per seed."""
    _run_differential(steps=24 * 3, seed=seed)


@pytest.mark.parametrize("seed", [11, 12])
def test_long_random_streams_match(seed: int) -> None:
    """A longer horizon, to reach ongoing-crop production and animal payouts."""
    _run_differential(steps=24 * 14, seed=seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [21])
def test_full_season_matches(seed: int) -> None:
    """A whole 30-day season, so late-season dynamics are covered too.

    Strawberry and melon only reach first yield around day 10, and ongoing
    crops hit their max-yield lifespan later still, so the last third of the
    season exercises transitions the shorter runs never see.
    """
    _run_differential(steps=24 * 30, seed=seed)


def test_market_price_matches_official_across_inventory_range() -> None:
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
    shape = (2, 2, MAX_UNITS)
    mshape = (2, 2, E.MAX_MARKET_ORDERS)
    zeros = jnp.zeros(shape, dtype=jnp.int32)
    mop = jnp.zeros(mshape, dtype=jnp.int32).at[0, 0, 0].set(E.MARKET_BUY_SEED)
    mqty = jnp.zeros(mshape, dtype=jnp.int32).at[0, 0, 0].set(3)

    out = E.step(
        state, zeros, zeros, zeros, mop, jnp.zeros(mshape, dtype=jnp.int32), mqty
    )

    assert int(out.seeds[0, 0, 0]) == 3
    assert int(out.seeds[1, 0, 0]) == 0
    assert float(out.money[0, 0]) < float(out.money[1, 0])


def test_batching_does_not_change_dynamics() -> None:
    """The same scripted episode must give the same result at any batch size."""
    steps = 24 * 4

    def rollout(batch: int) -> tuple[float, ...]:
        state = initial_state(batch, seed=7)
        for n in range(steps):
            hour = n % 24
            op_code = E.OP_PASS
            arg_code = 0
            if hour == 1:
                op_code, arg_code = E.OP_PLANT, 0
            elif hour in (2, 3):
                op_code = E.OP_WATER
            elif hour == 5:
                op_code = E.OP_HARVEST
            elif hour == 7:
                op_code = E.OP_DROP

            op = jnp.zeros((batch, 2, MAX_UNITS), dtype=jnp.int32)
            op = op.at[:, :, 0].set(op_code)
            arg = jnp.zeros((batch, 2, MAX_UNITS), dtype=jnp.int32)
            arg = arg.at[:, :, 0].set(arg_code)
            qty = jnp.zeros((batch, 2, MAX_UNITS), dtype=jnp.int32)

            mshape = (batch, 2, E.MAX_MARKET_ORDERS)
            mop = jnp.zeros(mshape, dtype=jnp.int32)
            mitem = jnp.zeros(mshape, dtype=jnp.int32)
            mqty = jnp.zeros(mshape, dtype=jnp.int32)
            if hour == 0:
                mop = mop.at[:, :, 0].set(E.MARKET_BUY_SEED)
                mqty = mqty.at[:, :, 0].set(1)
            elif hour == 8:
                mop = mop.at[:, :, 0].set(E.MARKET_SELL)
                mqty = mqty.at[:, :, 0].set(5)

            state = DETERMINISTIC_STEP(state, op, arg, qty, mop, mitem, mqty)
        return tuple(float(state.money[b, 0]) for b in range(batch))

    single = rollout(1)[0]
    for money in rollout(16):
        assert money == single
