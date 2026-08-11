"""Batched, jit-compiled Kaggriculture step — full rule coverage.

Every rule is expressed as a masked array update so one instruction stream
drives all ``B`` environments. This is a complete port of the official
interpreter: all unit ops, all market ops, animals, hired hands, land purchase,
fertilizer, town shops, and weed spawning are modelled.

Two structural differences from the engine, both forced by ``jit`` needing
static shapes, neither of which changes the dynamics:

- **Units are fixed-width.** Each player has ``MAX_UNITS`` slots with an active
  mask instead of a growing ``hands`` list. Hiring activates the next slot.
- **Market orders are fixed-width.** Each player submits exactly
  ``MAX_MARKET_ORDERS`` slots per turn; unused slots carry ``MARKET_NONE``.
  Per-order quantity is bounded by ``MAX_ORDER_UNITS``.

The RNG differs deliberately. The engine draws from ``random.Random`` seeded per
day, consuming a draw per empty tile, so the stream depends on board occupancy.
Reproducing that on-device is not practical, so weed spawn and shop unlocks use
a per-environment JAX key. The *distributions* match (same spawn probability,
same uniform shop choice, same caps); the exact sequence does not. Equivalence
against the engine is therefore asserted with RNG disabled, and the RNG paths
are tested against their distributions separately.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .state import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_NONE,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_NONE,
    CROP_ONGOING,
    CROP_SEED_COST,
    DEFAULT_WEED_SPAWN_CHANCE,
    KIND_COOP,
    KIND_EMPTY,
    KIND_LOCKED,
    KIND_PASTURE,
    KIND_PLANT,
    KIND_WEED,
    LAND_ORDER,
    LAND_PRICES,
    MAX_SHOP_INSTANCES,
    MAX_UNITS,
    N_ANIMALS,
    N_CROPS,
    N_PRODUCTS,
    N_SHOPS,
    PRODUCT_FERTILIZER,
    PRODUCT_WHEAT,
    SHED_CAPACITY,
    SHOP_DEMAND,
    TOWN_CENTER_MASK,
    TURNS_PER_DAY,
    EnvState,
    is_shed_adjacent,
    market_price,
    quadrant_of,
)

# --- Unit ops -------------------------------------------------------------

OP_PASS = 0
OP_NORTH = 1
OP_SOUTH = 2
OP_EAST = 3
OP_WEST = 4
OP_PLANT = 5
OP_WATER = 6
OP_HARVEST = 7
OP_DIG = 8
OP_BUILD_COOP = 9
OP_BUILD_PASTURE = 10
OP_PLACE = 11
OP_FEED = 12
OP_CARE = 13
OP_COLLECT_FERTILIZER = 14
OP_FERTILIZE = 15
OP_DROP = 16
OP_PICKUP = 17

OP_NAMES = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PLANT",
    "WATER",
    "HARVEST",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "PLACE",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "FERTILIZE",
    "DROP",
    "PICKUP",
)
N_OPS = len(OP_NAMES)

# --- Market ops -----------------------------------------------------------

MARKET_NONE = 0
MARKET_SELL = 1
MARKET_BUY_SEED = 2
MARKET_BUY_PRODUCT = 3
MARKET_BUY_ANIMAL = 4
MARKET_HIRE = 5
MARKET_BUY_LAND = 6

MARKET_OP_NAMES = (
    "NONE",
    "SELL",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "HIRE",
    "BUY_LAND",
)

#: Orders per player per turn, matching ``maxMarketOrdersPerTurn``.
MAX_MARKET_ORDERS = 10

#: Units a single order may move. The engine's per-unit lockstep loop is a real
#: serial dependence, so it becomes a fixed-trip ``fori_loop``; this bounds it.
#: The shed holds 100 items, so no single legal order can exceed that.
MAX_ORDER_UNITS = 100

#: (dx, dy) per op code; y grows downward. Non-move ops contribute zero.
_DX = jnp.zeros(N_OPS, dtype=jnp.int32).at[OP_EAST].set(1).at[OP_WEST].set(-1)
_DY = jnp.zeros(N_OPS, dtype=jnp.int32).at[OP_SOUTH].set(1).at[OP_NORTH].set(-1)


#: Fibonacci hire costs, ``_fib(0)=1, _fib(1)=1, _fib(2)=2, ...``.
def _fib_table(n: int) -> list[int]:
    out: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


HIRE_COST = jnp.array(_fib_table(MAX_UNITS + 1), dtype=jnp.int32)


# --- Tile gather / scatter helpers ---------------------------------------
#
# Planes are (B, 2, H, W) and unit coordinates are (B, 2, U), so every access
# is a per-unit gather or scatter. Scatters over units are sequential in intent
# (two units on the same tile both act), and `.at[].set` with duplicate indices
# has undefined resolution order — so ops that could collide use `.add` or an
# explicit max, never a raw `.set` over the unit axis.


def _gather_unit(plane: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Read ``plane[b, p, y[b,p,u], x[b,p,u]]`` -> (B, 2, U)."""
    b, p, _u = x.shape
    bi = jnp.arange(b)[:, None, None]
    pi = jnp.arange(p)[None, :, None]
    return plane[bi, pi, y, x]


def _scatter_unit_add(
    plane: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, delta: jnp.ndarray
) -> jnp.ndarray:
    """Accumulate ``delta`` into ``plane`` at each unit's tile.

    ``.add`` is well defined when several units target the same tile, which is
    exactly what makes it the safe primitive for multi-unit updates.
    """
    b, p, _u = x.shape
    bi = jnp.arange(b)[:, None, None]
    pi = jnp.arange(p)[None, :, None]
    return plane.at[bi, pi, y, x].add(delta)


def _scatter_unit_max(
    plane: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, value: jnp.ndarray
) -> jnp.ndarray:
    """Take the elementwise max of ``plane`` and ``value`` at each unit's tile.

    Order-independent, so it resolves collisions deterministically. Used for
    flag-setting ops (WATER, FEED, CARE) where two units acting on one tile
    must produce the same result as one unit acting.
    """
    b, p, _u = x.shape
    bi = jnp.arange(b)[:, None, None]
    pi = jnp.arange(p)[None, :, None]
    return plane.at[bi, pi, y, x].max(value)


def _unit_mask_plane(
    shape: tuple[int, ...], x: jnp.ndarray, y: jnp.ndarray, flag: jnp.ndarray
) -> jnp.ndarray:
    """Count, per tile, how many units assert ``flag`` there."""
    zeros = jnp.zeros(shape, dtype=jnp.int32)
    return _scatter_unit_add(zeros, x, y, flag.astype(jnp.int32))


def _first_claimant(
    shape: tuple[int, ...], x: jnp.ndarray, y: jnp.ndarray, wants: jnp.ndarray
) -> jnp.ndarray:
    """Restrict ``wants`` to the lowest-index unit standing on each tile.

    Several ops are one-per-tile: two units on the same empty tile cannot both
    plant it, and the engine — applying units in order — serves the first and
    silently no-ops the rest. Reproducing that means finding, per tile, the
    minimum index among the units requesting it.

    The sentinel base matters. Scattering ``min`` onto a zeroed plane would
    make every tile's winner 0, so only unit 0 could ever act; the plane must
    start above any real unit index.
    """
    unit_ids = jnp.broadcast_to(jnp.arange(MAX_UNITS), wants.shape)
    sentinel = MAX_UNITS + 1
    b = x.shape[0]
    idx = (
        jnp.arange(b)[:, None, None],
        jnp.arange(2)[None, :, None],
        y,
        x,
    )
    winner = (
        jnp.full(shape, sentinel, dtype=jnp.int32)
        .at[idx]
        .min(jnp.where(wants, unit_ids, sentinel))
    )
    return wants & (unit_ids == _gather_unit(winner, x, y))


# --- Movement -------------------------------------------------------------


def _apply_moves(state: EnvState, op: jnp.ndarray) -> EnvState:
    """Move units, clamping at the board edge (off-board moves are no-ops).

    Movement onto LOCKED tiles is allowed, matching the engine: a hand can
    spawn on a locked shed-access tile and must be able to walk off it.
    """
    nx = state.unit_x + _DX[op]
    ny = state.unit_y + _DY[op]
    in_bounds = (nx >= 0) & (nx < BOARD_SIZE) & (ny >= 0) & (ny < BOARD_SIZE)
    live = state.unit_active == 1
    ok = in_bounds & live
    return state._replace(
        unit_x=jnp.where(ok, nx, state.unit_x),
        unit_y=jnp.where(ok, ny, state.unit_y),
    )


# --- Seed-gated planting --------------------------------------------------


def _resolve_plant_demand(
    state: EnvState, op: jnp.ndarray, crop_arg: jnp.ndarray
) -> jnp.ndarray:
    """Drop ALL PLANT requests for a crop whose demand exceeds seeds held.

    Mirrors the engine's atomic pre-check: it counts requests per crop across
    the farmer and every hand, and if the total exceeds the seed count it
    blocks every one of them rather than serving a prefix.
    """
    wants = (op == OP_PLANT) & (state.unit_active == 1)
    # (B, 2, N_CROPS) demand per crop.
    onehot = jax.nn.one_hot(crop_arg, N_CROPS, dtype=jnp.int32)
    demand = (wants[..., None] * onehot).sum(axis=2)
    blocked = demand > state.seeds  # (B, 2, N_CROPS)
    blocked_for_unit = jnp.take_along_axis(blocked, crop_arg, axis=2)
    return jnp.where(wants & blocked_for_unit, OP_PASS, op)


# --- Tile ops -------------------------------------------------------------


def _apply_plant(
    state: EnvState, op: jnp.ndarray, crop_arg: jnp.ndarray, day: jnp.ndarray
) -> EnvState:
    """PLANT on an empty owned tile, consuming one seed.

    Seed demand is pre-validated by :func:`_resolve_plant_demand`, so any unit
    still requesting PLANT here has a seed reserved for it. Two units standing
    on the same empty tile is the one remaining collision: both would plant.
    The engine serves the first and no-ops the second, so ties are broken by
    unit index.
    """
    x, y = state.unit_x, state.unit_y
    kind_here = _gather_unit(state.kind, x, y)
    live = state.unit_active == 1

    wants = (op == OP_PLANT) & live & (kind_here == KIND_EMPTY)

    # Break same-tile ties: only the lowest-index unit on a tile may plant.
    ok = _first_claimant(state.kind.shape, x, y, wants)

    day_b = jnp.broadcast_to(day[:, None, None], ok.shape)
    ongoing = CROP_ONGOING[crop_arg]
    lifespan = jnp.where(
        ongoing == 1,
        -1,
        (day_b + CROP_MAX_YIELD_DAY[crop_arg] + 1) * TURNS_PER_DAY,
    )

    flag = ok.astype(jnp.int32)
    shape = state.kind.shape
    planted_mask = _unit_mask_plane(shape, x, y, ok) > 0

    def place(
        plane: jnp.ndarray, value: jnp.ndarray, base: int | jnp.ndarray
    ) -> jnp.ndarray:
        # Zero the tile, then add the planted value: `.add` is collision-safe
        # and `ok` guarantees at most one unit per tile contributes.
        cleared = jnp.where(planted_mask, base, plane)
        return _scatter_unit_add(cleared, x, y, flag * (value - base))

    # consecutive_unwatered starts at 1: the planting day already counts as
    # unwatered, so a new plant must be watered the same day or it weeds over.
    return state._replace(
        kind=place(state.kind, jnp.full_like(flag, KIND_PLANT), KIND_EMPTY),
        crop=place(state.crop, crop_arg, CROP_NONE),
        planted_day=place(state.planted_day, day_b, 0),
        watered_today=place(state.watered_today, jnp.zeros_like(flag), 0),
        consecutive_unwatered=place(
            state.consecutive_unwatered, jnp.ones_like(flag), 0
        ),
        yield_units=place(state.yield_units, jnp.where(ongoing == 1, 0, 1), 0),
        max_lifespan_step=place(state.max_lifespan_step, lifespan, -1),
        fertilized_until_day=place(
            state.fertilized_until_day, jnp.full_like(flag, -1), -1
        ),
        seeds=state.seeds
        - (flag[..., None] * jax.nn.one_hot(crop_arg, N_CROPS, dtype=jnp.int32)).sum(
            axis=2
        ),
    )


def _apply_water(state: EnvState, op: jnp.ndarray, day: jnp.ndarray) -> EnvState:
    """WATER a plant, adding a yield unit inside the bonus window.

    ``watered_today`` gates the bonus, so two units watering the same tile in
    one turn award it once — the mask is computed per tile, not per unit.
    """
    x, y = state.unit_x, state.unit_y
    kind_here = _gather_unit(state.kind, x, y)
    watered_here = _gather_unit(state.watered_today, x, y)
    live = state.unit_active == 1

    wants = (op == OP_WATER) & live & (kind_here == KIND_PLANT) & (watered_here == 0)

    # Per-tile: was this tile watered by anyone this turn?
    newly = _unit_mask_plane(state.kind.shape, x, y, wants) > 0
    is_plant = state.kind == KIND_PLANT
    newly = newly & is_plant & (state.watered_today == 0)

    crop_idx = jnp.maximum(state.crop, 0)
    day_p = day[:, None, None, None]
    age = day_p - state.planted_day
    max_yield_day = CROP_MAX_YIELD_DAY[crop_idx]
    window_start = (max_yield_day + 1) // 2
    in_window = (age >= window_start) & (age <= max_yield_day)
    # Fertilizer doubles the watering bonus while active.
    bonus = jnp.where(state.fertilized_until_day >= day_p, 2, 1)
    gains = newly & in_window & (CROP_ONGOING[crop_idx] == 0)

    return state._replace(
        watered_today=jnp.where(newly, 1, state.watered_today),
        yield_units=jnp.where(
            gains,
            jnp.minimum(CROP_MAX_YIELD[crop_idx], state.yield_units + bonus),
            state.yield_units,
        ),
    )


def _apply_harvest(state: EnvState, op: jnp.ndarray, day: jnp.ndarray) -> EnvState:
    """HARVEST a mature plant or an animal's held product.

    The whole ``yield_units`` stack goes to one unit. When two units harvest
    the same tile the engine gives everything to the first and nothing to the
    second, so the lowest unit index wins.
    """
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    kind_here = _gather_unit(state.kind, x, y)
    crop_here = _gather_unit(state.crop, x, y)
    animal_here = _gather_unit(state.animal, x, y)
    planted = _gather_unit(state.planted_day, x, y)
    yields = _gather_unit(state.yield_units, x, y)

    crop_idx = jnp.maximum(crop_here, 0)
    day_b = jnp.broadcast_to(day[:, None, None], kind_here.shape)
    mature = (day_b - planted) >= CROP_FIRST_YIELD_DAY[crop_idx]

    is_plant = kind_here == KIND_PLANT
    has_animal = animal_here != ANIMAL_NONE
    wants = (
        (op == OP_HARVEST) & live & (yields > 0) & ((is_plant & mature) | has_animal)
    )

    # Lowest unit index on a contested tile takes the yield.
    ok = _first_claimant(state.kind.shape, x, y, wants)

    product = jnp.where(
        has_animal, ANIMAL_PRODUCT[jnp.maximum(animal_here, 0)], crop_idx
    )
    picked = jnp.where(ok, yields, 0)
    gain = picked[..., None] * jax.nn.one_hot(product, N_PRODUCTS, dtype=jnp.int32)

    # Non-ongoing crops vanish once picked; animals and ongoing crops persist.
    clears = ok & is_plant & (CROP_ONGOING[crop_idx] == 0)
    cleared_tiles = _unit_mask_plane(state.kind.shape, x, y, clears) > 0
    emptied = _unit_mask_plane(state.kind.shape, x, y, ok) > 0

    return state._replace(
        kind=jnp.where(cleared_tiles, KIND_EMPTY, state.kind),
        crop=jnp.where(cleared_tiles, CROP_NONE, state.crop),
        max_lifespan_step=jnp.where(cleared_tiles, -1, state.max_lifespan_step),
        fertilized_until_day=jnp.where(cleared_tiles, -1, state.fertilized_until_day),
        yield_units=jnp.where(emptied, 0, state.yield_units),
        carried=state.carried + gain,
    )


def _apply_dig(state: EnvState, op: jnp.ndarray) -> EnvState:
    """DIG clears a plant, weed, or empty structure. A placed animal blocks it."""
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    kind_here = _gather_unit(state.kind, x, y)
    animal_here = _gather_unit(state.animal, x, y)

    diggable = (
        (kind_here == KIND_PLANT)
        | (kind_here == KIND_WEED)
        | (kind_here == KIND_COOP)
        | (kind_here == KIND_PASTURE)
    )
    ok = (op == OP_DIG) & live & diggable & (animal_here == ANIMAL_NONE)
    hit = _unit_mask_plane(state.kind.shape, x, y, ok) > 0

    return state._replace(
        kind=jnp.where(hit, KIND_EMPTY, state.kind),
        crop=jnp.where(hit, CROP_NONE, state.crop),
        yield_units=jnp.where(hit, 0, state.yield_units),
        max_lifespan_step=jnp.where(hit, -1, state.max_lifespan_step),
        fertilized_until_day=jnp.where(hit, -1, state.fertilized_until_day),
    )


def _apply_build(state: EnvState, op: jnp.ndarray) -> EnvState:
    """BUILD_COOP / BUILD_PASTURE turn an empty owned tile into a structure."""
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    kind_here = _gather_unit(state.kind, x, y)
    empty = kind_here == KIND_EMPTY

    coop = (op == OP_BUILD_COOP) & live & empty
    pasture = (op == OP_BUILD_PASTURE) & live & empty

    shape = state.kind.shape
    # A tile may receive both requests; the lower unit index wins, matching the
    # engine's sequential application. Resolve each op to its first claimant,
    # then compare the two winners per tile.
    coop_first = _first_claimant(shape, x, y, coop)
    pasture_first = _first_claimant(shape, x, y, pasture)

    unit_ids = jnp.broadcast_to(jnp.arange(MAX_UNITS), coop.shape)
    sentinel = MAX_UNITS + 1
    idx = (
        jnp.arange(x.shape[0])[:, None, None],
        jnp.arange(2)[None, :, None],
        y,
        x,
    )
    coop_idx = (
        jnp.full(shape, sentinel, dtype=jnp.int32)
        .at[idx]
        .min(jnp.where(coop_first, unit_ids, sentinel))
    )
    pasture_idx = (
        jnp.full(shape, sentinel, dtype=jnp.int32)
        .at[idx]
        .min(jnp.where(pasture_first, unit_ids, sentinel))
    )
    coop_wins = coop_idx < pasture_idx
    pasture_wins = pasture_idx < coop_idx

    kind = jnp.where(coop_wins, KIND_COOP, state.kind)
    kind = jnp.where(pasture_wins, KIND_PASTURE, kind)
    return state._replace(kind=kind)


def _apply_place(
    state: EnvState,
    op: jnp.ndarray,
    tile_op: jnp.ndarray,
    item_arg: jnp.ndarray,
    qty_arg: jnp.ndarray,
    day: jnp.ndarray,
) -> EnvState:
    """PLACE an animal onto a matching structure, or drop one item in the shed.

    The engine overloads this op, and both halves matter. It first tries to
    place an animal: the item must name an animal, the unit must be carrying
    one, and it must be standing on a matching unoccupied COOP / PASTURE. If
    *any* of that fails it falls through to a shed drop — deposit ``qty`` of
    the named item, if the unit is shed-adjacent and the shed has room.

    Missing the fallthrough is not a harmless omission: a unit carrying an
    animal while standing on a shed tile silently keeps it instead of storing
    it, and the divergence only surfaces days later.

    The two branches see different ops. Animal placement mutates a tile, so it
    takes ``tile_op`` and is suppressed on LOCKED ground. The shed drop uses the
    tile only as a standing position — the shed itself is always owned — so it
    takes the unmasked ``op``. Three of the four shed-access tiles start LOCKED,
    so gating the drop on ownership would make the shed unreachable from them.
    """
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    kind_here = _gather_unit(state.kind, x, y)
    animal_here = _gather_unit(state.animal, x, y)

    # Livestock is addressed above the product range, matching PICKUP / DROP.
    names_animal = (item_arg >= N_PRODUCTS) & (item_arg < N_PRODUCTS + N_ANIMALS)
    animal_idx = jnp.clip(item_arg - N_PRODUCTS, 0, N_ANIMALS - 1)
    structure_ok = kind_here == ANIMAL_STRUCTURE[animal_idx]
    held = jnp.take_along_axis(state.carried_animals, animal_idx[..., None], axis=-1)[
        ..., 0
    ]
    wants = (
        (tile_op == OP_PLACE)
        & live
        & names_animal
        & structure_ok
        & (animal_here == ANIMAL_NONE)
        & (held > 0)
    )

    ok = _first_claimant(state.kind.shape, x, y, wants)
    placed = _unit_mask_plane(state.kind.shape, x, y, ok) > 0

    # `ok` admits at most one unit per tile, so `.add` onto a zeroed tile is a
    # collision-free write. Encode as animal+1 so ANIMAL_NONE (-1) is the base.
    animal_plane = _scatter_unit_add(
        jnp.where(placed, ANIMAL_NONE, state.animal),
        x,
        y,
        jnp.where(ok, animal_idx + 1, 0),
    )

    consumed = jax.nn.one_hot(animal_idx, N_ANIMALS, dtype=jnp.int32) * ok[
        ..., None
    ].astype(jnp.int32)

    state = state._replace(
        animal=animal_plane,
        carried_animals=state.carried_animals - consumed,
        placed_day=_scatter_unit_add(
            jnp.where(placed, 0, state.placed_day),
            x,
            y,
            jnp.where(ok, jnp.broadcast_to(day[:, None, None], ok.shape), 0),
        ),
        consecutive_unfed=jnp.where(placed, 0, state.consecutive_unfed),
        fed_today=jnp.where(placed, 0, state.fed_today),
        cared_today=jnp.where(placed, 0, state.cared_today),
        fertilizer_available=jnp.where(placed, 0, state.fertilizer_available),
        pending_care_bonus=jnp.where(placed, 0, state.pending_care_bonus),
        yield_units=jnp.where(placed, 0, state.yield_units),
    )

    # --- Fallthrough: anything that did not place an animal is a shed drop. ---
    drops = (op == OP_PLACE) & live & ~ok & is_shed_adjacent(x, y) & (qty_arg > 0)

    used = state.shed.sum(axis=-1) + state.animal_shed.sum(axis=-1)
    room = jnp.maximum(0, SHED_CAPACITY - used)  # (B, 2)

    is_product = (item_arg >= 0) & (item_arg < N_PRODUCTS)
    is_animal = (item_arg >= N_PRODUCTS) & (item_arg < N_PRODUCTS + N_ANIMALS)

    p_item = jnp.clip(item_arg, 0, N_PRODUCTS - 1)
    p_held = jnp.take_along_axis(state.carried, p_item[..., None], axis=-1)[..., 0]
    p_move = jnp.where(drops & is_product, jnp.minimum(qty_arg, p_held), 0)

    a_item = jnp.clip(item_arg - N_PRODUCTS, 0, N_ANIMALS - 1)
    a_held = jnp.take_along_axis(state.carried_animals, a_item[..., None], axis=-1)[
        ..., 0
    ]
    a_move = jnp.where(drops & is_animal, jnp.minimum(qty_arg, a_held), 0)

    # One unit acts per outer-loop iteration, so a single cap suffices here.
    total = (p_move + a_move).sum(axis=-1)
    scale = jnp.where(total > room, room, total)
    keep = jnp.where(total > 0, scale, 0)[..., None]
    p_move = jnp.minimum(p_move, keep)
    a_move = jnp.minimum(a_move, jnp.maximum(0, keep - p_move))

    p_hot = jax.nn.one_hot(p_item, N_PRODUCTS, dtype=jnp.int32)
    a_hot = jax.nn.one_hot(a_item, N_ANIMALS, dtype=jnp.int32)

    return state._replace(
        shed=state.shed + (p_move[..., None] * p_hot).sum(axis=2),
        animal_shed=state.animal_shed + (a_move[..., None] * a_hot).sum(axis=2),
        carried=state.carried - p_move[..., None] * p_hot,
        carried_animals=state.carried_animals - a_move[..., None] * a_hot,
    )


def _apply_animal_care(state: EnvState, op: jnp.ndarray) -> EnvState:
    """FEED, CARE, and COLLECT_FERTILIZER on a tile holding an animal."""
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    animal_here = _gather_unit(state.animal, x, y)
    has_animal = animal_here != ANIMAL_NONE

    # --- FEED: costs one WHEAT from the acting unit's carried inventory. ---
    fed_here = _gather_unit(state.fed_today, x, y)
    wheat_held = state.carried[..., PRODUCT_WHEAT]
    wants_feed = (
        (op == OP_FEED) & live & has_animal & (fed_here == 0) & (wheat_held > 0)
    )
    # One feed per tile: lowest unit index pays.
    unit_ids = jnp.broadcast_to(jnp.arange(MAX_UNITS), wants_feed.shape)
    big = MAX_UNITS + 1
    idx = (
        jnp.arange(x.shape[0])[:, None, None],
        jnp.arange(2)[None, :, None],
        y,
        x,
    )
    first_feed = (
        jnp.full(state.kind.shape, big)
        .at[idx]
        .min(jnp.where(wants_feed, unit_ids, big))
    )
    feeds = wants_feed & (unit_ids == _gather_unit(first_feed, x, y))

    carried = state.carried.at[..., PRODUCT_WHEAT].add(-feeds.astype(jnp.int32))
    fed_today = _scatter_unit_max(state.fed_today, x, y, feeds.astype(jnp.int32))

    # --- CARE: free, idempotent per day. ---
    cared_here = _gather_unit(state.cared_today, x, y)
    cares = (op == OP_CARE) & live & has_animal & (cared_here == 0)
    cared_today = _scatter_unit_max(state.cared_today, x, y, cares.astype(jnp.int32))

    # --- COLLECT_FERTILIZER: consumes the tile's one available unit. ---
    avail_here = _gather_unit(state.fertilizer_available, x, y)
    wants_collect = (
        (op == OP_COLLECT_FERTILIZER) & live & has_animal & (avail_here == 1)
    )
    first_collect = (
        jnp.full(state.kind.shape, big)
        .at[idx]
        .min(jnp.where(wants_collect, unit_ids, big))
    )
    collects = wants_collect & (unit_ids == _gather_unit(first_collect, x, y))
    carried = carried.at[..., PRODUCT_FERTILIZER].add(collects.astype(jnp.int32))
    taken = _unit_mask_plane(state.kind.shape, x, y, collects) > 0
    fertilizer_available = jnp.where(taken, 0, state.fertilizer_available)

    return state._replace(
        carried=carried,
        fed_today=fed_today,
        cared_today=cared_today,
        fertilizer_available=fertilizer_available,
    )


def _apply_fertilize(state: EnvState, op: jnp.ndarray, day: jnp.ndarray) -> EnvState:
    """FERTILIZE a plant, spending one carried FERTILIZER.

    Active for ``day``, ``day+1``, ``day+2`` — the engine stores ``day + 2``.
    """
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    kind_here = _gather_unit(state.kind, x, y)
    held = state.carried[..., PRODUCT_FERTILIZER]

    wants = (op == OP_FERTILIZE) & live & (kind_here == KIND_PLANT) & (held > 0)

    # Each unit spends its own fertilizer, but only one application per tile
    # has any effect, so let the lowest index pay and skip the rest.
    unit_ids = jnp.broadcast_to(jnp.arange(MAX_UNITS), wants.shape)
    big = MAX_UNITS + 1
    idx = (
        jnp.arange(x.shape[0])[:, None, None],
        jnp.arange(2)[None, :, None],
        y,
        x,
    )
    first = jnp.full(state.kind.shape, big).at[idx].min(jnp.where(wants, unit_ids, big))
    ok = wants & (unit_ids == _gather_unit(first, x, y))

    hit = _unit_mask_plane(state.kind.shape, x, y, ok) > 0
    day_p = day[:, None, None, None]
    return state._replace(
        carried=state.carried.at[..., PRODUCT_FERTILIZER].add(-ok.astype(jnp.int32)),
        fertilized_until_day=jnp.where(
            hit,
            jnp.maximum(state.fertilized_until_day, day_p + 2),
            state.fertilized_until_day,
        ),
    )


def _apply_shed_transfer(
    state: EnvState, op: jnp.ndarray, item_arg: jnp.ndarray, qty_arg: jnp.ndarray
) -> EnvState:
    """DROP (whole inventory into the shed) and PICKUP (one item out of it).

    Both require standing on a shed-access tile and both respect
    ``SHED_CAPACITY``. Units are served in index order so the capacity race
    resolves the same way the engine's sequential loop does.

    ``item_arg`` addresses a product in ``[0, N_PRODUCTS)`` and a livestock
    animal in ``[N_PRODUCTS, N_PRODUCTS + N_ANIMALS)``. The engine keeps both
    in one ``shed`` dict; they are split across two arrays here because animals
    are not tradeable products, so PICKUP has to reach either one.
    """
    x, y = state.unit_x, state.unit_y
    live = state.unit_active == 1
    adjacent = is_shed_adjacent(x, y)

    drops = (op == OP_DROP) & live & adjacent
    pick_base = (op == OP_PICKUP) & live & adjacent & (qty_arg > 0)
    picks_product = pick_base & (item_arg >= 0) & (item_arg < N_PRODUCTS)
    picks_animal = (
        pick_base & (item_arg >= N_PRODUCTS) & (item_arg < N_PRODUCTS + N_ANIMALS)
    )

    def unit_pass(
        u: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        shed, animal_shed, carried, carried_animals = carry
        # Livestock shares the shed's capacity with products.
        used = shed.sum(axis=-1) + animal_shed.sum(axis=-1)
        room = jnp.maximum(0, SHED_CAPACITY - used)  # (B, 2)

        # --- DROP: deposit this unit's whole inventory, overflow discarded. ---
        mine = carried[:, :, u, :]
        mine_animals = carried_animals[:, :, u, :]
        dropping = drops[:, :, u]
        # Deposit in product order, then livestock, until the room runs out.
        stacked = jnp.concatenate([mine, mine_animals], axis=-1)
        cumulative = jnp.cumsum(stacked, axis=-1)
        prior = cumulative - stacked
        deposit = jnp.clip(room[..., None] - prior, 0, stacked)
        deposit = jnp.where(dropping[..., None], deposit, 0)
        shed = shed + deposit[..., :N_PRODUCTS]
        animal_shed = animal_shed + deposit[..., N_PRODUCTS:]
        # The engine clears the inventory whether or not it fit.
        mine = jnp.where(dropping[..., None], 0, mine)
        mine_animals = jnp.where(dropping[..., None], 0, mine_animals)

        # --- PICKUP: take up to qty of one item out of the shed. ---
        want = qty_arg[:, :, u]

        item = jnp.clip(item_arg[:, :, u], 0, N_PRODUCTS - 1)
        available = jnp.take_along_axis(shed, item[..., None], axis=-1)[..., 0]
        take = jnp.where(picks_product[:, :, u], jnp.minimum(want, available), 0)
        hot = jax.nn.one_hot(item, N_PRODUCTS, dtype=jnp.int32)
        shed = shed - take[..., None] * hot
        mine = mine + take[..., None] * hot

        a_item = jnp.clip(item_arg[:, :, u] - N_PRODUCTS, 0, N_ANIMALS - 1)
        a_avail = jnp.take_along_axis(animal_shed, a_item[..., None], axis=-1)[..., 0]
        a_take = jnp.where(picks_animal[:, :, u], jnp.minimum(want, a_avail), 0)
        a_hot = jax.nn.one_hot(a_item, N_ANIMALS, dtype=jnp.int32)
        animal_shed = animal_shed - a_take[..., None] * a_hot
        mine_animals = mine_animals + a_take[..., None] * a_hot

        return (
            shed,
            animal_shed,
            carried.at[:, :, u, :].set(mine),
            carried_animals.at[:, :, u, :].set(mine_animals),
        )

    shed, animal_shed, carried, carried_animals = jax.lax.fori_loop(
        0,
        MAX_UNITS,
        unit_pass,
        (state.shed, state.animal_shed, state.carried, state.carried_animals),
    )
    return state._replace(
        shed=shed,
        animal_shed=animal_shed,
        carried=carried,
        carried_animals=carried_animals,
    )


# --- Market ---------------------------------------------------------------


def _do_hire(state: EnvState, wants: jnp.ndarray) -> EnvState:
    """Activate the next unit slot for players requesting HIRE.

    Cost is ``fib(hires_today)``; the new hand spawns on the shed-access tile
    with the fewest occupants, NWSE order breaking ties.
    """
    from .state import SHED_ACCESS_X, SHED_ACCESS_Y

    cost = HIRE_COST[jnp.clip(state.hires_today, 0, MAX_UNITS)].astype(jnp.float32)
    n_active = state.unit_active.sum(axis=-1)
    ok = wants & (state.money >= cost) & (n_active < MAX_UNITS)

    slot = jnp.clip(n_active, 0, MAX_UNITS - 1)

    # Occupancy of each shed-access tile by currently active units.
    on_tile = (
        (state.unit_x[..., None] == SHED_ACCESS_X)
        & (state.unit_y[..., None] == SHED_ACCESS_Y)
        & (state.unit_active == 1)[..., None]
    )
    occupancy = on_tile.sum(axis=2)  # (B, 2, 4)
    # argmin returns the first minimum, which is exactly the NWSE tie-break.
    choice = jnp.argmin(occupancy, axis=-1)
    spawn_x = SHED_ACCESS_X[choice]
    spawn_y = SHED_ACCESS_Y[choice]

    slot_hot = jax.nn.one_hot(slot, MAX_UNITS, dtype=jnp.int32) * ok[..., None].astype(
        jnp.int32
    )

    return state._replace(
        money=state.money - jnp.where(ok, cost, 0.0),
        hires_today=state.hires_today + ok.astype(jnp.int32),
        unit_active=state.unit_active + slot_hot,
        unit_x=jnp.where(slot_hot == 1, spawn_x[..., None], state.unit_x),
        unit_y=jnp.where(slot_hot == 1, spawn_y[..., None], state.unit_y),
        # A fresh hand carries nothing.
        carried=jnp.where(slot_hot[..., None] == 1, 0, state.carried),
        carried_animals=jnp.where(slot_hot[..., None] == 1, 0, state.carried_animals),
    )


def _do_buy_land(state: EnvState, wants: jnp.ndarray) -> EnvState:
    """Unlock the next quadrant in LAND_ORDER for players requesting BUY_LAND."""
    n = state.lands_bought
    has_room = n < len(LAND_ORDER)
    cost = LAND_PRICES[jnp.clip(n, 0, len(LAND_ORDER) - 1)].astype(jnp.float32)
    ok = wants & has_room & (state.money >= cost)

    quadrant = jnp.array(LAND_ORDER, dtype=jnp.int32)[
        jnp.clip(n, 0, len(LAND_ORDER) - 1)
    ]

    ys, xs = jnp.meshgrid(jnp.arange(BOARD_SIZE), jnp.arange(BOARD_SIZE), indexing="ij")
    tile_quad = quadrant_of(xs, ys)  # (H, W)
    unlocks = (tile_quad[None, None, :, :] == quadrant[:, :, None, None]) & ok[
        :, :, None, None
    ]

    return state._replace(
        money=state.money - jnp.where(ok, cost, 0.0),
        lands_bought=n + ok.astype(jnp.int32),
        kind=jnp.where(unlocks & (state.kind == KIND_LOCKED), KIND_EMPTY, state.kind),
    )


def _process_market(
    state: EnvState,
    market_op: jnp.ndarray,
    market_item: jnp.ndarray,
    market_qty: jnp.ndarray,
) -> EnvState:
    """Settle every player's order list in the engine's lockstep order.

    The engine walks order slots in parallel across players: for slot ``i`` it
    resolves both players' atomic ops (HIRE, BUY_LAND) in player order, then
    runs a per-unit loop where both players are quoted against the same
    pre-commit inventory and then both commit. That serial dependence is real,
    so the unit loop is a fixed-trip ``fori_loop`` bounded by
    :data:`MAX_ORDER_UNITS`, and prices are refreshed after each slot.

    Shapes: ``market_op`` / ``market_item`` / ``market_qty`` are
    ``(B, 2, MAX_MARKET_ORDERS)``.
    """

    def slot_step(i: int, st: EnvState) -> EnvState:
        op = market_op[:, :, i]
        item = market_item[:, :, i]
        qty = market_qty[:, :, i]

        # Atomic orders resolve once, before the per-unit loop.
        st = _do_hire(st, op == MARKET_HIRE)
        st = _do_buy_land(st, op == MARKET_BUY_LAND)

        remaining0 = jnp.where(
            (op == MARKET_SELL)
            | (op == MARKET_BUY_SEED)
            | (op == MARKET_BUY_PRODUCT)
            | (op == MARKET_BUY_ANIMAL),
            jnp.clip(qty, 0, MAX_ORDER_UNITS),
            0,
        )

        def unit_step(
            _: int,
            carry: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
        ) -> tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ]:
            market_inv, money, shed, seeds, animal_shed, remaining = carry

            active = remaining > 0
            prices = market_price(market_inv)  # (B, N_PRODUCTS)

            item_c = jnp.clip(item, 0, N_PRODUCTS - 1)
            sell_price = jnp.take_along_axis(prices, item_c, axis=-1)
            # BUY_PRODUCT quotes at post-buy inventory so a buy/sell round-trip
            # against an unchanged market nets zero.
            buy_prices = market_price(market_inv[:, None, :] - 1)
            buy_price = jnp.take_along_axis(buy_prices, item_c[..., None], axis=-1)[
                ..., 0
            ]

            crop_c = jnp.clip(item, 0, N_CROPS - 1)
            animal_c = jnp.clip(item, 0, N_ANIMALS - 1)
            seed_cost = CROP_SEED_COST[crop_c]
            animal_cost = ANIMAL_COST[animal_c]

            shed_total = shed.sum(axis=-1) + animal_shed.sum(axis=-1)
            has_room = shed_total < SHED_CAPACITY

            held = jnp.take_along_axis(shed, item_c[..., None], axis=-1)[..., 0]

            # BUY_PRODUCT is restricted to WHEAT and FERTILIZER by the engine.
            buyable_product = (item == PRODUCT_WHEAT) | (item == PRODUCT_FERTILIZER)

            can_sell = (
                active
                & (op == MARKET_SELL)
                & (item >= 0)
                & (item < N_PRODUCTS)
                & (held > 0)
            )
            can_buy_product = (
                active
                & (op == MARKET_BUY_PRODUCT)
                & buyable_product
                & (money >= buy_price)
                & has_room
            )
            can_buy_seed = (
                active
                & (op == MARKET_BUY_SEED)
                & (item >= 0)
                & (item < N_CROPS)
                & (money >= seed_cost)
            )
            can_buy_animal = (
                active
                & (op == MARKET_BUY_ANIMAL)
                & (item >= 0)
                & (item < N_ANIMALS)
                & (money >= animal_cost)
                & has_room
            )

            product_hot = jax.nn.one_hot(item_c, N_PRODUCTS, dtype=jnp.int32)
            seed_hot = jax.nn.one_hot(crop_c, N_CROPS, dtype=jnp.int32)
            animal_hot = jax.nn.one_hot(animal_c, N_ANIMALS, dtype=jnp.int32)

            # SELL: shed down, money up; the book only grows above the floor.
            shed = shed - can_sell[..., None].astype(jnp.int32) * product_hot
            money = money + jnp.where(can_sell, sell_price, 0).astype(jnp.float32)
            above_floor = sell_price > 1
            market_inv = market_inv + (
                (can_sell & above_floor)[..., None].astype(jnp.int32) * product_hot
            ).sum(axis=1)

            # BUY_PRODUCT: money down, shed up, book down.
            money = money - jnp.where(can_buy_product, buy_price, 0).astype(jnp.float32)
            shed = shed + can_buy_product[..., None].astype(jnp.int32) * product_hot
            market_inv = market_inv - (
                can_buy_product[..., None].astype(jnp.int32) * product_hot
            ).sum(axis=1)

            # BUY_SEED: seeds are minted, never drawn from the book.
            money = money - jnp.where(can_buy_seed, seed_cost, 0).astype(jnp.float32)
            seeds = seeds + can_buy_seed[..., None].astype(jnp.int32) * seed_hot

            # BUY_ANIMAL: lands in the shed as livestock, not as a product.
            money = money - jnp.where(can_buy_animal, animal_cost, 0).astype(
                jnp.float32
            )
            animal_shed = (
                animal_shed + can_buy_animal[..., None].astype(jnp.int32) * animal_hot
            )

            committed = can_sell | can_buy_product | can_buy_seed | can_buy_animal
            # A failed commit aborts the rest of that order, as in the engine.
            remaining = jnp.where(committed, remaining - 1, 0)
            return market_inv, money, shed, seeds, animal_shed, remaining

        market_inv, money, shed, seeds, animal_shed, _ = jax.lax.fori_loop(
            0,
            MAX_ORDER_UNITS,
            unit_step,
            (
                st.market_inv,
                st.money,
                st.shed,
                st.seeds,
                st.animal_shed,
                remaining0,
            ),
        )
        return st._replace(
            market_inv=market_inv,
            money=money,
            shed=shed,
            seeds=seeds,
            animal_shed=animal_shed,
        )

    out: EnvState = jax.lax.fori_loop(0, MAX_MARKET_ORDERS, slot_step, state)
    return out


# --- Town, decay, end of day ---------------------------------------------


def _town_consume(
    state: EnvState,
    step: jnp.ndarray,
    shop_interval: int,
    center_interval: int,
) -> EnvState:
    """Shop and town-centre demand, both keyed off the global step counter."""
    shop_fires = (step % shop_interval) == 0
    shop_demand = (state.shops[:, :, None] * SHOP_DEMAND[None, :, :]).sum(axis=1)
    shop_amount = jnp.where(shop_fires[:, None], shop_demand, 0)

    center_fires = (step % center_interval) == 0
    center_amount = jnp.where(center_fires, 1, 0)[:, None] * TOWN_CENTER_MASK

    return state._replace(market_inv=state.market_inv - shop_amount - center_amount)


def _decay_plants(state: EnvState, step: jnp.ndarray) -> EnvState:
    """Past its lifespan a plant loses a yield unit every other step."""
    mls = state.max_lifespan_step
    step_b = step[:, None, None, None]
    expired = (mls >= 0) & (step_b >= mls) & (((step_b - mls) % 2) == 0)
    is_plant = state.kind == KIND_PLANT

    hit = expired & is_plant
    new_yields = jnp.where(hit, state.yield_units - 1, state.yield_units)
    weeded = hit & (new_yields <= 0)

    return state._replace(
        kind=jnp.where(weeded, KIND_WEED, state.kind),
        crop=jnp.where(weeded, CROP_NONE, state.crop),
        yield_units=jnp.where(weeded, 0, new_yields),
        max_lifespan_step=jnp.where(weeded, -1, state.max_lifespan_step),
        fertilized_until_day=jnp.where(weeded, -1, state.fertilized_until_day),
    )


def _daily_refresh_plants(state: EnvState, day: jnp.ndarray) -> EnvState:
    """Watering accounting and ongoing-crop production at the day boundary."""
    is_plant = state.kind == KIND_PLANT
    was_watered = state.watered_today == 1

    unwatered = jnp.where(was_watered, 0, state.consecutive_unwatered + 1)
    # Two consecutive dry days turns a plant into a weed.
    weeded = is_plant & (unwatered >= 2)

    crop_idx = jnp.maximum(state.crop, 0)
    day_p = day[:, None, None, None]
    next_day = day_p + 1
    ongoing = CROP_ONGOING[crop_idx] == 1
    interval = jnp.maximum(CROP_INTERVAL[crop_idx], 1)
    since_first = next_day - state.planted_day - CROP_FIRST_YIELD_DAY[crop_idx]
    produces = (
        is_plant
        & ~weeded
        & ongoing
        & (since_first >= 0)
        & ((since_first % interval) == 0)
    )
    count = since_first // interval + 1
    produces = produces & (count <= CROP_MAX_YIELD[crop_idx])

    # Fertilizer bonus only applies on watered days (basic needs first).
    fertilized = was_watered & (state.fertilized_until_day >= day_p)
    gain = jnp.where(fertilized, 2, 1)
    yields = jnp.where(
        produces,
        jnp.minimum(CROP_MAX_YIELD[crop_idx], state.yield_units + gain),
        state.yield_units,
    )

    # An ongoing crop that just made its final delivery gets a lifespan.
    final = produces & (count == CROP_MAX_YIELD[crop_idx])
    max_lifespan = jnp.where(
        final, (next_day + 1) * TURNS_PER_DAY, state.max_lifespan_step
    )

    return state._replace(
        kind=jnp.where(weeded, KIND_WEED, state.kind),
        crop=jnp.where(weeded, CROP_NONE, state.crop),
        yield_units=jnp.where(weeded, 0, yields),
        max_lifespan_step=jnp.where(weeded, -1, max_lifespan),
        fertilized_until_day=jnp.where(weeded, -1, state.fertilized_until_day),
        consecutive_unwatered=jnp.where(
            is_plant, unwatered, state.consecutive_unwatered
        ),
        watered_today=jnp.zeros_like(state.watered_today),
    )


def _daily_refresh_animals(state: EnvState, day: jnp.ndarray) -> EnvState:
    """Feeding accounting, production, care bonus, and escape at day end."""
    has_animal = state.animal != ANIMAL_NONE
    fed = state.fed_today == 1

    unfed = jnp.where(fed, 0, state.consecutive_unfed + 1)
    # Two consecutive unfed days and the animal escapes; the structure remains.
    escapes = has_animal & (unfed >= 2)

    a_idx = jnp.maximum(state.animal, 0)
    next_day = (day + 1)[:, None, None, None]
    since_first = next_day - state.placed_day - ANIMAL_FIRST_YIELD_DAY[a_idx]
    interval = jnp.maximum(ANIMAL_INTERVAL[a_idx], 1)
    produces = (
        has_animal & ~escapes & (since_first >= 0) & ((since_first % interval) == 0)
    )

    # The care bonus is only consumed on a day the animal was also fed.
    bonus = jnp.where(produces & fed, state.pending_care_bonus, 0)
    yields = jnp.where(
        produces,
        jnp.minimum(ANIMAL_MAX_HELD[a_idx], state.yield_units + 1 + bonus),
        state.yield_units,
    )
    # Whether or not it paid out, a production day clears the banked bonus.
    pending = jnp.where(produces, 0, state.pending_care_bonus)
    # CARE on a fed day banks one unit toward the next production.
    pending = jnp.where(
        has_animal & ~escapes & (state.cared_today == 1) & fed, pending + 1, pending
    )

    keeps = has_animal & ~escapes
    return state._replace(
        # An escaped animal leaves its structure behind.
        animal=jnp.where(escapes, ANIMAL_NONE, state.animal),
        kind=jnp.where(escapes, ANIMAL_STRUCTURE[a_idx], state.kind),
        yield_units=jnp.where(
            escapes, 0, jnp.where(has_animal, yields, state.yield_units)
        ),
        consecutive_unfed=jnp.where(has_animal, unfed, state.consecutive_unfed),
        pending_care_bonus=jnp.where(escapes, 0, pending),
        fertilizer_available=jnp.where(
            escapes, 0, jnp.where(keeps, 1, state.fertilizer_available)
        ),
        fed_today=jnp.zeros_like(state.fed_today),
        cared_today=jnp.zeros_like(state.cared_today),
    )


def _spawn_weeds(state: EnvState, weed_chance: float) -> EnvState:
    """Each empty owned tile independently turns to weed with ``weed_chance``.

    The engine draws sequentially from a per-day ``random.Random``; here each
    environment carries its own key. Same distribution, different stream.
    """
    # Split per environment: `next_key` carries the stream forward, `sub` is
    # consumed here. vmap keeps each environment's stream independent.
    next_key, sub = jax.vmap(lambda k: tuple(jax.random.split(k)))(state.rng)
    per_env_shape = state.kind.shape[1:]  # (2, H, W)
    spawn = jax.vmap(lambda k: jax.random.uniform(k, per_env_shape))(sub)

    weeds = (state.kind == KIND_EMPTY) & (spawn < weed_chance)
    return state._replace(
        kind=jnp.where(weeds, KIND_WEED, state.kind),
        rng=next_key,
    )


def _unlock_shops(state: EnvState, day: jnp.ndarray, unlock_interval: int) -> EnvState:
    """On unlock days the town draws one shop uniformly, with replacement."""
    next_day = day + 1
    fires = (next_day > 0) & ((next_day % unlock_interval) == 0)
    has_room = state.shops.sum(axis=-1) < MAX_SHOP_INSTANCES

    next_key, sub = jax.vmap(lambda k: tuple(jax.random.split(k)))(state.rng)
    choice = jax.vmap(lambda k: jax.random.randint(k, (), 0, N_SHOPS))(sub)

    add = jax.nn.one_hot(choice, N_SHOPS, dtype=jnp.int32) * (fires & has_room)[
        :, None
    ].astype(jnp.int32)
    return state._replace(shops=state.shops + add, rng=next_key)


def _drop_inventories_to_shed(state: EnvState) -> EnvState:
    """Every unit's carried inventory drops into the shed; overflow is lost.

    Same deposit order as an explicit DROP — products then livestock, in index
    order, until ``SHED_CAPACITY`` runs out.
    """

    def unit_pass(
        u: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        shed, animal_shed, carried, carried_animals = carry
        used = shed.sum(axis=-1) + animal_shed.sum(axis=-1)
        room = jnp.maximum(0, SHED_CAPACITY - used)[..., None]

        stacked = jnp.concatenate(
            [carried[:, :, u, :], carried_animals[:, :, u, :]], axis=-1
        )
        cumulative = jnp.cumsum(stacked, axis=-1)
        prior = cumulative - stacked
        deposit = jnp.clip(room - prior, 0, stacked)

        return (
            shed + deposit[..., :N_PRODUCTS],
            animal_shed + deposit[..., N_PRODUCTS:],
            carried.at[:, :, u, :].set(0),
            carried_animals.at[:, :, u, :].set(0),
        )

    shed, animal_shed, carried, carried_animals = jax.lax.fori_loop(
        0,
        MAX_UNITS,
        unit_pass,
        (state.shed, state.animal_shed, state.carried, state.carried_animals),
    )
    return state._replace(
        shed=shed,
        animal_shed=animal_shed,
        carried=carried,
        carried_animals=carried_animals,
    )


def _end_of_day(
    state: EnvState,
    day: jnp.ndarray,
    weed_chance: float,
    unlock_interval: int,
) -> EnvState:
    """The engine's ``_end_of_day``, in its exact order."""
    state = _daily_refresh_plants(state, day)
    state = _daily_refresh_animals(state, day)
    state = _spawn_weeds(state, weed_chance)
    state = _drop_inventories_to_shed(state)

    # Hands vanish at day end; the farmer respawns at the default tile.
    from .state import default_spawn

    spawn_x, spawn_y = default_spawn()
    active = jnp.zeros_like(state.unit_active).at[:, :, 0].set(1)
    state = state._replace(
        unit_active=active,
        unit_x=jnp.full_like(state.unit_x, spawn_x),
        unit_y=jnp.full_like(state.unit_y, spawn_y),
        hires_today=jnp.zeros_like(state.hires_today),
        carried=jnp.zeros_like(state.carried),
        carried_animals=jnp.zeros_like(state.carried_animals),
    )
    return _unlock_shops(state, day, unlock_interval)


# --- Step -----------------------------------------------------------------


def _apply_unit_ops(
    state: EnvState,
    op: jnp.ndarray,
    arg: jnp.ndarray,
    qty: jnp.ndarray,
    day: jnp.ndarray,
) -> EnvState:
    """All unit ops, applied unit by unit as the engine applies them.

    The loop is **unit-major, not op-major**, and that ordering is load-bearing:
    the engine processes one unit's whole action before starting the next, so a
    later unit observes earlier units' tile mutations. A hand that DIGs a tile
    frees it for a subsequent hand's PLANT in the same turn. Applying each op
    across all units in turn — plant-for-everyone, then dig-for-everyone —
    silently gets that case wrong, and the difference is invisible until it
    changes a harvest several days later.

    Sequencing over ``MAX_UNITS`` also removes every same-tile tie: within one
    iteration exactly one unit per player acts, so no scatter can collide.
    """
    # The seed pre-check runs on the raw action list, before movement and
    # before the LOCKED guard. A unit whose PLANT will fail anyway — because it
    # stands on an occupied or locked tile — still counts toward the demand
    # that can block every other PLANT for that crop.
    op = _resolve_plant_demand(state, op, arg)

    crop_arg = jnp.clip(arg, 0, N_CROPS - 1)

    def one_unit(u: int, st: EnvState) -> EnvState:
        # Mask every other slot to PASS so the shared machinery, which is
        # written over the full (B, 2, MAX_UNITS) unit axis, advances exactly
        # one unit per player per iteration.
        slot = jnp.arange(MAX_UNITS) == u
        active = slot & (st.unit_active == 1)
        u_op = jnp.where(active, op, OP_PASS)

        st = _apply_moves(st, u_op)

        # Shed ops resolve before the LOCKED guard: three of the four
        # shed-access tiles start locked, and the shed itself is always owned.
        st = _apply_shed_transfer(st, u_op, arg, qty)

        # Everything below mutates the tile the unit stands on, so it requires
        # that tile to be owned.
        owned = _gather_unit(st.kind, st.unit_x, st.unit_y) != KIND_LOCKED
        tile_op = jnp.where(owned, u_op, OP_PASS)

        st = _apply_plant(st, tile_op, crop_arg, day)
        st = _apply_water(st, tile_op, day)
        st = _apply_harvest(st, tile_op, day)
        st = _apply_fertilize(st, tile_op, day)
        st = _apply_dig(st, tile_op)
        st = _apply_build(st, tile_op)
        # PLACE spans the guard: animal placement is tile-gated, the shed-drop
        # fallthrough is not, so it receives both ops.
        st = _apply_place(st, u_op, tile_op, arg, qty, day)
        return _apply_animal_care(st, tile_op)

    out: EnvState = jax.lax.fori_loop(0, MAX_UNITS, one_unit, state)
    return out


def _step_impl(
    state: EnvState,
    op: jnp.ndarray,
    arg: jnp.ndarray,
    qty: jnp.ndarray,
    market_op: jnp.ndarray,
    market_item: jnp.ndarray,
    market_qty: jnp.ndarray,
    weed_chance: float,
    shop_interval: int,
    center_interval: int,
    unlock_interval: int,
) -> EnvState:
    day = state.step // TURNS_PER_DAY

    state = _apply_unit_ops(state, op, arg, qty, day)
    state = _process_market(state, market_op, market_item, market_qty)
    state = _town_consume(state, state.step, shop_interval, center_interval)
    state = _decay_plants(state, state.step)

    closes_day = ((state.step + 1) % TURNS_PER_DAY) == 0
    eod = _end_of_day(state, day, weed_chance, unlock_interval)
    state = jax.tree_util.tree_map(
        lambda new, old: jnp.where(
            closes_day.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
        ),
        eod,
        state,
    )

    return state._replace(step=state.step + 1)


@jax.jit
def step(
    state: EnvState,
    op: jnp.ndarray,
    arg: jnp.ndarray,
    qty: jnp.ndarray,
    market_op: jnp.ndarray,
    market_item: jnp.ndarray,
    market_qty: jnp.ndarray,
) -> EnvState:
    """Advance every environment one turn under the default configuration.

    Shapes:
      ``op`` / ``arg`` / ``qty``:                 ``(B, 2, MAX_UNITS)``
      ``market_*``:                               ``(B, 2, MAX_MARKET_ORDERS)``

    ``arg`` carries the op's operand — a crop code for PLANT, an animal code
    for PLACE, a product code for PICKUP — and is ignored by ops that take no
    operand. ``qty`` is only read by PICKUP.

    Ordering mirrors the engine: unit actions, then the market, then plant
    decay, then end-of-day when the turn closes a day.
    """
    return _step_impl(
        state,
        op,
        arg,
        qty,
        market_op,
        market_item,
        market_qty,
        DEFAULT_WEED_SPAWN_CHANCE,
        4,
        24,
        3,
    )


def make_step(
    weed_chance: float = DEFAULT_WEED_SPAWN_CHANCE,
    shop_sell_interval: int = 4,
    center_sell_interval: int = 24,
    shop_unlock_interval: int = 3,
) -> Callable[
    [
        EnvState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    EnvState,
]:
    """Build a jitted step for a non-default configuration.

    The intervals and the weed chance are static (they shape the trace), so a
    changed configuration needs its own compiled step. Used by the equivalence
    tests, which disable the RNG paths to isolate the deterministic core.
    """

    @jax.jit
    def _step(
        state: EnvState,
        op: jnp.ndarray,
        arg: jnp.ndarray,
        qty: jnp.ndarray,
        market_op: jnp.ndarray,
        market_item: jnp.ndarray,
        market_qty: jnp.ndarray,
    ) -> EnvState:
        return _step_impl(
            state,
            op,
            arg,
            qty,
            market_op,
            market_item,
            market_qty,
            weed_chance,
            shop_sell_interval,
            center_sell_interval,
            shop_unlock_interval,
        )

    return _step
