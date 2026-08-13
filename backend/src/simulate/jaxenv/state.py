"""Batched Kaggriculture state as flat JAX arrays.

The official engine holds state as nested Python dicts with per-tile ``dict``
objects. That shape cannot go on an accelerator, so this module re-expresses the
same information as a struct-of-arrays with a leading batch axis ``B``: every
field is a dense array and all ``B`` environments advance in lockstep under one
:func:`jax.jit`-compiled step.

Layout choices that matter:

- **Tiles are planes, not objects.** A tile's ``kind`` / ``crop`` /
  ``planted_day`` / ``watered_today`` / ``consecutive_unwatered`` /
  ``yield_units`` / ``max_lifespan_step`` / ``fertilized_until_day`` each become
  their own ``(B, 2, H, W)`` plane. Branching per tile becomes ``jnp.where``.
- **Units are a fixed-width axis.** The engine grows ``farm["hands"]`` as a
  Python list; a traced program cannot. Instead every player carries
  ``MAX_UNITS`` slots (index 0 = the main farmer, 1.. = hands) with an
  ``unit_active`` mask. Hiring flips a mask bit rather than appending.
- **No Python control flow on traced values.** Every rule is a masked update, so
  the same instruction stream runs for all environments regardless of what any
  individual board contains.
- **int32 throughout** for counts and days; float32 only for money, matching the
  engine's integer-rounded prices.

This is a reimplementation, not a wrapper, so it carries divergence risk that
the ``fast`` engine does not. Equivalence against the official engine is
asserted in ``tests/`` and is the only thing that makes this module usable.

``jax`` needs no entry in ``pyproject.toml``: ``kaggle-environments`` already
requires it directly, so it is present wherever the engine is. Note that
installing ``jax-metal`` into the project venv is actively harmful — it takes
over ``jax.devices()`` for every test, and it runs this workload roughly 100x
*slower* than CPU (see ``docs/plans/simulator/jax-gpu.md``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: Tile kind codes. EMPTY must be 0 so a zeroed board is empty.
KIND_EMPTY = 0
KIND_PLANT = 1
KIND_WEED = 2
KIND_LOCKED = 3
KIND_COOP = 4
KIND_PASTURE = 5

#: Crop codes, ordered to match CROP_* tables below.
CROP_NONE = -1
CROP_WHEAT = 0
CROP_CARROT = 1
CROP_TOMATO = 2
CROP_STRAWBERRY = 3
CROP_MELON = 4

CROP_NAMES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
N_CROPS = len(CROP_NAMES)

#: Per-crop constants, indexed by crop code. Mirrors ``CROPS`` in the engine.
CROP_SEED_COST = jnp.array([10, 20, 50, 100, 80], dtype=jnp.int32)
CROP_FIRST_YIELD_DAY = jnp.array([2, 2, 8, 10, 10], dtype=jnp.int32)
CROP_MAX_YIELD_DAY = jnp.array([4, 3, 8, 10, 12], dtype=jnp.int32)
CROP_INTERVAL = jnp.array([0, 0, 1, 2, 0], dtype=jnp.int32)
CROP_MAX_YIELD = jnp.array([6, 4, 4, 4, 6], dtype=jnp.int32)
CROP_ONGOING = jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32)

#: Animal codes.
ANIMAL_NONE = -1
ANIMAL_GOOSE = 0
ANIMAL_COW = 1
ANIMAL_SHEEP = 2

ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")
N_ANIMALS = len(ANIMAL_NAMES)

#: Per-animal constants, indexed by animal code. Mirrors ``ANIMALS``.
ANIMAL_COST = jnp.array([300, 400, 500], dtype=jnp.int32)
#: Structure kind each animal requires (COOP for geese, PASTURE otherwise).
ANIMAL_STRUCTURE = jnp.array([KIND_COOP, KIND_PASTURE, KIND_PASTURE], dtype=jnp.int32)
ANIMAL_FIRST_YIELD_DAY = jnp.array([4, 8, 6], dtype=jnp.int32)
ANIMAL_INTERVAL = jnp.array([1, 2, 3], dtype=jnp.int32)
ANIMAL_MAX_HELD = jnp.array([4, 6, 6], dtype=jnp.int32)

#: Product codes. Crops occupy 0..4 so a crop code doubles as a product code.
PRODUCT_NAMES = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
N_PRODUCTS = len(PRODUCT_NAMES)

PRODUCT_WHEAT = 0
PRODUCT_EGG = 5
PRODUCT_MILK = 6
PRODUCT_WOOL = 7
PRODUCT_FERTILIZER = 8

#: Product each animal yields, indexed by animal code.
ANIMAL_PRODUCT = jnp.array([PRODUCT_EGG, PRODUCT_MILK, PRODUCT_WOOL], dtype=jnp.int32)

#: Market shape function codes.
SHAPE_LINEAR = 0
SHAPE_SQ = 1
SHAPE_SQRT = 2
SHAPE_LOG = 3
SHAPE_LOG10 = 4

_SHAPE_CODE = {
    "linear": SHAPE_LINEAR,
    "sq": SHAPE_SQ,
    "sqrt": SHAPE_SQRT,
    "log": SHAPE_LOG,
    "log10": SHAPE_LOG10,
}

#: MARKET_PARAMS from the engine, transposed into parallel arrays.
_RAW_MARKET = (
    # base,   T,  below_func, below_target, above_func, above_target
    (25, 400, "sqrt", 0.80, "log", 0.20),  # WHEAT
    (35, 450, "log", 0.20, "sqrt", 0.70),  # CARROT
    (60, 200, "linear", 0.40, "sqrt", 0.60),  # TOMATO
    (120, 100, "sqrt", 0.70, "linear", 1.60),  # STRAWBERRY
    (250, 300, "log", 0.20, "sq", 3.60),  # MELON
    (50, 332, "linear", 0.40, "log", 0.20),  # EGG
    (160, 122, "sqrt", 0.60, "linear", 1.60),  # MILK
    (200, 105, "log", 0.20, "sq", 3.20),  # WOOL
    (100, 200, "linear", 0.40, "linear", 0.40),  # FERTILIZER
)

MARKET_BASE = jnp.array([r[0] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_T = jnp.array([r[1] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_BELOW_FUNC = jnp.array([_SHAPE_CODE[r[2]] for r in _RAW_MARKET], dtype=jnp.int32)
MARKET_BELOW_TARGET = jnp.array([r[3] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_ABOVE_FUNC = jnp.array([_SHAPE_CODE[r[4]] for r in _RAW_MARKET], dtype=jnp.int32)
MARKET_ABOVE_TARGET = jnp.array([r[5] for r in _RAW_MARKET], dtype=jnp.float32)

MARKET_I0 = 10_000
PRICE_FLOOR = 1

BOARD_SIZE = 10
STARTING_MONEY = 3000
SHED_CAPACITY = 100
TURNS_PER_DAY = 24

#: Quadrant codes, matching ``_quadrant_of``: "NW","NE","SW","SE".
QUAD_NW = 0
QUAD_NE = 1
QUAD_SW = 2
QUAD_SE = 3

#: Purchase order and prices from ``LAND_ORDER`` / ``LAND_PRICES``.
LAND_ORDER = (QUAD_NE, QUAD_SW, QUAD_SE)
LAND_PRICES = jnp.array([1000, 2000, 4000], dtype=jnp.int32)

#: Maximum units (farmer + hands) a player can control simultaneously.
#: Slot 0 is the main farmer, so the hand capacity is ``MAX_UNITS - 1``.
#:
#: The engine has no hard cap. Hire cost is ``fib(n)`` for the n-th hire of the
#: day and resets daily, so the ceiling is set by how much a player can spend in
#: one day. Measured cumulative cost:
#:
#:     $3,000 (starting money) -> 16 hands
#:     $10,000                 -> 18 hands
#:     $50,000                 -> 22 hands
#:     $1,000,000              -> 28 hands
#:
#: An earlier value of 16 was wrong: it left only 15 hand slots while the engine
#: reaches 16 on the starting money alone, and a differential test caught the
#: divergence at the boundary. 32 covers 28 hands with headroom; fib growth makes
#: anything beyond that unreachable (the 29th hire alone costs ~$832k).
MAX_UNITS = 32

#: Town shops, alphabetically sorted to match ``rng.choice(sorted(SHOPS))``.
SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
N_SHOPS = len(SHOP_NAMES)

_SHOP_PRODUCTS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "YARN_STORE": ("WOOL",),
}


def _shop_demand_row(name: str) -> list[int]:
    """Per-product consumption for one shop instance.

    A single-product shop consumes 2 per tick, multi-product shops 1 each —
    the ``multiplier`` in ``_town_consume``.
    """
    products = _SHOP_PRODUCTS[name]
    multiplier = 2 if len(products) == 1 else 1
    row = [0] * N_PRODUCTS
    for item in products:
        row[PRODUCT_NAMES.index(item)] = multiplier
    return row


#: (N_SHOPS, N_PRODUCTS) demand matrix; row i is one instance of SHOP_NAMES[i].
SHOP_DEMAND = jnp.array(
    [_shop_demand_row(name) for name in SHOP_NAMES], dtype=jnp.int32
)

#: Town centre consumes one of every product except fertilizer.
TOWN_CENTER_MASK = jnp.array(
    [0 if name == "FERTILIZER" else 1 for name in PRODUCT_NAMES], dtype=jnp.int32
)

#: ``MAX_SHOP_INSTANCES``: the town never unlocks more than this many shops.
MAX_SHOP_INSTANCES = 8

DEFAULT_WEED_SPAWN_CHANCE = 0.005
DEFAULT_TOWN_SHOP_UNLOCK_INTERVAL = 3
DEFAULT_TOWN_SHOP_SELL_INTERVAL = 4
DEFAULT_TOWN_CENTER_SELL_INTERVAL = 24
DEFAULT_MAX_MARKET_ORDERS = 10
DEFAULT_FARM_HAND_COST_MULT = 1


class EnvState(NamedTuple):
    """Batched state. Leading axis is the environment index ``B``.

    Player-indexed fields carry a ``2`` axis directly after the batch axis;
    unit-indexed fields carry ``(B, 2, MAX_UNITS, ...)``.
    """

    step: jnp.ndarray  # (B,) int32
    money: jnp.ndarray  # (B, 2) float32

    # Tile planes, per player: (B, 2, H, W)
    kind: jnp.ndarray
    crop: jnp.ndarray
    planted_day: jnp.ndarray
    watered_today: jnp.ndarray
    consecutive_unwatered: jnp.ndarray
    yield_units: jnp.ndarray
    max_lifespan_step: jnp.ndarray
    fertilized_until_day: jnp.ndarray
    # Animal planes. ``animal`` is ANIMAL_NONE where no animal is placed; the
    # tile kind still records the underlying COOP / PASTURE structure.
    animal: jnp.ndarray
    placed_day: jnp.ndarray
    consecutive_unfed: jnp.ndarray
    fed_today: jnp.ndarray
    cared_today: jnp.ndarray
    fertilizer_available: jnp.ndarray
    pending_care_bonus: jnp.ndarray

    # Unit positions and liveness: (B, 2, MAX_UNITS)
    unit_x: jnp.ndarray
    unit_y: jnp.ndarray
    unit_active: jnp.ndarray
    hires_today: jnp.ndarray  # (B, 2) int32

    # Inventories
    shed: jnp.ndarray  # (B, 2, N_PRODUCTS) int32
    # Livestock held in the shed awaiting placement. The engine keeps animals
    # in the same `shed` dict as products and counts them against the same
    # capacity, but they are not tradeable products, so they get their own
    # axis here and are added into the capacity sum explicitly.
    animal_shed: jnp.ndarray  # (B, 2, N_ANIMALS) int32
    carried: jnp.ndarray  # (B, 2, MAX_UNITS, N_PRODUCTS) int32
    # Livestock a unit is carrying, after PICKUP and before PLACE.
    carried_animals: jnp.ndarray  # (B, 2, MAX_UNITS, N_ANIMALS) int32
    seeds: jnp.ndarray  # (B, 2, N_CROPS) int32

    # Land: how many extra quadrants each player has bought, (B, 2) int32.
    lands_bought: jnp.ndarray

    # Shared market inventory: (B, N_PRODUCTS) int32
    market_inv: jnp.ndarray
    # Shared town: how many instances of each shop are unlocked, (B, N_SHOPS).
    shops: jnp.ndarray
    # Per-environment RNG key driving weed spawn and shop unlocks: (B, 2) uint32
    rng: jnp.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.step.shape[0])

    @property
    def day(self) -> jnp.ndarray:
        return self.step // TURNS_PER_DAY

    @property
    def hour(self) -> jnp.ndarray:
        return self.step % TURNS_PER_DAY

    @property
    def farmer_x(self) -> jnp.ndarray:
        """Main farmer's x, for callers that ignore hands."""
        return self.unit_x[:, :, 0]

    @property
    def farmer_y(self) -> jnp.ndarray:
        return self.unit_y[:, :, 0]


def _shape_fn(code: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Vectorised ``_shape``: select by code rather than branching."""
    x = jnp.maximum(x, 0.0)
    # Guard the inputs of sqrt/log so no branch produces NaN/inf before select.
    safe = jnp.maximum(x, 1e-12)
    options = jnp.stack(
        [
            x,
            x * x,
            jnp.sqrt(safe),
            jnp.log1p(x),
            jnp.log10(1.0 + x),
        ]
    )
    return jnp.take_along_axis(options, code[None, ...], axis=0)[0]


def market_price(inventory: jnp.ndarray) -> jnp.ndarray:
    """Price for every product from its inventory.

    ``inventory`` is ``(..., N_PRODUCTS)``; the result has the same shape.
    Mirrors the engine exactly, including ``int(round(price))`` and the floor.
    """
    inv = inventory.astype(jnp.float32)
    below = inv < MARKET_I0
    delta = jnp.where(below, MARKET_I0 - inv, inv - MARKET_I0)

    func = jnp.where(below, MARKET_BELOW_FUNC, MARKET_ABOVE_FUNC)
    target = jnp.where(below, MARKET_BELOW_TARGET, MARKET_ABOVE_TARGET)

    codes = jnp.broadcast_to(func, delta.shape)
    denom = _shape_fn(codes, jnp.broadcast_to(MARKET_T, delta.shape))
    amp = target * MARKET_BASE / denom
    magnitude = amp * _shape_fn(codes, delta)

    price = jnp.where(below, MARKET_BASE + magnitude, MARKET_BASE - magnitude)
    # The engine rounds half away from zero via Python's round-then-int on a
    # positive value; jnp.round is banker's rounding, so add the tie offset.
    rounded = jnp.floor(price + 0.5)
    return jnp.maximum(PRICE_FLOOR, rounded).astype(jnp.int32)


def quadrant_of(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Quadrant code for a tile, matching the engine's ``_quadrant_of``."""
    half = BOARD_SIZE // 2
    north = y < half
    west = x < half
    return jnp.where(
        north,
        jnp.where(west, QUAD_NW, QUAD_NE),
        jnp.where(west, QUAD_SW, QUAD_SE),
    ).astype(jnp.int32)


#: Shed-access tiles in the engine's NWSE order: (x, y) pairs.
def shed_access_tiles() -> tuple[tuple[int, int], ...]:
    half = BOARD_SIZE // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


SHED_ACCESS_X = jnp.array([t[0] for t in shed_access_tiles()], dtype=jnp.int32)
SHED_ACCESS_Y = jnp.array([t[1] for t in shed_access_tiles()], dtype=jnp.int32)


def is_shed_adjacent(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """True where a unit stands on one of the four shed-access tiles."""
    return ((x[..., None] == SHED_ACCESS_X) & (y[..., None] == SHED_ACCESS_Y)).any(
        axis=-1
    )


def default_spawn() -> tuple[int, int]:
    """First shed-access tile inside the NW quadrant."""
    half = BOARD_SIZE // 2
    return (half - 1, half - 1)


def initial_state(batch_size: int, seed: int = 0) -> EnvState:
    """Fresh batched state matching the engine's initial board.

    Only the NW quadrant starts unlocked, the farmer spawns at the engine's
    default position, and the market sits at equilibrium. ``seed`` seeds the
    per-environment RNG that drives weed spawn and shop unlocks; each
    environment in the batch gets a distinct stream.
    """
    b, h, w = batch_size, BOARD_SIZE, BOARD_SIZE

    ys, xs = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    locked = quadrant_of(xs, ys) != QUAD_NW
    kind = jnp.where(locked, KIND_LOCKED, KIND_EMPTY).astype(jnp.int32)
    kind = jnp.broadcast_to(kind, (b, 2, h, w))

    zeros_tile = jnp.zeros((b, 2, h, w), dtype=jnp.int32)
    spawn_x, spawn_y = default_spawn()

    unit_shape = (b, 2, MAX_UNITS)
    # Slot 0 is the main farmer and is always active; hands start inactive.
    active = jnp.zeros(unit_shape, dtype=jnp.int32).at[:, :, 0].set(1)

    return EnvState(
        step=jnp.zeros((b,), dtype=jnp.int32),
        money=jnp.full((b, 2), STARTING_MONEY, dtype=jnp.float32),
        kind=kind,
        crop=jnp.full((b, 2, h, w), CROP_NONE, dtype=jnp.int32),
        planted_day=zeros_tile,
        watered_today=zeros_tile,
        consecutive_unwatered=zeros_tile,
        yield_units=zeros_tile,
        max_lifespan_step=jnp.full((b, 2, h, w), -1, dtype=jnp.int32),
        fertilized_until_day=jnp.full((b, 2, h, w), -1, dtype=jnp.int32),
        animal=jnp.full((b, 2, h, w), ANIMAL_NONE, dtype=jnp.int32),
        placed_day=zeros_tile,
        consecutive_unfed=zeros_tile,
        fed_today=zeros_tile,
        cared_today=zeros_tile,
        fertilizer_available=zeros_tile,
        pending_care_bonus=zeros_tile,
        unit_x=jnp.full(unit_shape, spawn_x, dtype=jnp.int32),
        unit_y=jnp.full(unit_shape, spawn_y, dtype=jnp.int32),
        unit_active=active,
        hires_today=jnp.zeros((b, 2), dtype=jnp.int32),
        shed=jnp.zeros((b, 2, N_PRODUCTS), dtype=jnp.int32),
        animal_shed=jnp.zeros((b, 2, N_ANIMALS), dtype=jnp.int32),
        carried=jnp.zeros((b, 2, MAX_UNITS, N_PRODUCTS), dtype=jnp.int32),
        carried_animals=jnp.zeros((b, 2, MAX_UNITS, N_ANIMALS), dtype=jnp.int32),
        seeds=jnp.zeros((b, 2, N_CROPS), dtype=jnp.int32),
        lands_bought=jnp.zeros((b, 2), dtype=jnp.int32),
        market_inv=jnp.full((b, N_PRODUCTS), MARKET_I0, dtype=jnp.int32),
        shops=jnp.zeros((b, N_SHOPS), dtype=jnp.int32),
        rng=jax.vmap(jax.random.PRNGKey)(seed + jnp.arange(b, dtype=jnp.uint32)),
    )


def tree_flatten_check(state: EnvState) -> None:
    """Assert every leaf carries the same batch size (cheap sanity guard)."""
    sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(state)}
    if len(sizes) != 1:
        raise ValueError(f"inconsistent batch sizes across state leaves: {sizes}")
