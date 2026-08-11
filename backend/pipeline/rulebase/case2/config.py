"""Static game constants and case2 strategy tuning.

Every value under "Engine constants" is mirrored from
``kaggle_environments/envs/kaggriculture/kaggriculture.py`` (pinned at 1.32.6).
They are duplicated rather than imported because submitted code may not rely on
engine internals staying importable.
"""

from __future__ import annotations

from typing import Final, TypedDict


class CropSpec(TypedDict):
    seed: int
    first_yield_day: int
    max_yield_day: int
    interval: int
    max_yield: int
    ongoing: bool


class AnimalSpec(TypedDict):
    cost: int
    structure: str
    first_yield_day: int
    interval: int
    max_held: int
    product: str


# --------------------------------------------------------------------------
# Engine constants
# --------------------------------------------------------------------------

CROPS: dict[str, CropSpec] = {
    "WHEAT": {
        "seed": 10,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}

ANIMALS: dict[str, AnimalSpec] = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "first_yield_day": 4,
        "interval": 1,
        "max_held": 4,
        "product": "EGG",
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "first_yield_day": 8,
        "interval": 2,
        "max_held": 6,
        "product": "MILK",
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "first_yield_day": 6,
        "interval": 3,
        "max_held": 6,
        "product": "WOOL",
    },
}

PRODUCTS: Final[tuple[str, ...]] = (
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

MARKET_I0: Final[int] = 10_000
PRICE_FLOOR: Final[int] = 1

#: base / T / shape functions per product. `amp = target * base / f(T)`.
MARKET_PARAMS: Final[dict[str, dict[str, float | str]]] = {
    "WHEAT": {
        "base": 25,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "T": 450,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "T": 332,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

#: Town shops and the products each unlocked instance consumes per tick.
SHOPS: Final[dict[str, tuple[str, ...]]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

LAND_ORDER: Final[tuple[str, ...]] = ("NE", "SW", "SE")
LAND_PRICES: Final[tuple[int, ...]] = (1000, 2000, 4000)

BOARD_SIZE: int = 10
TURNS_PER_DAY: Final[int] = 24
SHED_CAPACITY: Final[int] = 100
MAX_MARKET_ORDERS: Final[int] = 10
EPISODE_STEPS: Final[int] = 720

# --------------------------------------------------------------------------
# Strategy tuning (case2)
# --------------------------------------------------------------------------

#: Hands hired each morning. Fibonacci cost 1+1+2+3+5+8 = 20/day for 6.
TARGET_HANDS: Final[int] = 6

#: Extra quadrants to buy, and the day each purchase is attempted.
#: SE ($4000) is deliberately skipped -- near-zero purchase rate in top replays.
LAND_BUY_DAYS: Final[tuple[int, ...]] = (5, 9)

#: Pasture/animal plan. COW only: MILK is demanded by 3 shop types, and the
#: 2-day interval pays back faster than SHEEP's 3-day.
TARGET_COWS: Final[int] = 2

#: Seed crop mix, expressed *per unlocked quadrant* so the plan grows with the
#: farm instead of leaving newly bought land empty. WHEAT doubles as animal feed
#: and is demanded by 5 shop types; MELON is the high-base cash crop.
WHEAT_TILES_PER_QUADRANT: Final[int] = 8
MELON_TILES_PER_QUADRANT: Final[int] = 10

#: Seeds bought per top-up. Sized to fill a quadrant's worth of empty tiles
#: rather than trickling in, which otherwise leaves land fallow for days.
SEED_BATCH: Final[int] = 6

#: Stop planting crops that cannot mature before the season ends.
LAST_DAY: Final[int] = 29

#: Sell batch caps, chosen so a single order does not walk far down the curve.
SELL_BATCH: Final[dict[str, int]] = {
    "MELON": 12,
    "MILK": 6,
    "WOOL": 6,
    "STRAWBERRY": 8,
    "WHEAT": 10,
    "CARROT": 10,
    "TOMATO": 8,
    "EGG": 10,
    "FERTILIZER": 4,
}

#: Wheat kept in the shed as animal feed rather than sold.
FEED_RESERVE_PER_ANIMAL: Final[int] = 4

#: Liquidation window. The interpreter marks DONE after step 718, so an action
#: planned for step 719 never executes -- everything must be sold by then.
LIQUIDATION_START_STEP: Final[int] = 700
FINAL_EXECUTED_STEP: Final[int] = 718
