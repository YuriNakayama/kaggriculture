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
# Feature flags (case7)
# --------------------------------------------------------------------------
#
# case7 lifts three mechanisms out of case3 (a 2005-line port whose score cannot
# be attributed to any single part). Each is gated by a flag so its individual
# contribution can be measured by re-running with one flag flipped, rather than
# inferred. The shipped configuration has all three on.

#: Scale the herd and the workforce with the farm instead of pinning them at
#: 2 cows / 6 hands. case3 finishes on 15 animals and 13 hands and earns 39% of
#: its revenue from MILK + WOOL; case2 earns 23% from a 2-cow herd.
ENABLE_HERD_EXPANSION: bool = True

#: Size the melon planting from the melon price and the opponent's melon tile
#: count, rather than a fixed per-quadrant number.
#:
#: OFF: measured as *exactly* neutral -- 47,356 with and without, to the dollar,
#: over 30 episodes. No opponent available here ever trips a threshold: `starter`
#: grows only carrots and case5 peaks at one melon tile, against a MIN branch
#: that needs 12. The logic is sound in principle (it is case3's only
#: opponent-aware mechanism) but is untested code on this ladder, so it stays
#: off until there is an opponent that can exercise it.
ENABLE_OPPONENT_AWARE_MELON: bool = False

#: Grow STRAWBERRY.
#:
#: OFF: measured at **-1,919** over 30 episodes. The hypothesis was that
#: strawberry supplies 22% of case3's revenue against 0% of case2's, so adding
#: it should pay. It does not, because case7 cannot use it the way case3 does:
#: strawberry needs 10 days to first yield and then produces on a 2-day
#: interval, so 6-7 tiles sat occupied for most of the season and returned 24
#: units total, while displacing melon that would have earned more on the same
#: ground. Kept behind the flag as a recorded negative result.
ENABLE_STRAWBERRY: bool = False

# --------------------------------------------------------------------------
# Strategy tuning
# --------------------------------------------------------------------------

#: Hands hired each morning when herd expansion is off. Fibonacci cost
#: 1+1+2+3+5+8 = 20/day for 6.
TARGET_HANDS: Final[int] = 6

#: Workforce bounds when herd expansion is on. Hire cost is fib(n) for the n-th
#: hire of the day, so 12 hands cost 376/day and the 13th adds 233 by itself.
MIN_HANDS: Final[int] = 6
MAX_HANDS: Final[int] = 12

#: Jobs one hand is expected to clear per turn-budget; used to size the
#: workforce against outstanding work.
JOBS_PER_HAND: Final[int] = 7

#: Extra quadrants to buy, and the day each purchase is attempted.
#: SE ($4000) is deliberately skipped -- near-zero purchase rate in top replays.
LAND_BUY_DAYS: Final[tuple[int, ...]] = (5, 9)

#: Pasture/animal plan when herd expansion is off. COW only: MILK is demanded by
#: 3 shop types, and the 2-day interval pays back faster than SHEEP's 3-day.
TARGET_COWS: Final[int] = 2

#: Herd ramp when expansion is on. Cows lead (2-day interval vs sheep's 3), with
#: sheep mixed in so MILK and WOOL sit on independent price curves and a glut in
#: one does not drag the other down.
#:
#: The ramp starts small on purpose. A cow costs $400 and first yields on day 8,
#: so livestock bought on day 0 is dead capital for a third of the season while
#: still eating a wheat a day. Buying four up front off a $3,000 start empties
#: the treasury before the first harvest, which then starves hiring and feed --
#: measured at ~29 animals lost to starvation and a final bank *below* case2's.
HERD_SEQUENCE: Final[tuple[str, ...]] = ("COW", "COW", "COW", "SHEEP")
CORE_HERD: Final[int] = 2
MID_HERD: Final[int] = 8
TARGET_HERD: Final[int] = 14
HERD_EXPANSION_DAY: Final[int] = 6
HERD_FINAL_DAY: Final[int] = 12

#: Buying an animal after this day cannot repay itself before the season ends
#: (COW first yields on day 8, SHEEP on day 6).
ANIMAL_PURCHASE_LAST_DAY: Final[int] = 18

#: Pasture slots per quadrant. Kept away from the shed-access tiles so the
#: logistics loop is never blocked.
ANIMAL_SLOTS_PER_QUADRANT: Final[int] = 5

#: Seed crop mix, expressed *per unlocked quadrant* so the plan grows with the
#: farm instead of leaving newly bought land empty. WHEAT doubles as animal feed
#: and is demanded by 5 shop types; MELON is the high-base cash crop.
WHEAT_TILES_PER_QUADRANT: Final[int] = 8
MELON_TILES_PER_QUADRANT: Final[int] = 10
STRAWBERRY_TILES_PER_QUADRANT: Final[int] = 4

#: Opponent-aware melon sizing bounds, applied per quadrant.
MELON_TILES_MIN: Final[int] = 8
MELON_TILES_MAX: Final[int] = 12
#: Above this melon price we expand; at or below the low mark we contract.
MELON_PRICE_HIGH: Final[int] = 300
MELON_PRICE_LOW: Final[int] = 170
#: Opponent melon tile counts that trigger contraction.
MELON_OPPONENT_CROWDED: Final[int] = 12
MELON_OPPONENT_BUSY: Final[int] = 9
MELON_OPPONENT_SPARSE: Final[int] = 5

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

#: Cash held back from livestock purchases so seed and feed buying never
#: starves. An unfed animal escapes after two days, losing the whole $400-500
#: purchase, so buying one animal fewer is always cheaper than failing to feed
#: the ones already owned.
CASH_RESERVE: Final[int] = 400

#: Extra reserve per animal already owned, on top of CASH_RESERVE. Wheat costs
#: ~$25, so this keeps several days of feed buyable for the whole herd even if
#: every crop fails.
CASH_RESERVE_PER_ANIMAL: Final[int] = 120

#: Liquidation window. The interpreter marks DONE after step 718, so an action
#: planned for step 719 never executes -- everything must be sold by then.
LIQUIDATION_START_STEP: Final[int] = 700
FINAL_EXECUTED_STEP: Final[int] = 718
