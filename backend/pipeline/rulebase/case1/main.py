"""Rulebase case1 — multi-tile wheat loop.

The goal of this case is to be a correct, non-crashing baseline that clears
``starter``, not to be clever. ``starter`` farms a single tile of carrot; this
works several tiles of wheat with hired hands.

Wheat is chosen deliberately. Its market shape above the equilibrium ``I0`` is
``log`` with ``above_target = 0.20``, so it is the one product that tolerates a
sustained glut: dumping a full field's output moves the price from $25 to about
$20, whereas the same volume of strawberry or melon hits the $1 floor. A steady
seller beats a high-value one that craters on the second sale.

Engine details this relies on, each verified against the interpreter:

- ``consecutive_unwatered`` starts at **1** on the planting day, and a plant
  turns into a weed once it reaches 2 at end-of-day. So a new plant must be
  watered the same day it goes in — there is no grace period.
- Per-unit inventories are emptied into the shed automatically at end of day,
  so ``HARVEST`` alone is enough; no ``PLACE`` trip to the shed is needed.
- ``SELL`` draws from the shed, not from unit inventories, so harvested wheat is
  only sellable from the next day onward.
- Wheat reaches ``yield_units`` 4 without fertilizer: watering inside the bonus
  window (``max_yield_day + 1) // 2`` through day 4 adds one unit per watering.
  Harvesting a non-ongoing crop clears the tile, freeing it to replant.
"""

from __future__ import annotations

from typing import Any

PASS_TURN: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

WHEAT_SEED_COST = 10
WHEAT_MAX_YIELD_DAY = 4
BOARD_HALF = 5

#: Engine cap on market orders per turn. The 11th is silently discarded, so the
#: order list is built against this budget explicitly.
MAX_MARKET_ORDERS = 10

#: Units of wheat to offer per turn. Small and steady: each unit sells at the
#: pre-sale price, and the opponent is selling into the same book concurrently.
SELL_CHUNK = 6

#: Keep a seed buffer so a free tile is never idle waiting on a purchase.
SEED_BUFFER = 6

#: Below this, stop buying seeds and hiring — running out of cash mid-season is
#: worse than leaving a tile fallow.
CASH_FLOOR = 200

#: Hire while hands are cheap. Cost is fib(n) for the n-th hire of the day, so
#: the first few are nearly free and each one is an extra action per turn.
MAX_HANDS_PER_DAY = 3

#: Liquidate everything over the final days — unsold inventory scores zero.
LIQUIDATION_DAY = 28
FINAL_DAY = 29


def _tile_at(farm: dict[str, Any], x: int, y: int) -> Any:
    tiles = farm.get("tiles") or []
    if 0 <= y < len(tiles):
        row = tiles[y]
        if 0 <= x < len(row):
            return row[x]
    return "LOCKED"


def _is_own_wheat(tile: Any) -> bool:
    return (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "WHEAT"
    )


def _step_toward(pos: tuple[int, int], target: tuple[int, int]) -> list[Any]:
    """One orthogonal step toward ``target``; ``PASS`` when already there."""
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _unit_action(
    tile: Any, day: int, seeds_left: int, in_nw: bool
) -> tuple[list[Any], int]:
    """Decide one unit's op from the tile it stands on.

    Returns the action and how many seeds it consumed, so the caller can keep
    the turn's total ``PLANT`` count within the seed budget — the engine drops
    *every* ``PLANT`` for a crop when the turn over-requests it.

    ``None`` as the action means "nothing to do here", which tells the caller to
    move this unit to a tile that does need work.
    """
    if not in_nw:
        return ["PASS"], 0

    if _is_own_wheat(tile):
        age = day - int(tile.get("planted_day", day))
        if age >= WHEAT_MAX_YIELD_DAY and int(tile.get("yield_units", 0)) > 0:
            return ["HARVEST"], 0
        if not tile.get("watered_today", False):
            # Also covers the planting day, which already counts as unwatered.
            return ["WATER"], 0
        return [], 0

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"], 0

    if tile is None and seeds_left > 0 and day < LIQUIDATION_DAY:
        return ["PLANT", "WHEAT"], 1

    return [], 0


def _as_pos(pos: Any) -> tuple[int, int] | None:
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        return int(pos[0]), int(pos[1])
    except (TypeError, ValueError):
        return None


def _tile_needs_work(tile: Any, day: int, seeds_left: int) -> bool:
    """Would a unit standing here have something useful to do?"""
    if _is_own_wheat(tile):
        age = day - int(tile.get("planted_day", day))
        ripe = age >= WHEAT_MAX_YIELD_DAY and int(tile.get("yield_units", 0)) > 0
        return ripe or not tile.get("watered_today", False)
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return True
    return tile is None and seeds_left > 0 and day < LIQUIDATION_DAY


def _nearest_task_tile(
    farm: dict[str, Any],
    here: tuple[int, int],
    day: int,
    seeds_left: int,
    claimed: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Closest NW tile needing work, by Manhattan distance, excluding claims."""
    best: tuple[int, int] | None = None
    best_distance = -1
    for y in range(BOARD_HALF):
        for x in range(BOARD_HALF):
            if (x, y) in claimed:
                continue
            if not _tile_needs_work(_tile_at(farm, x, y), day, seeds_left):
                continue
            distance = abs(x - here[0]) + abs(y - here[1])
            if best is None or distance < best_distance:
                best, best_distance = (x, y), distance
    return best


def _in_starting_quadrant(pos: Any) -> bool:
    """NW is the only quadrant this case works; tile ops elsewhere are no-ops."""
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False
    try:
        x, y = int(pos[0]), int(pos[1])
    except (TypeError, ValueError):
        return False
    return x < BOARD_HALF and y < BOARD_HALF


def _market_orders(
    farm: dict[str, Any],
    shed: dict[str, Any],
    seeds: dict[str, Any],
    day: int,
    hires_today: int,
) -> list[list[Any]]:
    """Build the turn's orders, highest priority first, capped at the engine limit."""
    orders: list[list[Any]] = []
    money = int(farm.get("money", 0) or 0)
    stock = int(shed.get("WHEAT", 0) or 0)

    if stock > 0:
        if day >= LIQUIDATION_DAY:
            # Nothing carries over past the final turn, so dump the lot even at
            # a worse price. Spread across the last days to soften the impact.
            chunk = stock if day >= FINAL_DAY else max(SELL_CHUNK, stock // 2)
            orders.append(["SELL", "WHEAT", chunk])
        else:
            orders.append(["SELL", "WHEAT", min(SELL_CHUNK, stock)])

    if day < LIQUIDATION_DAY and money > CASH_FLOOR:
        held = int(seeds.get("WHEAT", 0) or 0)
        shortfall = SEED_BUFFER - held
        affordable = (money - CASH_FLOOR) // WHEAT_SEED_COST
        want = min(shortfall, affordable)
        if want > 0:
            orders.append(["BUY_SEED", "WHEAT", want])

        # Hands only last the day, so hire early and let them water/harvest.
        if hires_today < MAX_HANDS_PER_DAY and money > CASH_FLOOR * 2:
            orders.append(["HIRE"])

    return orders[:MAX_MARKET_ORDERS]


def _decide(obs: dict[str, Any]) -> dict[str, Any]:
    farms = obs.get("farms") or []
    player = int(obs.get("player", 0) or 0)
    if player >= len(farms):
        return PASS_TURN

    farm = farms[player] or {}
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    day = int(obs.get("day", 0) or 0)

    hands = farm.get("hands") or []
    positions = [farm.get("farmer"), *hands]

    seeds_left = int(seeds.get("WHEAT", 0) or 0)
    # Tiles another unit is already standing on or heading for. Without this,
    # every idle unit walks to the same nearest tile and only one gets to act.
    claimed: set[tuple[int, int]] = set()
    unit_actions: list[list[Any]] = []

    for pos in positions:
        here = _as_pos(pos)
        in_nw = _in_starting_quadrant(pos)
        tile: Any = "LOCKED"
        if in_nw and here is not None:
            tile = _tile_at(farm, here[0], here[1])

        action, used = _unit_action(tile, day, seeds_left, in_nw)
        if action:
            seeds_left -= used
            if here is not None:
                claimed.add(here)
            unit_actions.append(action)
            continue

        # Nothing to do underfoot: walk to the nearest tile that needs work.
        if here is None:
            unit_actions.append(["PASS"])
            continue
        target = _nearest_task_tile(farm, here, day, seeds_left, claimed)
        if target is None:
            unit_actions.append(["PASS"])
            continue
        claimed.add(target)
        unit_actions.append(_step_toward(here, target))

    return {
        "farmer": unit_actions[0] if unit_actions else ["PASS"],
        # Exactly one entry per hired hand, in order — a mismatch silently
        # drops actions.
        "hands": unit_actions[1 : len(hands) + 1],
        "market": _market_orders(
            farm, shed, seeds, day, int(farm.get("hires_today", 0) or 0)
        ),
    }


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    """Entry point. Must never raise: an exception forfeits the episode."""
    try:
        return _decide(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
