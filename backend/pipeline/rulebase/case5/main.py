"""rulebase/case5 -- "Melon Maxxer", the official getting-started baseline.

Ported from the competition tutorial notebook
(``bovard__kaggriculture-getting-started``). A single farmer, melon monocrop, no
hands, no land purchases, and a fixed price threshold below which it will not
sell.

Kept deliberately close to the original: this case exists as a reference point
for what the tutorial strategy is actually worth, not as a competitive entry.
Two changes were necessary:

* the notebook imports ``CROPS`` from ``kaggle_environments``; the two constants
  it needs are inlined instead, because the submission archive cannot rely on
  engine internals staying importable,
* the top level is wrapped so the agent cannot raise (an uncaught exception
  forfeits the episode).

Known weaknesses, several of which the tutorial names itself: it never hires
(the most expensive mistake in the meta), it works a single tile at a time, it
dumps the whole shed in one order and craters its own price, and it never
liquidates before the season ends.
"""

from __future__ import annotations

from typing import Any

#: From CROPS["MELON"] in kaggriculture.py at 1.32.6.
MELON_SEED_COST = 80
MELON_MAX_YIELD_DAY = 12

#: The notebook's fixed floor: hold melons unless the market pays at least this.
SELL_THRESHOLD = 200


def _step_toward(fx: int, fy: int, tx: int, ty: int) -> str | None:
    """Greedy axis-step, x before y. No pathfinding in the original."""
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _find_target_tile(
    farm: dict[str, Any], board_size: int, have_seed: bool
) -> tuple[int, int, str] | None:
    """Nearest tile needing work, harvest first, then water, then plant."""
    fx, fy = farm["farmer"]
    candidates: list[tuple[int, int, str]] = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "MELON"
            ):
                purpose = None
                ripe = tile.get("yield_units", 0) > 0
                if ripe and tile.get("planted_day") is not None:
                    purpose = "harvest"
                if not tile.get("watered_today"):
                    purpose = "water" if purpose is None else purpose
                if purpose:
                    candidates.append((x, y, purpose))
            elif tile is None and have_seed:
                candidates.append((x, y, "plant"))

    if not candidates:
        return None

    priority = {"harvest": 0, "water": 1, "plant": 2}
    candidates.sort(key=lambda c: (priority[c[2]], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def _decide(obs: Any) -> dict[str, Any]:
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)

    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    market_prices = (obs.get("market", {}) or {}).get("prices", {})
    melon_price = market_prices.get("MELON", 0)

    market: list[list[Any]] = []

    # Sell melons only when the market is paying enough.
    melons_in_shed = shed.get("MELON", 0)
    if melons_in_shed > 0 and melon_price >= SELL_THRESHOLD:
        market.append(["SELL", "MELON", melons_in_shed])

    # Top up seed inventory so the next empty tile can be planted.
    if seeds.get("MELON", 0) == 0 and farm["money"] >= MELON_SEED_COST:
        market.append(["BUY_SEED", "MELON", 1])

    farmer: list[Any] = ["PASS"]

    if (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "MELON"
    ):
        age = day - tile["planted_day"]
        if age >= MELON_MAX_YIELD_DAY and tile["yield_units"] > 0:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
        else:
            target = _find_target_tile(farm, board_size, seeds.get("MELON", 0) > 0)
            if target:
                step = _step_toward(fx, fy, target[0], target[1])
                if step:
                    farmer = [step]
    elif tile is None and seeds.get("MELON", 0) > 0:
        farmer = ["PLANT", "MELON"]
    else:
        target = _find_target_tile(farm, board_size, seeds.get("MELON", 0) > 0)
        if target:
            step = _step_toward(fx, fy, target[0], target[1])
            if step:
                farmer = [step]

    return {"farmer": farmer, "hands": [], "market": market}


def agent(obs: Any) -> dict[str, Any]:
    """Kaggriculture agent entry point."""
    try:
        return _decide(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
