"""rulebase/case6 -- single-tile wheat loop.

Ported from ``delayedkarma__whatwheatwrought-a-kaggriculture-tale``. The
smallest coherent agent in the notebook set: buy a wheat seed, plant it, water
it, harvest on day 2, sell, repeat. No hands, no land, no animals.

The premise is that wheat has the fastest cycle of any crop (first yield on day
2 versus melon's day 10), so a single farmer working one tile at a time still
turns money over continuously. It is a floor, not a strategy -- one farmer can
work exactly one tile per turn, so throughput is capped regardless of how fast
the crop is.

Ported as-is apart from the never-raise wrapper and defensive key access. The
original's movement scan is preserved, including its quirk of walking toward
already-planted tiles.
"""

from __future__ import annotations

from typing import Any

#: From CROPS["WHEAT"] in kaggriculture.py at 1.32.6.
WHEAT_SEED_COST = 10
WHEAT_FIRST_YIELD_DAY = 2

BOARD_SIZE = 10


def _decide(obs: Any) -> dict[str, Any]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    private = obs.get("private", {}) or {}

    day = obs.get("day", 0)
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    money = farm["money"]
    market: list[list[Any]] = []

    # Sell the whole wheat stock. Wheat's glut curve is shallow (log, T=400), so
    # dumping it costs far less than it would for a premium good.
    if shed.get("WHEAT", 0) > 0:
        market.append(["SELL", "WHEAT", shed["WHEAT"]])

    # Keep exactly one seed in hand.
    if seeds.get("WHEAT", 0) == 0 and money >= WHEAT_SEED_COST:
        market.append(["BUY_SEED", "WHEAT", 1])

    if tile is None and seeds.get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        age = day - tile["planted_day"]
        if age >= WHEAT_FIRST_YIELD_DAY and tile["yield_units"] > 0:
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}

    # Walk toward the first empty or planted tile in scan order.
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            t = farm["tiles"][y][x]
            if t is None or (isinstance(t, dict) and t.get("kind") == "PLANT"):
                if x != fx:
                    step = "EAST" if x > fx else "WEST"
                    return {"farmer": [step], "hands": [], "market": market}
                if y != fy:
                    step = "SOUTH" if y > fy else "NORTH"
                    return {"farmer": [step], "hands": [], "market": market}

    return {"farmer": ["PASS"], "hands": [], "market": market}


def agent(obs: Any) -> dict[str, Any]:
    """Kaggriculture agent entry point."""
    try:
        return _decide(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
