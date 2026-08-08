"""Rulebase case1 — wheat loop.

The simplest agent that actually earns money: keep the farmer on one tile,
plant wheat, water it through the bonus window, harvest, sell.

Wheat is chosen deliberately as the first baseline. It is the cheapest seed
(10), reaches first yield fastest (2 days), and its market curve is the most
forgiving on the sell side (``above_func=log, above_target=0.20``), so dumping
the whole harvest at once barely moves the price. Premium crops would earn more
per unit but collapse to the $1 floor when sold in bulk, which a market-blind
agent like this one cannot avoid.

Deliberately does NOT: hire hands, buy land, keep animals, or time the market.
Those belong to later cases so their effect can be measured against this floor.
"""

from __future__ import annotations

from typing import Any

# Action returned when anything at all goes wrong. Always legal.
SAFE_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

WHEAT = "WHEAT"
WHEAT_SEED_COST = 10
#: Wheat yields from day 2 and stops improving at day 4; the watering bonus
#: window is ceil(4/2)=2 through 4. Harvesting at 4 takes the full bonus.
WHEAT_HARVEST_AGE = 4
#: The engine silently discards market orders past this many per turn.
MAX_MARKET_ORDERS = 10


def _tile_at(farm: dict[str, Any], x: int, y: int) -> Any:
    """Return the tile at (x, y), or None if out of bounds."""
    tiles = farm.get("tiles") or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _build_market_orders(
    money: float, shed: dict[str, Any], seeds: dict[str, Any]
) -> list[list[Any]]:
    """Sell the harvest, then top up seed stock.

    Selling comes first: it is what generates money, and if the order list were
    ever truncated we would rather lose a purchase than a sale.
    """
    orders: list[list[Any]] = []

    wheat_in_shed = int(shed.get(WHEAT, 0) or 0)
    if wheat_in_shed > 0:
        orders.append(["SELL", WHEAT, wheat_in_shed])

    if int(seeds.get(WHEAT, 0) or 0) == 0 and money >= WHEAT_SEED_COST:
        orders.append(["BUY_SEED", WHEAT, 1])

    return orders[:MAX_MARKET_ORDERS]


def _choose_farmer_op(tile: Any, seeds: dict[str, Any], day: int) -> list[Any]:
    """Decide the farmer's single op for this turn."""
    # Empty tile: plant if a seed is on hand.
    if tile is None:
        if int(seeds.get(WHEAT, 0) or 0) > 0:
            return ["PLANT", WHEAT]
        return ["PASS"]

    if not isinstance(tile, dict):
        # "LOCKED" — should not happen at the start tile, but never act on it.
        return ["PASS"]

    kind = tile.get("kind")

    if kind == "WEED":
        return ["DIG"]

    if kind == "PLANT":
        age = day - int(tile.get("planted_day", day))
        if age >= WHEAT_HARVEST_AGE:
            return ["HARVEST"]
        # Watering is what builds the yield bonus, and a plant left dry for two
        # consecutive days turns into a weed.
        if not tile.get("watered_today", False):
            return ["WATER"]
        return ["PASS"]

    return ["PASS"]


def agent(obs: Any) -> dict[str, Any]:
    """Entry point called once per turn by the engine."""
    try:
        player = int(obs["player"])
        day = int(obs.get("day", 0))
        farm = obs["farms"][player]
        private = obs.get("private") or {}

        shed = private.get("shed") or {}
        seeds = private.get("seeds") or {}
        money = float(farm.get("money", 0.0))

        fx, fy = farm["farmer"]
        tile = _tile_at(farm, int(fx), int(fy))

        return {
            "farmer": _choose_farmer_op(tile, seeds, day),
            # This case never hires, so there are never any hands to command.
            "hands": [],
            "market": _build_market_orders(money, shed, seeds),
        }
    except Exception:
        # An uncaught exception forfeits the episode; a wasted turn does not.
        return dict(SAFE_ACTION)
