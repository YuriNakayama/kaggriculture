"""Market order construction.

Orders are capped at ``MAX_MARKET_ORDERS`` and the engine drops overflow
silently *from the tail*, so the list is built in deliberate priority order:
SELL first (it funds everything behind it in the same turn), then the purchases
that unblock production.
"""

from __future__ import annotations

import math

from config import (
    ANIMALS,
    CROPS,
    LAND_BUY_DAYS,
    LAND_PRICES,
    LIQUIDATION_START_STEP,
    MARKET_I0,
    MARKET_PARAMS,
    MAX_MARKET_ORDERS,
    PRICE_FLOOR,
    SEED_BATCH,
    SELL_BATCH,
    TARGET_COWS,
    TARGET_HANDS,
)
from observe import Snapshot
from tasks import crop_targets, feed_needed, plantable

Order = list[str | int]


def _shape(func: str, x: float) -> float:
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    return x


def price_at(item: str, inventory: int) -> int:
    """Replicate the engine's price curve for a hypothetical inventory."""
    p = MARKET_PARAMS.get(item)
    if p is None:
        return PRICE_FLOOR
    base = float(p["base"])
    t = float(p["T"])
    if inventory < MARKET_I0:
        func = str(p["below_func"])
        amp = float(p["below_target"]) * base / _shape(func, t)
        price = base + amp * _shape(func, MARKET_I0 - inventory)
    else:
        func = str(p["above_func"])
        amp = float(p["above_target"]) * base / _shape(func, t)
        price = base - amp * _shape(func, inventory - MARKET_I0)
    return max(PRICE_FLOOR, int(round(price)))


def sale_proceeds(item: str, quantity: int, inventory: int) -> int:
    """Total money from selling ``quantity`` units, walking the curve down.

    Mirrors the engine's per-unit lockstep: each unit is quoted at the inventory
    left by the previous one, and a unit sold at the floor does not add supply.
    """
    total = 0
    inv = inventory
    for _ in range(max(0, quantity)):
        unit_price = price_at(item, inv)
        total += unit_price
        if unit_price > PRICE_FLOOR:
            inv += 1
    return total


def _sell_batch_size(item: str, held: int, inventory: int, liquidating: bool) -> int:
    if held <= 0:
        return 0
    if liquidating:
        return held
    return min(held, SELL_BATCH.get(item, 8))


def _impact_score(item: str, quantity: int, inventory: int) -> float:
    """Rank sells by proceeds per unit of price damage caused.

    Selling into a steep curve costs future revenue, so among equally valuable
    batches the one that moves its own price least goes first.
    """
    if quantity <= 0:
        return 0.0
    proceeds = sale_proceeds(item, quantity, inventory)
    before = price_at(item, inventory)
    after = price_at(item, inventory + quantity)
    damage = max(0, before - after)
    return proceeds / (1.0 + damage)


def _sell_orders(snap: Snapshot, liquidating: bool) -> list[Order]:
    """Sell shed produce, holding back wheat needed as animal feed."""
    reserve = 0 if liquidating else feed_needed(snap)
    candidates: list[tuple[float, str, int]] = []

    for item, held in snap.shed.items():
        if held <= 0 or item in ANIMALS:
            continue
        available = held - reserve if item == "WHEAT" else held
        if available <= 0:
            continue
        inv = snap.market_inventory.get(item, MARKET_I0)
        qty = _sell_batch_size(item, available, inv, liquidating)
        if qty <= 0:
            continue
        candidates.append((_impact_score(item, qty, inv), item, qty))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [["SELL", item, qty] for _, item, qty in candidates]


def _seed_orders(snap: Snapshot) -> list[Order]:
    """Top up seed stock for the crops the planting plan still wants.

    Seeds are bought in batches sized to the number of empty tiles waiting, so
    newly unlocked land gets sown promptly instead of one tile per turn.
    """
    orders: list[Order] = []
    planted: dict[str, int] = {}
    for tile in snap.all_tiles():
        if tile.kind == "PLANT" and tile.crop:
            planted[tile.crop] = planted.get(tile.crop, 0) + 1

    empty_tiles = sum(1 for t in snap.all_tiles() if t.is_empty)
    budget = snap.money

    for crop, target in sorted(crop_targets(snap).items()):
        if not plantable(snap, crop):
            continue
        deficit = target - planted.get(crop, 0)
        if deficit <= 0:
            continue
        want = min(deficit, SEED_BATCH, empty_tiles) - snap.seeds.get(crop, 0)
        if want <= 0:
            continue
        cost = int(CROPS[crop]["seed"]) * want
        if cost > budget:
            want = int(budget // int(CROPS[crop]["seed"]))
            cost = int(CROPS[crop]["seed"]) * want
        if want > 0:
            budget -= cost
            orders.append(["BUY_SEED", crop, want])
    return orders


def _feed_orders(snap: Snapshot) -> list[Order]:
    """Buy wheat when the herd would otherwise go unfed.

    Animals escape after two unfed days, so feed is bought rather than waited
    for whenever home-grown wheat has not caught up.
    """
    need = feed_needed(snap)
    if need <= 0:
        return []
    have = snap.shed.get("WHEAT", 0) + sum(
        inv.get("WHEAT", 0) for inv in snap.inventories
    )
    short = need - have
    if short <= 0:
        return []
    return [["BUY_PRODUCT", "WHEAT", short]]


def _animal_orders(snap: Snapshot) -> list[Order]:
    """Buy cows up to the herd target, once a pasture is ready for them."""
    owned = sum(1 for t in snap.all_tiles() if t.kind == "ANIMAL")
    carried = sum(inv.get("COW", 0) for inv in snap.inventories)
    in_shed = snap.shed.get("COW", 0)
    total = owned + carried + in_shed
    if total >= TARGET_COWS:
        return []
    free_pastures = sum(1 for t in snap.all_tiles() if t.kind == "PASTURE")
    want = min(TARGET_COWS - total, max(0, free_pastures - carried - in_shed))
    if want <= 0:
        return []
    if snap.money < ANIMALS["COW"]["cost"] * want:
        want = int(snap.money // ANIMALS["COW"]["cost"])
    return [["BUY_ANIMAL", "COW", want]] if want > 0 else []


def _hire_orders(snap: Snapshot) -> list[Order]:
    """Hire the day's hands at the first turn of the day.

    Hands disappear at end of day and ``hires_today`` resets, so this fires once
    each morning. Cost is fib(n) for the n-th hire, i.e. 20 total for six.
    """
    if snap.hour != 0:
        return []
    want = TARGET_HANDS - snap.hires_today
    return [["HIRE"] for _ in range(max(0, want))]


def _land_orders(snap: Snapshot) -> list[Order]:
    """Buy the next quadrant on schedule when it is affordable."""
    extra_owned = len(snap.unlocked_quadrants) - 1
    if extra_owned >= len(LAND_BUY_DAYS):
        return []
    if snap.day < LAND_BUY_DAYS[extra_owned]:
        return []
    cost = LAND_PRICES[extra_owned]
    # Keep a working buffer so land never starves seed and feed purchases.
    if snap.money < cost + 500:
        return []
    return [["BUY_LAND"]]


def build_market(snap: Snapshot) -> list[Order]:
    """Assemble this turn's market orders, highest priority first."""
    liquidating = snap.step >= LIQUIDATION_START_STEP

    orders: list[Order] = []
    # SELL leads: proceeds settle before later orders in the same list, so a
    # sale can fund the purchases queued behind it this very turn.
    orders.extend(_sell_orders(snap, liquidating))

    if not liquidating:
        orders.extend(_hire_orders(snap))
        orders.extend(_feed_orders(snap))
        orders.extend(_seed_orders(snap))
        orders.extend(_animal_orders(snap))
        orders.extend(_land_orders(snap))

    return orders[:MAX_MARKET_ORDERS]
