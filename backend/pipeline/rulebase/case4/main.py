"""rulebase/case4 -- CARE-first animal husbandry.

Ported from ``dariushafshar__care-is-a-4-2x-multiplier-animal-economics``. The
agent grows no crops and runs no market model: it keeps a mixed COW/SHEEP herd
fed, cared for and harvested every day, buys its feed, and drips product into
the market in small batches.

Two species on purpose -- MILK and WOOL sit on independent price curves, so a
glut in one does not drag the other down.

## Why CARE dominates this design

``CARE`` is not a percentage multiplier. Reading ``_daily_refresh_animals`` in
the engine: a day where the animal is both fed and cared for increments
``pending_care_bonus``, and that bonus is paid out in full on the next
production tick -- but only if the animal was fed that day too. Steady state is
therefore ``1 + interval`` units per cycle, so the slower the animal, the more
CARE is worth: measured over 30 days the notebook reports GOOSE 2.07x, COW
3.25x, SHEEP 4.22x.

The notebook's ablation is 12/12 paired seeds in favour of CARE, mean final bank
7.12x. That figure is against the weak ``starter`` baseline and on 1.32.6 -- the
same ablation gives 30.53x on 1.29.3, so it is strongly version-dependent.

## Deliberate limitation

Herd size is capped at 4 COW + 3 SHEEP. The notebook's own 288-episode McNemar
testing found every attempt to grow the herd was *significantly worse*
(sheep 6->8: z = -4.52; cows 8->10: z = -3.53), because product price collapses
faster than volume grows. Its conclusion: "CARE decides which animal you keep.
The town sink decides how many."

Ported as-is apart from the never-raise wrapper, defensive key access, and
removal of a mutation of the observation's shed dict during worker assignment.
"""

from __future__ import annotations

from typing import Any

Pos = tuple[int, int]

SHED_TILES: tuple[Pos, ...] = ((4, 4), (5, 4), (4, 5), (5, 5))

#: Two species on purpose: two independent price curves.
HERD: tuple[tuple[str, int], ...] = (("COW", 4), ("SHEEP", 3))

#: NW is the only quadrant unlocked at the start, so every spot needs x<5, y<5.
#: Ordered by walking distance from the shed-access tile (4,4).
PASTURE_SPOTS: tuple[Pos, ...] = (
    (4, 3),
    (3, 4),
    (3, 3),
    (4, 2),
    (2, 4),
    (2, 3),
    (3, 2),
    (4, 1),
)

PRODUCTS: tuple[str, ...] = ("MILK", "WOOL", "EGG", "FERTILIZER")

#: Batch size for product sales. Premium goods hit the $1 floor after only
#: 60-80 units, so the herd's output is dripped rather than dumped.
SELL_BATCH = 4

MAX_MARKET_ORDERS = 10


def _step(pos: Pos, target: Pos) -> str | None:
    """Greedy axis-step, x before y."""
    x, y = pos
    tx, ty = target
    if x < tx:
        return "EAST"
    if x > tx:
        return "WEST"
    if y < ty:
        return "SOUTH"
    if y > ty:
        return "NORTH"
    return None


def _dist(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _decide(obs: Any) -> dict[str, Any]:
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    me = farms[player]
    priv = obs.get("private", {}) or {}
    tiles = me["tiles"]
    shed = priv.get("shed", {}) or {}
    money = me["money"]
    invs = priv.get("inventories", []) or []

    placed: dict[str, int] = {}
    empty_pastures: list[Pos] = []
    animal_tiles: list[tuple[Pos, dict[str, Any]]] = []
    free_spots: list[Pos] = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if isinstance(t, dict) and "animal" in t:
                placed[t["animal"]] = placed.get(t["animal"], 0) + 1
                animal_tiles.append(((x, y), t))
            elif isinstance(t, dict) and t.get("kind") == "PASTURE":
                empty_pastures.append((x, y))
            elif t is None and (x, y) in PASTURE_SPOTS:
                free_spots.append((x, y))

    n_target = sum(n for _, n in HERD)
    n_structures = len(empty_pastures) + sum(placed.values())
    shed_animals = sum(shed.get(a, 0) for a, _ in HERD)

    # ---------------- market ----------------
    market: list[list[Any]] = []
    carried_wheat = sum(i.get("WHEAT", 0) for i in invs)
    total_wheat = shed.get("WHEAT", 0) + carried_wheat
    n_alive = sum(placed.values())
    # Keep roughly three days of feed in stock. Feed is bought, not grown: this
    # agent plants nothing.
    if total_wheat < max(8, n_alive * 3) and money > 600:
        market.append(["BUY_PRODUCT", "WHEAT", 12])

    for animal, target_count in HERD:
        if placed.get(animal, 0) + shed.get(animal, 0) < target_count and money > 1400:
            market.append(["BUY_ANIMAL", animal, 1])
            break

    for product in ("MILK", "WOOL"):
        held = shed.get(product, 0)
        if held > 0:
            market.append(["SELL", product, min(held, SELL_BATCH)])

    if obs.get("day", 0) >= 1 and me.get("hires_today", 0) < 3 and money > 2000:
        market.append(["HIRE"])

    # ---------------- workers ----------------
    workers: list[Pos] = [tuple(me["farmer"])] + [tuple(p) for p in me.get("hands", [])]
    claimed: set[Pos] = set()
    ops: list[list[Any]] = []

    # Local tally of shed animals so two workers in the same turn cannot both
    # claim the last one. The original decremented the observation's own dict.
    shed_claimable = {a: shed.get(a, 0) for a, _ in HERD}

    for wi, pos in enumerate(workers):
        inv = invs[wi] if wi < len(invs) else {}
        x, y = pos
        here = tiles[y][x]
        carrying_animal = next((a for a, _ in HERD if inv.get(a, 0) > 0), None)
        job: list[Any] | None = None

        # --- act where we stand ---
        if isinstance(here, dict) and "animal" in here:
            if here.get("yield_units", 0) > 0:
                job = ["HARVEST"]
            elif not here.get("fed_today") and inv.get("WHEAT", 0) > 0:
                job = ["FEED"]
            elif not here.get("cared_today"):
                job = ["CARE"]
        elif (
            carrying_animal and isinstance(here, dict) and here.get("kind") == "PASTURE"
        ):
            job = ["PLACE", carrying_animal, 1]
        elif here is None and (x, y) in PASTURE_SPOTS and n_structures < n_target:
            job = ["BUILD_PASTURE"]
        elif pos in SHED_TILES:
            if any(inv.get(p, 0) for p in PRODUCTS):
                job = ["DROP"]
            elif not carrying_animal and shed_animals > 0 and empty_pastures:
                for a, _ in HERD:
                    if shed_claimable.get(a, 0) > 0:
                        job = ["PICKUP", a, 1]
                        shed_claimable[a] -= 1
                        break
            elif inv.get("WHEAT", 0) < 3 and shed.get("WHEAT", 0) > 0:
                job = ["PICKUP", "WHEAT", 4]

        # --- otherwise, go somewhere useful ---
        if job is None:
            target: Pos | None = None
            if carrying_animal and empty_pastures:
                target = min(
                    (p for p in empty_pastures if p not in claimed),
                    key=lambda p: _dist(p, pos),
                    default=None,
                )
            if (
                target is None
                and not carrying_animal
                and shed_animals > 0
                and empty_pastures
            ):
                target = min(SHED_TILES, key=lambda s: _dist(s, pos))
            if target is None and n_structures < n_target and free_spots:
                target = min(
                    (s for s in free_spots if s not in claimed),
                    key=lambda s: _dist(s, pos),
                    default=None,
                )
            if target is None and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                target = min(SHED_TILES, key=lambda s: _dist(s, pos))
            if target is None:
                # Score animals by what they need, discounted by walking distance.
                best: Pos | None = None
                best_score = 0.0
                for (ax, ay), t in animal_tiles:
                    if (ax, ay) in claimed:
                        continue
                    score = 0.0
                    if t.get("yield_units", 0) > 0:
                        score += 3
                    if not t.get("fed_today") and inv.get("WHEAT", 0) > 0:
                        score += 2
                    if not t.get("cared_today"):
                        score += 1
                    score -= 0.1 * _dist((ax, ay), pos)
                    if score > best_score:
                        best, best_score = (ax, ay), score
                target = best
            if target is None and any(inv.get(p, 0) for p in PRODUCTS):
                target = min(SHED_TILES, key=lambda s: _dist(s, pos))
            if target is not None:
                claimed.add(target)
                mv = _step(pos, target)
                job = [mv] if mv else ["PASS"]
            else:
                job = ["PASS"]

        ops.append(job)

    return {
        "farmer": ops[0],
        "hands": ops[1:],
        "market": market[:MAX_MARKET_ORDERS],
    }


def agent(obs: Any) -> dict[str, Any]:
    """Kaggriculture agent entry point."""
    try:
        return _decide(obs)
    except Exception:
        try:
            farms = obs.get("farms", []) or []
            player = int(obs.get("player", 0))
            n_hands = len(farms[player].get("hands", []))
            return {
                "farmer": ["PASS"],
                "hands": [["PASS"] for _ in range(n_hands)],
                "market": [],
            }
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
