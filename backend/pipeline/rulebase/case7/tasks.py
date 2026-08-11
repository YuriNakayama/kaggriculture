"""Task generation and per-unit assignment.

The board is partitioned into per-unit zones so that units never contend for the
same tile: unit *i* only ever claims tasks inside zone *i*. This is what keeps
every hired hand busy without any collision handling.

Tasks carry a priority (lower = more urgent). The ordering encodes the two
deadlines the engine enforces at end of day:

* ``consecutive_unwatered >= 2`` turns a plant into a weed,
* ``consecutive_unfed >= 2`` makes an animal escape.

so FEED and rescue-WATER outrank everything that merely grows the farm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

from config import (
    ANIMALS,
    ANIMAL_SLOTS_PER_QUADRANT,
    BOARD_SIZE,
    CORE_HERD,
    CROPS,
    ENABLE_HERD_EXPANSION,
    ENABLE_OPPONENT_AWARE_MELON,
    ENABLE_STRAWBERRY,
    FEED_RESERVE_PER_ANIMAL,
    HERD_EXPANSION_DAY,
    HERD_FINAL_DAY,
    HERD_SEQUENCE,
    LAST_DAY,
    MARKET_PARAMS,
    MELON_OPPONENT_BUSY,
    MELON_OPPONENT_CROWDED,
    MELON_OPPONENT_SPARSE,
    MELON_PRICE_HIGH,
    MELON_PRICE_LOW,
    MELON_TILES_MAX,
    MELON_TILES_MIN,
    MELON_TILES_PER_QUADRANT,
    MID_HERD,
    STRAWBERRY_TILES_PER_QUADRANT,
    TARGET_COWS,
    TARGET_HERD,
    WHEAT_TILES_PER_QUADRANT,
)
from observe import Pos, Snapshot, Tile

#: Shed access tiles -- standing on one of these enables PICKUP / PLACE / DROP.
SHED_TILES: Final[tuple[Pos, ...]] = ((4, 4), (5, 4), (4, 5), (5, 5))

#: Priorities. Lower runs first.
P_FEED: Final[int] = 0
P_FETCH_FEED: Final[int] = 1
P_RESCUE_WATER: Final[int] = 2
P_HARVEST: Final[int] = 3
P_FETCH_ANIMAL: Final[int] = 4
P_PLACE_ANIMAL: Final[int] = 5
P_BUILD_PASTURE: Final[int] = 6
P_WATER: Final[int] = 7
P_CARE: Final[int] = 8
P_PLANT: Final[int] = 9
P_DIG: Final[int] = 10
P_COLLECT_FERT: Final[int] = 11


@dataclass(frozen=True)
class Task:
    """A unit of work at a target tile."""

    priority: int
    target: Pos
    op: list[str | int]
    #: Set when the task consumes a carried item, so assignment can check it.
    needs_item: str | None = None
    #: Shed work (PICKUP) is valid from any of the four access tiles, so the
    #: claiming unit retargets to whichever one is nearest to it.
    at_shed: bool = False
    #: Exempt from the zone preference. Zones spread routine work evenly, but
    #: livestock clusters into a few columns -- with 8 animals across 4 of 8
    #: zones, only the 4 owning units could ever feed, so most of the herd
    #: starved while idle units stood in empty zones. Deadline work is claimable
    #: farm-wide; only growth work is zone-bound.
    zone_free: bool = False


def quadrant_of(x: int, y: int) -> str:
    half = BOARD_SIZE // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def first_step_towards(start: Pos, goal: Pos) -> str | None:
    """BFS the first move from ``start`` toward ``goal``.

    Movement is legal onto any in-bounds tile including LOCKED ones (the engine
    permits it so hands spawning on locked shed tiles are not stranded), so the
    grid is treated as fully connected and BFS is really just a shortest path
    with deterministic tie-breaking.
    """
    if start == goal:
        return None

    order: Final[tuple[tuple[str, int, int], ...]] = (
        ("NORTH", 0, -1),
        ("SOUTH", 0, 1),
        ("EAST", 1, 0),
        ("WEST", -1, 0),
    )
    queue: deque[tuple[Pos, str]] = deque()
    seen: set[Pos] = {start}
    for name, dx, dy in order:
        nxt = (start[0] + dx, start[1] + dy)
        if 0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE:
            seen.add(nxt)
            queue.append((nxt, name))

    while queue:
        pos, first = queue.popleft()
        if pos == goal:
            return first
        for _, dx, dy in order:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen:
                continue
            if not (0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE):
                continue
            seen.add(nxt)
            queue.append((nxt, first))
    return None


def nearest_shed_tile(pos: Pos) -> Pos:
    return min(SHED_TILES, key=lambda t: (_manhattan(pos, t), t))


def workable_tiles(snap: Snapshot) -> list[Tile]:
    """Unlocked tiles, excluding the four shed-access cells.

    The shed tiles stay free so units can always reach the shed to pick up feed
    and drop harvests.
    """
    return [
        t for t in snap.all_tiles() if t.kind != "LOCKED" and t.pos not in SHED_TILES
    ]


def melon_tiles_per_quadrant(snap: Snapshot) -> int:
    """How much melon to grow, given the price and the opponent's melon crop.

    Both players sell into one book, so the melon price at harvest depends on
    the opponent's supply as much as ours. MELON's glut curve is the steepest in
    the game (``sq``, above_target 3.60) and no town shop consumes melon at all,
    so a crowded market has nothing draining it -- worth planting less of.
    """
    if not ENABLE_OPPONENT_AWARE_MELON:
        return MELON_TILES_PER_QUADRANT

    price = snap.prices.get("MELON", int(MARKET_PARAMS["MELON"]["base"]))
    opponent = snap.opponent_crop_tiles("MELON")

    if price >= MELON_PRICE_HIGH and opponent <= MELON_OPPONENT_SPARSE:
        return MELON_TILES_MAX
    if price <= MELON_PRICE_LOW or opponent >= MELON_OPPONENT_CROWDED:
        return MELON_TILES_MIN
    if opponent >= MELON_OPPONENT_BUSY:
        return MELON_TILES_PER_QUADRANT - 1
    return MELON_TILES_PER_QUADRANT


def crop_targets(snap: Snapshot) -> dict[str, int]:
    """Tile count wanted per crop, scaled by how much land is unlocked."""
    quadrants = max(1, len(snap.unlocked_quadrants))
    targets = {
        "MELON": melon_tiles_per_quadrant(snap) * quadrants,
        "WHEAT": WHEAT_TILES_PER_QUADRANT * quadrants,
    }
    if ENABLE_STRAWBERRY:
        targets["STRAWBERRY"] = STRAWBERRY_TILES_PER_QUADRANT * quadrants
    return targets


def plantable(snap: Snapshot, crop: str) -> bool:
    """True while a fresh planting can still reach its first harvest."""
    return snap.day + CROPS[crop]["first_yield_day"] <= LAST_DAY


def _planting_plan(snap: Snapshot) -> dict[str, int]:
    """How many more tiles of each crop we want planted right now."""
    planted: dict[str, int] = {}
    for t in snap.all_tiles():
        if t.kind == "PLANT" and t.crop:
            planted[t.crop] = planted.get(t.crop, 0) + 1

    want: dict[str, int] = {}
    for crop, target in crop_targets(snap).items():
        # MELON needs 10 days to first yield; stop planting once it cannot mature.
        if plantable(snap, crop):
            want[crop] = max(0, target - planted.get(crop, 0))
    return want


def herd_target(snap: Snapshot) -> int:
    """How many animals we want housed right now.

    Ramped rather than bought all at once: an animal costs $400-500 and eats a
    wheat a day from the moment it lands, so buying the full herd on day 0 both
    starves the seed budget and creates a feed demand nothing is producing yet.
    """
    if not ENABLE_HERD_EXPANSION:
        return TARGET_COWS

    if snap.day >= HERD_FINAL_DAY:
        target = TARGET_HERD
    elif snap.day >= HERD_EXPANSION_DAY:
        target = MID_HERD
    else:
        target = CORE_HERD

    # Never plan more animals than there are pasture slots on unlocked land.
    quadrants = max(1, len(snap.unlocked_quadrants))
    return min(target, ANIMAL_SLOTS_PER_QUADRANT * quadrants)


def next_animal(snap: Snapshot) -> str:
    """Which species to buy next, following the ramp order.

    Cows lead (2-day interval against sheep's 3), with sheep mixed in so MILK
    and WOOL sit on independent price curves.
    """
    if not ENABLE_HERD_EXPANSION:
        return "COW"
    owned = sum(1 for t in snap.all_tiles() if t.kind == "ANIMAL")
    owned += sum(snap.shed.get(a, 0) for a in ("COW", "SHEEP"))
    return HERD_SEQUENCE[owned % len(HERD_SEQUENCE)]


def _pasture_targets(snap: Snapshot) -> int:
    """Number of pastures still to build."""
    existing = sum(1 for t in snap.all_tiles() if t.kind in ("PASTURE", "ANIMAL"))
    return max(0, herd_target(snap) - existing)


def feed_needed(snap: Snapshot) -> int:
    """Wheat units to keep on hand for animals."""
    animals = sum(1 for t in snap.all_tiles() if t.kind == "ANIMAL")
    return animals * FEED_RESERVE_PER_ANIMAL


def build_tasks(snap: Snapshot) -> list[Task]:
    """Generate every task worth doing this turn, unordered by unit."""
    tasks: list[Task] = []
    tiles = workable_tiles(snap)

    carried_wheat = sum(inv.get("WHEAT", 0) for inv in snap.inventories)
    shed_wheat = snap.shed.get("WHEAT", 0)

    hungry = 0
    for t in tiles:
        if t.kind == "ANIMAL":
            # Feeding is the hard deadline: two unfed days and the animal is gone.
            if not t.fed_today:
                hungry += 1
                if carried_wheat > 0:
                    tasks.append(
                        Task(
                            P_FEED,
                            t.pos,
                            ["FEED"],
                            needs_item="WHEAT",
                            zone_free=True,
                        )
                    )
            if t.yield_units > 0:
                tasks.append(Task(P_HARVEST, t.pos, ["HARVEST"], zone_free=True))
            # CARE banks +1 unit paid out on the next fed production day.
            if not t.cared_today:
                tasks.append(Task(P_CARE, t.pos, ["CARE"], zone_free=True))
            if t.fertilizer_available:
                tasks.append(Task(P_COLLECT_FERT, t.pos, ["COLLECT_FERTILIZER"]))

        elif t.kind == "PLANT":
            if not t.watered_today:
                # consecutive_unwatered == 1 means missing today turns it to weed.
                rescue = t.consecutive_unwatered >= 1
                prio = P_RESCUE_WATER if rescue else P_WATER
                tasks.append(Task(prio, t.pos, ["WATER"], zone_free=rescue))
            if t.yield_units > 0 and t.crop:
                age = snap.day - t.planted_day
                if age >= CROPS[t.crop]["first_yield_day"]:
                    tasks.append(Task(P_HARVEST, t.pos, ["HARVEST"]))

        elif t.kind == "WEED":
            tasks.append(Task(P_DIG, t.pos, ["DIG"]))

    # Fetching feed is a task in its own right, not something only an idle unit
    # does: an animal with nobody carrying wheat generates no FEED task at all,
    # and two such days in a row lose the animal.
    if hungry > carried_wheat and shed_wheat > 0:
        take = min(shed_wheat, hungry - carried_wheat + FEED_RESERVE_PER_ANIMAL)
        tasks.append(
            Task(P_FETCH_FEED, SHED_TILES[0], ["PICKUP", "WHEAT", take], at_shed=True)
        )

    # Pastures, then cows placed onto them.
    pastures_wanted = _pasture_targets(snap)
    if pastures_wanted > 0:
        empties = sorted(
            (t for t in tiles if t.is_empty),
            key=lambda t: (_manhattan(t.pos, (4, 4)), t.pos),
        )
        for t in empties[:pastures_wanted]:
            tasks.append(Task(P_BUILD_PASTURE, t.pos, ["BUILD_PASTURE"]))

    # Placing the herd. Both species share the PASTURE structure, so whichever
    # animal a unit happens to be carrying can fill any free pasture.
    free_pastures = [t for t in tiles if t.kind == "PASTURE"]
    pasture_species = [
        a for a in ("COW", "SHEEP") if ANIMALS[a]["structure"] == "PASTURE"
    ]

    for species in pasture_species:
        if sum(inv.get(species, 0) for inv in snap.inventories) > 0:
            for t in free_pastures:
                tasks.append(
                    Task(
                        P_PLACE_ANIMAL,
                        t.pos,
                        ["PLACE", species],
                        needs_item=species,
                    )
                )

    if free_pastures:
        # A bought animal lands in the shed and stays there until someone
        # carries it out, so an empty pasture has to pull it from the shed
        # first. Without this the herd never gets placed.
        for species in pasture_species:
            in_shed = snap.shed.get(species, 0)
            carried = sum(inv.get(species, 0) for inv in snap.inventories)
            if in_shed > 0 and carried == 0:
                take = min(in_shed, len(free_pastures))
                tasks.append(
                    Task(
                        P_FETCH_ANIMAL,
                        SHED_TILES[0],
                        ["PICKUP", species, take],
                        at_shed=True,
                    )
                )
                break

    # Planting: only onto empty tiles not already claimed for pasture.
    plan = _planting_plan(snap)
    reserved = {task.target for task in tasks if task.op[0] == "BUILD_PASTURE"}
    empties = [t for t in tiles if t.is_empty and t.pos not in reserved]
    empties.sort(key=lambda t: (_manhattan(t.pos, (4, 4)), t.pos))
    cursor = 0
    for crop, count in sorted(plan.items()):
        available_seed = snap.seeds.get(crop, 0)
        n = min(count, available_seed, len(empties) - cursor)
        for _ in range(max(0, n)):
            tasks.append(Task(P_PLANT, empties[cursor].pos, ["PLANT", crop]))
            cursor += 1

    return tasks


def _zone_of(pos: Pos, n_units: int) -> int:
    """Assign a tile to a unit zone by column band.

    Columns are split into ``n_units`` contiguous vertical bands. This keeps each
    unit's tasks spatially clustered, so travel time stays low and two units
    never target the same tile.
    """
    if n_units <= 1:
        return 0
    band = max(1, BOARD_SIZE // n_units)
    return min(n_units - 1, pos[0] // band)


def assign(snap: Snapshot, tasks: list[Task]) -> list[list[str | int]]:
    """Pick one action per unit (farmer first, then each hand in order).

    Units claim tasks greedily by (priority, travel distance). A task already
    claimed by an earlier unit is removed from the pool, so the returned list
    never contains two units working the same tile.
    """
    units = snap.units
    n_units = len(units)
    actions: list[list[str | int]] = [["PASS"] for _ in range(n_units)]
    if n_units == 0:
        return actions

    pool = sorted(tasks, key=lambda t: (t.priority, t.target))
    claimed: set[Pos] = set()

    for idx, pos in enumerate(units):
        inv = snap.inventory_of(idx)
        best: Task | None = None
        best_key: tuple[int, int, Pos] | None = None

        for task in pool:
            target = nearest_shed_tile(pos) if task.at_shed else task.target
            if task.target in claimed:
                continue
            # A task needing an item is only assignable to a unit carrying it.
            if task.needs_item and inv.get(task.needs_item, 0) <= 0:
                continue
            # Prefer tasks in this unit's own zone; others only as a fallback.
            # Shed work is central and belongs to no zone, so it is exempt.
            own_zone = (
                task.at_shed or task.zone_free or _zone_of(target, n_units) == idx
            )
            key = (
                task.priority + (0 if own_zone else 100),
                _manhattan(pos, target),
                target,
            )
            if best_key is None or key < best_key:
                best, best_key = task, key

        if best is None:
            actions[idx] = _idle_action(snap, idx, pos)
            continue

        claimed.add(best.target)
        goal = nearest_shed_tile(pos) if best.at_shed else best.target
        if pos == goal:
            actions[idx] = list(best.op)
        else:
            move = first_step_towards(pos, goal)
            actions[idx] = [move] if move else ["PASS"]

    return actions


def _idle_action(snap: Snapshot, idx: int, pos: Pos) -> list[str | int]:
    """What a unit does when it has no task.

    Idle units restock feed from the shed so that FEED tasks are assignable on
    later turns, and otherwise deposit whatever they are carrying.
    """
    inv = snap.inventory_of(idx)
    at_shed = pos in SHED_TILES

    if at_shed:
        # Deposit produce so it is sellable (SELL draws from the shed, not
        # from a carried inventory).
        sellable = {k: v for k, v in inv.items() if v > 0 and k != "WHEAT"}
        if sellable:
            item = sorted(sellable)[0]
            return ["PLACE", item, sellable[item]]
        wanted = feed_needed(snap)
        carried = inv.get("WHEAT", 0)
        if carried < wanted and snap.shed.get("WHEAT", 0) > 0:
            take = min(wanted - carried, snap.shed.get("WHEAT", 0))
            if take > 0:
                return ["PICKUP", "WHEAT", take]
        return ["PASS"]

    # Head to the shed when carrying anything, or when feed is needed.
    if inv or feed_needed(snap) > 0:
        move = first_step_towards(pos, nearest_shed_tile(pos))
        if move:
            return [move]
    return ["PASS"]


def animal_cost(animal: str) -> int:
    return ANIMALS[animal]["cost"]
