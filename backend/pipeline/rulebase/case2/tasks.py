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
    BOARD_SIZE,
    CROPS,
    FEED_RESERVE_PER_ANIMAL,
    LAST_DAY,
    MELON_TILES_PER_QUADRANT,
    TARGET_COWS,
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


def crop_targets(snap: Snapshot) -> dict[str, int]:
    """Tile count wanted per crop, scaled by how much land is unlocked."""
    quadrants = max(1, len(snap.unlocked_quadrants))
    return {
        "MELON": MELON_TILES_PER_QUADRANT * quadrants,
        "WHEAT": WHEAT_TILES_PER_QUADRANT * quadrants,
    }


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


def _pasture_targets(snap: Snapshot) -> int:
    """Number of pastures still to build."""
    existing = sum(1 for t in snap.all_tiles() if t.kind in ("PASTURE", "ANIMAL"))
    return max(0, TARGET_COWS - existing)


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
                    tasks.append(Task(P_FEED, t.pos, ["FEED"], needs_item="WHEAT"))
            if t.yield_units > 0:
                tasks.append(Task(P_HARVEST, t.pos, ["HARVEST"]))
            # CARE banks +1 unit paid out on the next fed production day.
            if not t.cared_today:
                tasks.append(Task(P_CARE, t.pos, ["CARE"]))
            if t.fertilizer_available:
                tasks.append(Task(P_COLLECT_FERT, t.pos, ["COLLECT_FERTILIZER"]))

        elif t.kind == "PLANT":
            if not t.watered_today:
                # consecutive_unwatered == 1 means missing today turns it to weed.
                prio = P_RESCUE_WATER if t.consecutive_unwatered >= 1 else P_WATER
                tasks.append(Task(prio, t.pos, ["WATER"]))
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

    carried_cows = sum(inv.get("COW", 0) for inv in snap.inventories)
    free_pastures = [t for t in tiles if t.kind == "PASTURE"]
    if carried_cows > 0:
        for t in free_pastures:
            tasks.append(
                Task(P_PLACE_ANIMAL, t.pos, ["PLACE", "COW"], needs_item="COW")
            )
    elif free_pastures and snap.shed.get("COW", 0) > 0:
        # A bought animal lands in the shed and stays there until someone
        # carries it out, so an empty pasture has to pull it from the shed
        # first. Without this the herd never gets placed.
        take = min(snap.shed.get("COW", 0), len(free_pastures))
        tasks.append(
            Task(P_FETCH_ANIMAL, SHED_TILES[0], ["PICKUP", "COW", take], at_shed=True)
        )

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
            own_zone = task.at_shed or _zone_of(target, n_units) == idx
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
