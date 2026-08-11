"""Parse the raw ``obs`` dict into a typed snapshot.

Every access is defensive: a missing or malformed key yields a usable default
rather than raising, because an exception inside the agent forfeits the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import ANIMALS, BOARD_SIZE, TURNS_PER_DAY

Pos = tuple[int, int]


@dataclass(frozen=True)
class Tile:
    """One board cell, normalised into a flat shape."""

    x: int
    y: int
    #: "LOCKED" | "EMPTY" | "WEED" | "PLANT" | "COOP" | "PASTURE" | "ANIMAL"
    kind: str
    crop: str | None = None
    animal: str | None = None
    yield_units: int = 0
    planted_day: int = 0
    watered_today: bool = False
    consecutive_unwatered: int = 0
    fed_today: bool = False
    cared_today: bool = False
    consecutive_unfed: int = 0
    fertilizer_available: bool = False

    @property
    def pos(self) -> Pos:
        return (self.x, self.y)

    @property
    def is_empty(self) -> bool:
        return self.kind == "EMPTY"


@dataclass
class Snapshot:
    """Everything the policy needs for one turn."""

    step: int
    day: int
    hour: int
    player: int
    money: float
    tiles: list[list[Tile]]
    farmer: Pos
    hands: list[Pos]
    unlocked_quadrants: list[str]
    hires_today: int
    shed: dict[str, int]
    seeds: dict[str, int]
    inventories: list[dict[str, int]]
    prices: dict[str, int]
    market_inventory: dict[str, int]
    unlocked_shops: list[str]
    opponent_tiles: list[list[Tile]] = field(default_factory=list)
    opponent_money: float = 0.0

    @property
    def units(self) -> list[Pos]:
        """Farmer first, then hands, matching the action-dict ordering."""
        return [self.farmer, *self.hands]

    def tile_at(self, pos: Pos) -> Tile | None:
        x, y = pos
        if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]):
            return self.tiles[y][x]
        return None

    def inventory_of(self, unit_idx: int) -> dict[str, int]:
        if 0 <= unit_idx < len(self.inventories):
            return self.inventories[unit_idx]
        return {}

    def all_tiles(self) -> list[Tile]:
        return [t for row in self.tiles for t in row]

    def opponent_crop_tiles(self, crop: str) -> int:
        """How many tiles the opponent is growing ``crop`` on.

        Farms are public, so this is legitimate information: it is the only
        read on what the opponent will shortly dump into the shared market.
        """
        return sum(
            1
            for row in self.opponent_tiles
            for t in row
            if t.kind == "PLANT" and t.crop == crop
        )


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_pos(value: Any, default: Pos = (4, 4)) -> Pos:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_as_int(value[0]), _as_int(value[1]))
    return default


def _parse_tile(raw: Any, x: int, y: int) -> Tile:
    if raw == "LOCKED":
        return Tile(x=x, y=y, kind="LOCKED")
    if raw is None:
        return Tile(x=x, y=y, kind="EMPTY")
    if not isinstance(raw, dict):
        return Tile(x=x, y=y, kind="EMPTY")

    kind = str(raw.get("kind", ""))
    if "animal" in raw:
        return Tile(
            x=x,
            y=y,
            kind="ANIMAL",
            animal=str(raw.get("animal", "")),
            yield_units=_as_int(raw.get("yield_units")),
            fed_today=bool(raw.get("fed_today", False)),
            cared_today=bool(raw.get("cared_today", False)),
            consecutive_unfed=_as_int(raw.get("consecutive_unfed")),
            fertilizer_available=bool(raw.get("fertilizer_available", False)),
        )
    if kind == "PLANT":
        return Tile(
            x=x,
            y=y,
            kind="PLANT",
            crop=str(raw.get("crop", "")),
            yield_units=_as_int(raw.get("yield_units")),
            planted_day=_as_int(raw.get("planted_day")),
            watered_today=bool(raw.get("watered_today", False)),
            consecutive_unwatered=_as_int(raw.get("consecutive_unwatered")),
        )
    if kind in ("COOP", "PASTURE", "WEED"):
        return Tile(x=x, y=y, kind=kind)
    return Tile(x=x, y=y, kind="EMPTY")


def _parse_grid(raw_tiles: Any) -> list[list[Tile]]:
    grid: list[list[Tile]] = []
    if not isinstance(raw_tiles, list):
        raw_tiles = []
    for y in range(BOARD_SIZE):
        row_raw = (
            raw_tiles[y]
            if y < len(raw_tiles) and isinstance(raw_tiles[y], list)
            else []
        )
        row = [
            _parse_tile(row_raw[x] if x < len(row_raw) else "LOCKED", x, y)
            for x in range(BOARD_SIZE)
        ]
        grid.append(row)
    return grid


def _parse_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): _as_int(v) for k, v in raw.items()}


def parse(obs: Any) -> Snapshot:
    """Build a Snapshot from the engine observation.

    ``obs`` behaves like a dict but may be a Struct; ``.get`` is available on
    both, so access goes exclusively through it.
    """
    get = obs.get if hasattr(obs, "get") else (lambda k, d=None: getattr(obs, k, d))

    player = _as_int(get("player", 0))
    day = _as_int(get("day", 0))
    hour = _as_int(get("hour", 0))
    step = _as_int(get("step", day * TURNS_PER_DAY + hour))

    farms = get("farms", []) or []
    if not isinstance(farms, list) or player >= len(farms):
        farms = [{}, {}]
        player = 0
    farm = farms[player] if isinstance(farms[player], dict) else {}
    opp_idx = 1 - player if len(farms) > 1 else player
    opp_farm = farms[opp_idx] if isinstance(farms[opp_idx], dict) else {}

    private = get("private", {}) or {}
    if not isinstance(private, dict):
        private = {}

    market = get("market", {}) or {}
    if not isinstance(market, dict):
        market = {}

    town = get("town", {}) or {}
    shops = town.get("unlocked_shops", []) if isinstance(town, dict) else []

    raw_inventories = private.get("inventories", [])
    inventories = (
        [_parse_counts(inv) for inv in raw_inventories]
        if isinstance(raw_inventories, list)
        else []
    )

    raw_hands = farm.get("hands", [])
    hands = [_as_pos(p) for p in raw_hands] if isinstance(raw_hands, list) else []

    quads = farm.get("unlocked_quadrants", ["NW"])
    if not isinstance(quads, list):
        quads = ["NW"]

    return Snapshot(
        step=step,
        day=day,
        hour=hour,
        player=player,
        money=float(_as_int(farm.get("money", 0))),
        tiles=_parse_grid(farm.get("tiles")),
        farmer=_as_pos(farm.get("farmer")),
        hands=hands,
        unlocked_quadrants=[str(q) for q in quads],
        hires_today=_as_int(farm.get("hires_today")),
        shed=_parse_counts(private.get("shed")),
        seeds=_parse_counts(private.get("seeds")),
        inventories=inventories,
        prices=_parse_counts(market.get("prices")),
        market_inventory=_parse_counts(market.get("inventory")),
        unlocked_shops=[str(s) for s in shops] if isinstance(shops, list) else [],
        opponent_tiles=_parse_grid(opp_farm.get("tiles")),
        opponent_money=float(_as_int(opp_farm.get("money", 0))),
    )


def animal_product(animal: str | None) -> str | None:
    """Product an animal yields, or None if unknown."""
    if animal and animal in ANIMALS:
        return ANIMALS[animal]["product"]
    return None
