"""Catch actions the engine would silently discard.

Every invalid action in Kaggriculture is a no-op: a misspelled op, a target
outside the board, an order past the per-turn cap, a ``PLANT`` without seeds.
None of them raise, none of them warn. Bugs therefore show up only as a lower
score, which is indistinguishable from a weaker strategy.

This wrapper sits between the agent and the engine during **evaluation only**
and records the gap between what the agent asked for and what the engine would
actually apply. It never runs inside a submission — ``backend/src/**`` is not
part of the tarball.

Every check mirrors a specific engine behaviour, cited per method below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Ops that move a unit one tile.
MOVE_OPS: frozenset[str] = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})

#: Ops that use the shed, valid only from a shed-access tile.
SHED_OPS: frozenset[str] = frozenset({"PICKUP", "PLACE", "DROP"})

#: Ops acting on the tile the unit stands on.
TILE_OPS: frozenset[str] = frozenset(
    {
        "PLANT",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    }
)

UNIT_OPS: frozenset[str] = MOVE_OPS | SHED_OPS | TILE_OPS | frozenset({"PASS"})

#: Market ops taking ``[op, item, n]``.
MARKET_ITEM_OPS: frozenset[str] = frozenset(
    {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
)

#: Market ops taking no arguments.
MARKET_ATOMIC_OPS: frozenset[str] = frozenset({"HIRE", "BUY_LAND"})

MARKET_OPS: frozenset[str] = MARKET_ITEM_OPS | MARKET_ATOMIC_OPS

CROP_NAMES: frozenset[str] = frozenset(
    {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
)
ANIMAL_NAMES: frozenset[str] = frozenset({"GOOSE", "COW", "SHEEP"})
PRODUCT_NAMES: frozenset[str] = frozenset(
    {
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
    }
)

#: Only these two can be bought back from the market.
BUYABLE_PRODUCTS: frozenset[str] = frozenset({"WHEAT", "FERTILIZER"})

DEFAULT_MAX_MARKET_ORDERS = 10
DEFAULT_BOARD_SIZE = 10


@dataclass(frozen=True)
class ValidationIssue:
    """One action the engine would have discarded."""

    step: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"step {self.step}: {self.kind} — {self.detail}"


def _shed_access_tiles(board_size: int) -> set[tuple[int, int]]:
    """The four inner-corner tiles from which the shed is reachable."""
    half = board_size // 2
    return {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }


def _quadrant_of(x: int, y: int, board_size: int) -> str:
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


class ActionValidator:
    """Wraps an agent and records actions the engine would silently drop."""

    def __init__(self, player: int, *, max_issues: int = 200) -> None:
        self.player = player
        self.max_issues = max_issues
        self._issues: list[ValidationIssue] = []
        self._step = 0

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(self._issues)

    def _record(self, kind: str, detail: str) -> None:
        # Bounded: a systematically broken agent would otherwise log 720 turns
        # worth of identical issues and dominate the report.
        if len(self._issues) < self.max_issues:
            self._issues.append(
                ValidationIssue(step=self._step, kind=kind, detail=detail)
            )

    def wrap(self, agent: Any) -> Callable[[dict[str, Any]], Any]:
        """Return a callable that validates ``agent``'s output then passes it on.

        ``agent`` may be a callable or a path string. Paths are loaded through
        the engine's own loader so the wrapped agent behaves identically to an
        unwrapped one.
        """
        inner = as_callable(agent)

        def wrapped(obs: dict[str, Any]) -> Any:
            action = inner(obs)
            try:
                self.validate(obs, action)
            except (
                Exception
            ) as exc:  # pragma: no cover - validator must not break a run
                self._record("validator_error", repr(exc))
            self._step += 1
            return action

        return wrapped

    def validate(self, obs: dict[str, Any], action: Any) -> None:
        """Check one turn's action against the engine's acceptance rules."""
        self._step = int(_get(obs, "step", self._step) or self._step)

        if not isinstance(action, dict):
            self._record(
                "action_not_dict",
                f"agent returned {type(action).__name__}, engine applies PASS",
            )
            return

        farm = _own_farm(obs, self.player)
        board_size = _board_size(obs, farm)

        self._check_units(obs, action, farm, board_size)
        self._check_market(obs, action, farm)

    def _check_units(
        self,
        obs: dict[str, Any],
        action: dict[str, Any],
        farm: dict[str, Any] | None,
        board_size: int,
    ) -> None:
        farmer = action.get("farmer")
        hands = action.get("hands", [])

        if farmer is None:
            self._record("missing_farmer", "no 'farmer' key; engine applies PASS")
        if not isinstance(hands, list):
            self._record(
                "hands_not_list",
                f"'hands' is {type(hands).__name__}; engine treats it as empty",
            )
            hands = []

        expected_hands = len(_hand_positions(farm))
        if len(hands) != expected_hands:
            self._record(
                "hand_count_mismatch",
                f"{len(hands)} hand action(s) for {expected_hands} hired hand(s)",
            )

        positions = _unit_positions(farm)
        unit_actions = [farmer, *hands]
        for idx, unit_action in enumerate(unit_actions):
            pos = positions[idx] if idx < len(positions) else None
            self._check_unit(idx, unit_action, pos, board_size, farm)

        self._check_plant_budget(obs, unit_actions)

    def _check_unit(
        self,
        idx: int,
        unit_action: Any,
        pos: tuple[int, int] | None,
        board_size: int,
        farm: dict[str, Any] | None,
    ) -> None:
        name = "farmer" if idx == 0 else f"hand[{idx - 1}]"

        if unit_action is None:
            return
        if not isinstance(unit_action, list) or not unit_action:
            self._record(
                "unit_action_malformed",
                f"{name}: expected a non-empty list, got {unit_action!r}",
            )
            return

        op = unit_action[0]
        if not isinstance(op, str) or op not in UNIT_OPS:
            self._record("unknown_unit_op", f"{name}: {op!r} is not a unit op")
            return

        if op == "PLANT" and len(unit_action) < 2:
            self._record("plant_missing_crop", f"{name}: PLANT needs a crop argument")
        elif op == "PLANT" and unit_action[1] not in CROP_NAMES:
            self._record("unknown_crop", f"{name}: PLANT {unit_action[1]!r}")

        if pos is None:
            return
        x, y = pos

        # Movement off the board is a no-op (engine checks bounds before moving).
        if op in MOVE_OPS:
            dx, dy = {
                "NORTH": (0, -1),
                "SOUTH": (0, 1),
                "EAST": (1, 0),
                "WEST": (-1, 0),
            }[op]
            if not (0 <= x + dx < board_size and 0 <= y + dy < board_size):
                self._record(
                    "move_off_board",
                    f"{name}: {op} from ({x},{y}) leaves the "
                    f"{board_size}x{board_size} board",
                )
            return

        # Shed ops resolve before the LOCKED guard but require an access tile.
        if op in SHED_OPS and (x, y) not in _shed_access_tiles(board_size):
            self._record(
                "shed_not_adjacent",
                f"{name}: {op} at ({x},{y}) is not a shed-access tile",
            )
            return

        # Tile ops no-op on locked quadrants.
        if op in TILE_OPS:
            quadrant = _quadrant_of(x, y, board_size)
            unlocked = _unlocked_quadrants(farm)
            if unlocked is not None and quadrant not in unlocked:
                self._record(
                    "locked_quadrant",
                    f"{name}: {op} at ({x},{y}) in locked quadrant {quadrant}",
                )

    def _check_plant_budget(self, obs: dict[str, Any], unit_actions: list[Any]) -> None:
        """The engine drops *all* PLANTs of a crop when demand exceeds seeds.

        This is atomic per crop, so one over-request wastes every unit that
        asked for that crop this turn — an expensive silent failure.
        """
        demand: dict[str, int] = {}
        for unit_action in unit_actions:
            if (
                isinstance(unit_action, list)
                and len(unit_action) >= 2
                and unit_action[0] == "PLANT"
            ):
                crop = unit_action[1]
                if isinstance(crop, str):
                    demand[crop] = demand.get(crop, 0) + 1

        if not demand:
            return

        seeds = _seeds(obs)
        if seeds is None:
            return

        for crop, wanted in demand.items():
            held = int(seeds.get(crop, 0) or 0)
            if wanted > held:
                self._record(
                    "plant_over_seeds",
                    f"{wanted}x PLANT {crop} with {held} seed(s) — "
                    f"engine drops all of them",
                )

    def _check_market(
        self, obs: dict[str, Any], action: dict[str, Any], farm: dict[str, Any] | None
    ) -> None:
        orders = action.get("market", [])
        if not isinstance(orders, list):
            self._record(
                "market_not_list",
                f"'market' is {type(orders).__name__}; engine treats it as empty",
            )
            return

        cap = _max_market_orders(obs)
        if len(orders) > cap:
            self._record(
                "market_orders_truncated",
                f"{len(orders)} orders exceeds maxMarketOrdersPerTurn={cap}; "
                f"the last {len(orders) - cap} are discarded",
            )

        for i, order in enumerate(orders[:cap]):
            self._check_order(i, order)

    def _check_order(self, index: int, order: Any) -> None:
        if not isinstance(order, list) or not order:
            self._record(
                "order_malformed", f"market[{index}]: expected a non-empty list"
            )
            return

        op = order[0]
        if not isinstance(op, str) or op not in MARKET_OPS:
            self._record("unknown_market_op", f"market[{index}]: {op!r}")
            return

        if op in MARKET_ATOMIC_OPS:
            return

        # BUY_*/SELL need [op, item, n]; a 2-element order is dropped outright.
        if len(order) < 3:
            self._record(
                "order_missing_quantity",
                f"market[{index}]: {op} needs [op, item, n], got {order!r}",
            )
            return

        item, raw_n = order[1], order[2]
        try:
            n = int(raw_n)
        except (TypeError, ValueError):
            self._record(
                "order_bad_quantity", f"market[{index}]: {op} quantity {raw_n!r}"
            )
            return
        if n <= 0:
            self._record(
                "order_bad_quantity", f"market[{index}]: {op} quantity {n} must be > 0"
            )
            return

        self._check_order_item(index, op, item)

    def _check_order_item(self, index: int, op: str, item: Any) -> None:
        valid: frozenset[str]
        if op == "BUY_SEED":
            valid = CROP_NAMES
        elif op == "BUY_ANIMAL":
            valid = ANIMAL_NAMES
        elif op == "BUY_PRODUCT":
            valid = BUYABLE_PRODUCTS
        else:
            valid = PRODUCT_NAMES

        if item not in valid:
            hint = (
                " (only WHEAT and FERTILIZER can be bought back)"
                if op == "BUY_PRODUCT"
                else ""
            )
            self._record("order_bad_item", f"market[{index}]: {op} {item!r}{hint}")


def as_callable(agent: Any) -> Callable[[dict[str, Any]], Any]:
    """Turn an agent spec into a callable, loading paths via the engine."""
    if callable(agent):
        return agent  # type: ignore[no-any-return]

    from kaggle_environments import make
    from kaggle_environments.agent import build_agent

    env = make("kaggriculture", debug=True)

    # build_agent returns a bare action-returning callable. Going through
    # `Agent` instead would nest a second timing/capture layer inside the one
    # env.run already applies, double-counting each turn's duration.
    built, _parallelizable = build_agent(agent, env.agents, env.name)

    # File-backed agents arrive as callable_agent(observation, configuration);
    # builtins and plain callables take the observation alone. Decide from the
    # signature rather than catching TypeError, which would also swallow a
    # genuine TypeError raised inside the agent.
    takes_config = _arity(built) >= 2
    configuration = env.configuration

    def act(obs: dict[str, Any]) -> Any:
        if takes_config:
            return built(obs, configuration)
        return built(obs)

    return act


def _arity(func: Any) -> int:
    code = getattr(func, "__code__", None)
    if code is not None:
        return int(getattr(code, "co_argcount", 1))
    return 1


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _own_farm(obs: dict[str, Any], player: int) -> dict[str, Any] | None:
    farms = _get(obs, "farms")
    if isinstance(farms, list) and player < len(farms):
        farm = farms[player]
        if isinstance(farm, dict):
            return farm
    return None


def _board_size(obs: dict[str, Any], farm: dict[str, Any] | None) -> int:
    tiles = _get(farm, "tiles") if farm is not None else None
    if isinstance(tiles, list) and tiles:
        return len(tiles)
    return DEFAULT_BOARD_SIZE


def _as_pos(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _hand_positions(farm: dict[str, Any] | None) -> list[tuple[int, int]]:
    hands = _get(farm, "hands") if farm is not None else None
    if not isinstance(hands, list):
        return []
    return [pos for pos in (_as_pos(h) for h in hands) if pos is not None]


def _unit_positions(farm: dict[str, Any] | None) -> list[tuple[int, int] | None]:
    farmer = _as_pos(_get(farm, "farmer")) if farm is not None else None
    return [farmer, *_hand_positions(farm)]


def _unlocked_quadrants(farm: dict[str, Any] | None) -> set[str] | None:
    raw = _get(farm, "unlocked_quadrants") if farm is not None else None
    if isinstance(raw, (list, tuple, set)):
        return {str(q) for q in raw}
    return None


def _seeds(obs: dict[str, Any]) -> dict[str, Any] | None:
    private = _get(obs, "private")
    seeds = _get(private, "seeds") if private is not None else None
    return seeds if isinstance(seeds, dict) else None


def _max_market_orders(obs: dict[str, Any]) -> int:
    raw = _get(obs, "maxMarketOrdersPerTurn")
    if isinstance(raw, int) and raw > 0:
        return raw
    return DEFAULT_MAX_MARKET_ORDERS
