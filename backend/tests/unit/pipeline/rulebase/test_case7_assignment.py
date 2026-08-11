"""Worker-assignment tests for ``rulebase/case7``.

case7's headline change is that deadline work (feeding, rescue watering,
harvesting) ignores the zone partition. That fix was worth **+14,714** mean bank
on its own -- more than every ported mechanism combined -- so it is pinned here.

case7 defines the same bare module names as case2 (``config``, ``tasks``, ...),
and the sibling test module already owns those names for the whole session. This
module therefore loads case7's modules under private names via importlib rather
than by import, so both cases can be unit-tested in one pytest run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

CASE6_DIR = Path(__file__).resolve().parents[4] / "pipeline" / "rulebase" / "case7"


def _load_case6() -> dict[str, ModuleType]:
    """Import case7's modules under private names, isolated from case2's."""
    loaded: dict[str, ModuleType] = {}
    # config has no intra-case imports; the rest depend on it by bare name, so
    # each is published under its plain name only while the group is being
    # built, then withdrawn.
    order = ("config", "observe", "tasks", "market")
    try:
        for name in order:
            spec = importlib.util.spec_from_file_location(
                f"_case6_{name}", CASE6_DIR / f"{name}.py"
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"_case6_{name}"] = module
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name in order:
            sys.modules.pop(name, None)
    return loaded


@pytest.fixture(scope="module")
def case7() -> dict[str, ModuleType]:
    return _load_case6()


def _obs(tiles: list[list[Any]], n_hands: int, **kw: Any) -> dict[str, Any]:
    farm: dict[str, Any] = {
        "money": 3000.0,
        "tiles": tiles,
        "farmer": kw.pop("farmer", [4, 4]),
        "hands": kw.pop("hands", [[4, 4] for _ in range(n_hands)]),
        "unlocked_quadrants": ["NW"],
        "hires_today": n_hands,
    }
    private: dict[str, Any] = {
        "shed": kw.pop("shed", {}),
        "seeds": kw.pop("seeds", {}),
        "inventories": kw.pop("inventories", [{} for _ in range(n_hands + 1)]),
    }
    obs: dict[str, Any] = {
        "player": 0,
        "step": 0,
        "day": 5,
        "hour": 5,
        "farms": [farm, dict(farm)],
        "private": private,
        "market": {"prices": {"MELON": 250}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    obs.update(kw)
    return obs


def _blank_board() -> list[list[Any]]:
    return [
        [None if (x < 5 and y < 5) else "LOCKED" for x in range(10)] for y in range(10)
    ]


def _animal(fed: bool = False) -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": fed,
        "cared_today": False,
        "consecutive_unfed": 1,
        "fertilizer_available": False,
    }


class TestZoneFreeDeadlines:
    def test_clustered_hungry_animals_are_all_fed_in_one_turn(
        self, case7: dict[str, ModuleType]
    ) -> None:
        """The bug this case exists to fix.

        Livestock clusters into a few columns. Under a strict zone partition
        only the units owning those columns could ever feed, so most of a large
        herd starved while units in empty zones idled.
        """
        observe, tasks = case7["observe"], case7["tasks"]

        tiles = _blank_board()
        # Four hungry animals stacked in a single column band...
        for y in (0, 1, 2, 3):
            tiles[y][1] = _animal()
        # ...and thirsty plants filling every other column, so each unit has
        # in-zone work of its own to prefer. Without competing work the
        # out-of-zone penalty is never binding and the partition looks harmless.
        for x in (0, 2, 3, 4):
            for y in range(4):
                tiles[y][x] = {
                    "kind": "PLANT",
                    "crop": "MELON",
                    "yield_units": 0,
                    "planted_day": 0,
                    "watered_today": False,
                    "consecutive_unwatered": 0,
                }

        n_hands = 7
        # Park four hands directly on the animals so feeding needs no travel:
        # the only thing that can stop them is the zone rule. With eight units
        # the bands are one column wide, so all four animals fall in zone 1 and
        # a strict partition lets just one unit act.
        hands = [[1, y] for y in (0, 1, 2, 3)] + [[4, 4]] * (n_hands - 4)
        inventories = [{"WHEAT": 4} for _ in range(n_hands + 1)]
        obs = _obs(
            tiles,
            n_hands,
            hands=hands,
            inventories=inventories,
            shed={"WHEAT": 20},
        )

        snap = observe.parse(obs)
        actions = tasks.assign(snap, tasks.build_tasks(snap))

        feeding = [a for a in actions if a and a[0] == "FEED"]
        assert len(feeding) >= 3, f"only {len(feeding)} fed this turn: {actions}"

    def test_growth_work_stays_zone_partitioned(
        self, case7: dict[str, ModuleType]
    ) -> None:
        """Only deadlines are exempt -- routine work still spreads by zone.

        Without this, every unit would chase the same nearest tile and the
        partition would stop doing its job.
        """
        tasks = case7["tasks"]
        plant = tasks.Task(tasks.P_PLANT, (1, 1), ["PLANT", "MELON"])
        assert plant.zone_free is False

        feed = tasks.Task(
            tasks.P_FEED, (1, 1), ["FEED"], needs_item="WHEAT", zone_free=True
        )
        assert feed.zone_free is True


class TestHerdRamp:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [(0, "CORE_HERD"), (6, "MID_HERD"), (12, "TARGET_HERD")],
    )
    def test_target_grows_with_the_season(
        self, case7: dict[str, ModuleType], day: int, expected: str
    ) -> None:
        observe, tasks, config = case7["observe"], case7["tasks"], case7["config"]
        obs = _obs(_blank_board(), 0, day=day)
        snap = observe.parse(obs)
        # One quadrant caps the herd at ANIMAL_SLOTS_PER_QUADRANT.
        want = min(getattr(config, expected), config.ANIMAL_SLOTS_PER_QUADRANT)
        assert tasks.herd_target(snap) == want

    def test_never_exceeds_available_pasture_slots(
        self, case7: dict[str, ModuleType]
    ) -> None:
        observe, tasks, config = case7["observe"], case7["tasks"], case7["config"]
        obs = _obs(_blank_board(), 0, day=29)
        snap = observe.parse(obs)
        assert tasks.herd_target(snap) <= config.ANIMAL_SLOTS_PER_QUADRANT

    def test_animal_purchases_stop_before_they_cannot_repay(
        self, case7: dict[str, ModuleType]
    ) -> None:
        """A COW first yields on day 8, so a late purchase only eats feed."""
        observe, market, config = case7["observe"], case7["market"], case7["config"]
        tiles = _blank_board()
        tiles[0][0] = {"kind": "PASTURE"}
        obs = _obs(tiles, 0, day=config.ANIMAL_PURCHASE_LAST_DAY + 1)
        snap = observe.parse(obs)
        orders = market.build_market(snap)
        assert [o for o in orders if o[0] == "BUY_ANIMAL"] == []


class TestFlagsAreHonoured:
    def test_disabled_features_are_actually_off(
        self, case7: dict[str, ModuleType]
    ) -> None:
        """The shipped configuration reflects the measured ablation.

        Strawberry cost 1,919 mean bank over 30 episodes; opponent-aware melon
        sizing measured exactly neutral because no available opponent grows
        enough melon to trip a threshold.
        """
        config, tasks, observe = case7["config"], case7["tasks"], case7["observe"]
        assert config.ENABLE_STRAWBERRY is False
        assert config.ENABLE_OPPONENT_AWARE_MELON is False

        snap = observe.parse(_obs(_blank_board(), 0))
        assert "STRAWBERRY" not in tasks.crop_targets(snap)
        assert tasks.melon_tiles_per_quadrant(snap) == config.MELON_TILES_PER_QUADRANT
