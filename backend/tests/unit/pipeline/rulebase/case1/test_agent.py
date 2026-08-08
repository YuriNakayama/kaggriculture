"""Unit tests for the rulebase/case1 wheat-loop agent.

Invalid actions are silent no-ops in this engine, so a malformed action never
raises — it just wastes the turn. These tests are the only thing standing
between a typo and a season of PASSes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_CASE_DIR = Path(__file__).resolve().parents[5] / "pipeline" / "rulebase" / "case1"


def _load_agent_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rulebase_case1_main", _CASE_DIR / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent_module = _load_agent_module()
agent = agent_module.agent


def make_obs(
    *,
    tile: Any = None,
    seeds: dict[str, int] | None = None,
    shed: dict[str, int] | None = None,
    money: float = 3000.0,
    day: int = 0,
) -> dict[str, Any]:
    """Minimal observation with the farmer at its start tile (4, 4)."""
    tiles: list[list[Any]] = [[None] * 10 for _ in range(10)]
    tiles[4][4] = tile
    return {
        "player": 0,
        "day": day,
        "hour": 0,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {},
        ],
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
        "town": {"unlocked_shops": []},
        "private": {
            "shed": shed or {},
            "seeds": seeds or {},
            "inventories": [{}],
        },
    }


def test_returns_all_required_keys() -> None:
    action = agent(make_obs())
    assert set(action) == {"farmer", "hands", "market"}


def test_buys_seed_when_none_held() -> None:
    action = agent(make_obs(seeds={}))
    assert ["BUY_SEED", "WHEAT", 1] in action["market"]


def test_does_not_buy_seed_when_already_held() -> None:
    action = agent(make_obs(seeds={"WHEAT": 1}))
    assert not any(o[0] == "BUY_SEED" for o in action["market"])


def test_does_not_buy_seed_when_broke() -> None:
    action = agent(make_obs(seeds={}, money=0.0))
    assert not any(o[0] == "BUY_SEED" for o in action["market"])


def test_plants_on_empty_tile_with_seed() -> None:
    action = agent(make_obs(tile=None, seeds={"WHEAT": 1}))
    assert action["farmer"] == ["PLANT", "WHEAT"]


def test_passes_on_empty_tile_without_seed() -> None:
    action = agent(make_obs(tile=None, seeds={}))
    assert action["farmer"] == ["PASS"]


def test_waters_unwatered_young_plant() -> None:
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": False}
    action = agent(make_obs(tile=tile, day=1))
    assert action["farmer"] == ["WATER"]


def test_does_not_rewater_same_day() -> None:
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": True}
    action = agent(make_obs(tile=tile, day=1))
    assert action["farmer"] == ["PASS"]


def test_harvests_at_target_age() -> None:
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": True}
    action = agent(make_obs(tile=tile, day=agent_module.WHEAT_HARVEST_AGE))
    assert action["farmer"] == ["HARVEST"]


def test_digs_weed() -> None:
    action = agent(make_obs(tile={"kind": "WEED"}))
    assert action["farmer"] == ["DIG"]


def test_sells_wheat_in_shed() -> None:
    action = agent(make_obs(shed={"WHEAT": 4}))
    assert ["SELL", "WHEAT", 4] in action["market"]


def test_sell_is_ordered_before_buy() -> None:
    """Orders past the cap are dropped from the tail, so sales must come first."""
    action = agent(make_obs(shed={"WHEAT": 4}, seeds={}))
    ops = [o[0] for o in action["market"]]
    assert ops.index("SELL") < ops.index("BUY_SEED")


def test_never_exceeds_market_order_cap() -> None:
    action = agent(make_obs(shed={"WHEAT": 99}, seeds={}))
    assert len(action["market"]) <= agent_module.MAX_MARKET_ORDERS


def test_never_acts_on_locked_tile() -> None:
    action = agent(make_obs(tile="LOCKED"))
    assert action["farmer"] == ["PASS"]


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"player": 0},
        {"player": 0, "farms": []},
        {"player": 5, "farms": [{}]},
        None,
    ],
)
def test_malformed_observation_returns_safe_action(broken: Any) -> None:
    """An uncaught exception forfeits the episode — never raise."""
    action = agent(broken)
    assert action == {"farmer": ["PASS"], "hands": [], "market": []}
