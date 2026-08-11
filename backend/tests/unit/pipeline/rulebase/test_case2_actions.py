"""Action-construction tests for ``rulebase/case2``.

The engine treats every invalid action as a silent no-op: a typo'd op, an
out-of-range target, or an 11th market order is discarded without any signal.
Bugs therefore surface as a low score, never as a crash, which makes these
assertions the only mechanism that can detect them.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

# Ops the engine accepts, transcribed from `_apply_unit_action` and
# `_parse_order` in kaggle_environments/envs/kaggriculture/kaggriculture.py.
UNIT_OPS = {
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "PICKUP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DROP",
}
MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}

BOARD = 10


def _empty_tiles() -> list[list[Any]]:
    """A fresh board: NW unlocked, everything else LOCKED."""
    return [
        [None if (x < 5 and y < 5) else "LOCKED" for x in range(BOARD)]
        for y in range(BOARD)
    ]


def _obs(**overrides: Any) -> dict[str, Any]:
    """A well-formed turn-0 observation, overridable per test."""
    farm: dict[str, Any] = {
        "money": 3000.0,
        "tiles": _empty_tiles(),
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    farm.update(overrides.pop("farm", {}))
    private: dict[str, Any] = {
        "shed": {},
        "seeds": {},
        "inventories": [{}],
    }
    private.update(overrides.pop("private", {}))
    obs: dict[str, Any] = {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [farm, dict(farm)],
        "private": private,
        "market": {
            "prices": {"WHEAT": 25, "MELON": 250, "MILK": 160},
            "inventory": {"WHEAT": 10000, "MELON": 10000, "MILK": 10000},
        },
        "town": {"unlocked_shops": []},
    }
    obs.update(overrides)
    return obs


def _assert_well_formed(action: dict[str, Any], *, n_hands: int) -> None:
    """Every structural invariant the engine silently depends on."""
    assert set(action) == {"farmer", "hands", "market"}

    assert isinstance(action["farmer"], list) and action["farmer"]
    assert action["farmer"][0] in UNIT_OPS

    # Exactly one entry per hired hand, in order: a short list silently drops
    # actions, a long one is ignored past the end.
    assert len(action["hands"]) == n_hands
    for hand in action["hands"]:
        assert isinstance(hand, list) and hand
        assert hand[0] in UNIT_OPS

    # Orders past maxMarketOrdersPerTurn are dropped from the tail.
    assert len(action["market"]) <= 10
    for order in action["market"]:
        assert isinstance(order, list) and order
        assert order[0] in MARKET_OPS
        if order[0] in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
            # `_parse_order` discards the order unless it has 3 fields and a
            # positive integer quantity.
            assert len(order) == 3
            assert isinstance(order[2], int)
            assert order[2] > 0


class TestActionShape:
    def test_opening_turn_is_well_formed(self, case2: ModuleType) -> None:
        assert case2.agent(_obs()) is not None
        _assert_well_formed(case2.agent(_obs()), n_hands=0)

    @pytest.mark.parametrize("n_hands", [0, 1, 3, 6, 12])
    def test_one_entry_per_hired_hand(self, case2: ModuleType, n_hands: int) -> None:
        obs = _obs(
            farm={"hands": [[4, 4] for _ in range(n_hands)], "hires_today": n_hands},
            private={
                "shed": {},
                "seeds": {},
                "inventories": [{} for _ in range(n_hands + 1)],
            },
        )
        _assert_well_formed(case2.agent(obs), n_hands=n_hands)

    def test_market_orders_respect_the_cap(self, case2: ModuleType) -> None:
        # A full shed offers far more sellable lines than the 10-order budget.
        shed = {
            "WHEAT": 40,
            "CARROT": 40,
            "TOMATO": 40,
            "STRAWBERRY": 40,
            "MELON": 40,
            "EGG": 40,
            "MILK": 40,
            "WOOL": 40,
            "FERTILIZER": 40,
        }
        obs = _obs(private={"shed": shed, "seeds": {}, "inventories": [{}]})
        action = case2.agent(obs)
        assert len(action["market"]) <= 10
        _assert_well_formed(action, n_hands=0)

    def test_sell_never_exceeds_shed_stock(self, case2: ModuleType) -> None:
        # Selling more than the shed holds aborts the order mid-way in
        # `_commit_unit`, wasting one of the ten slots.
        obs = _obs(private={"shed": {"MELON": 3}, "seeds": {}, "inventories": [{}]})
        sells = [o for o in case2.agent(obs)["market"] if o[0] == "SELL"]
        for order in sells:
            assert order[2] <= 3


class TestRobustness:
    """The agent must never raise: an exception forfeits the episode."""

    @pytest.mark.parametrize(
        "obs",
        [
            pytest.param({}, id="empty"),
            pytest.param({"player": 0}, id="player-only"),
            pytest.param({"farms": [], "player": 0}, id="no-farms"),
            pytest.param({"farms": [{}], "player": 0, "private": {}}, id="empty-farm"),
            pytest.param(
                {"farms": [{}], "player": 5, "private": {}}, id="player-out-of-range"
            ),
            pytest.param(
                {
                    "farms": [{"tiles": "nonsense", "money": None}],
                    "player": 0,
                    "private": {},
                },
                id="malformed-tiles",
            ),
            pytest.param(
                {
                    "farms": [{"tiles": [], "hands": "nope"}],
                    "player": 0,
                    "private": None,
                },
                id="malformed-hands",
            ),
        ],
    )
    def test_malformed_observation_yields_legal_action(
        self, case2: ModuleType, obs: dict[str, Any]
    ) -> None:
        action = case2.agent(obs)
        assert set(action) == {"farmer", "hands", "market"}
        assert action["farmer"][0] in UNIT_OPS

    def test_missing_keys_do_not_raise(self, case2: ModuleType) -> None:
        obs = _obs()
        for key in list(obs):
            partial = {k: v for k, v in obs.items() if k != key}
            action = case2.agent(partial)
            assert action["farmer"][0] in UNIT_OPS


class TestLayerInvariants:
    """Assert the guards where they are enforced, not only through ``agent``.

    ``main._decide`` re-caps the market list and pads ``hands``, so a bug in the
    layer underneath is invisible from the outside. These tests pin the
    behaviour of that layer directly.
    """

    def test_assign_returns_one_action_per_unit(self) -> None:
        import observe
        import tasks

        obs = _obs(
            farm={"hands": [[1, 1], [2, 2], [3, 3]], "hires_today": 3},
            private={"shed": {}, "seeds": {"MELON": 4}, "inventories": [{}] * 4},
        )
        snap = observe.parse(obs)
        actions = tasks.assign(snap, tasks.build_tasks(snap))
        # farmer + 3 hands
        assert len(actions) == 4
        for action in actions:
            assert isinstance(action, list) and action
            assert action[0] in UNIT_OPS

    def test_build_market_caps_orders_at_the_engine_limit(self) -> None:
        import market
        import observe

        shed = {
            "WHEAT": 60,
            "CARROT": 60,
            "TOMATO": 60,
            "STRAWBERRY": 60,
            "MELON": 60,
            "EGG": 60,
            "MILK": 60,
            "WOOL": 60,
            "FERTILIZER": 60,
        }
        obs = _obs(
            hour=0,
            private={"shed": shed, "seeds": {}, "inventories": [{}]},
        )
        orders = market.build_market(observe.parse(obs))
        assert len(orders) <= 10

    def test_price_model_matches_the_engine(self) -> None:
        # The sell ranking is only as good as this curve; drift here silently
        # mis-ranks every order.
        import market
        from kaggle_environments.envs.kaggriculture import kaggriculture as engine

        for item in ("WHEAT", "MELON", "MILK", "WOOL", "STRAWBERRY"):
            for inventory in (9_000, 9_990, 10_000, 10_050, 10_400, 12_000):
                assert market.price_at(item, inventory) == engine.market_price(
                    item, inventory
                ), f"{item} @ {inventory}"


class TestPolicyBehaviour:
    def test_hires_at_the_start_of_each_day(self, case2: ModuleType) -> None:
        # Hands vanish at end of day and `hires_today` resets, so hiring has to
        # happen again every morning.
        action = case2.agent(_obs(hour=0, day=3, step=72))
        assert sum(1 for o in action["market"] if o[0] == "HIRE") > 0

    def test_does_not_rehire_mid_day(self, case2: ModuleType) -> None:
        obs = _obs(
            hour=5,
            farm={"hands": [[4, 4]] * 6, "hires_today": 6},
            private={"shed": {}, "seeds": {}, "inventories": [{}] * 7},
        )
        assert [o for o in case2.agent(obs)["market"] if o[0] == "HIRE"] == []

    def test_liquidates_before_the_final_executed_step(self, case2: ModuleType) -> None:
        # The interpreter marks DONE after step 718, so a sale planned for 719
        # never executes and unsold stock scores zero.
        obs = _obs(
            step=710,
            day=29,
            hour=14,
            private={"shed": {"MELON": 30}, "seeds": {}, "inventories": [{}]},
        )
        sells = [o for o in case2.agent(obs)["market"] if o[0] == "SELL"]
        assert sells, "expected liquidation to sell remaining stock"
        assert sum(o[2] for o in sells if o[1] == "MELON") == 30

    def test_plants_when_seed_and_empty_land_are_available(
        self, case2: ModuleType
    ) -> None:
        obs = _obs(private={"shed": {}, "seeds": {"MELON": 5}, "inventories": [{}]})
        action = case2.agent(obs)
        ops = [action["farmer"][0], *(h[0] for h in action["hands"])]
        # The farmer starts on a shed tile, so the first move is toward land.
        assert set(ops) <= UNIT_OPS

    def test_never_plants_a_crop_it_has_no_seed_for(self, case2: ModuleType) -> None:
        # The interpreter drops ALL plant requests for a crop when the turn's
        # requests exceed the seed count, so over-requesting wastes every unit.
        obs = _obs(
            farm={"hands": [[2, 2]] * 6, "hires_today": 6},
            private={"shed": {}, "seeds": {"MELON": 2}, "inventories": [{}] * 7},
        )
        action = case2.agent(obs)
        units = [action["farmer"], *action["hands"]]
        planted = [u for u in units if u[0] == "PLANT" and u[1] == "MELON"]
        assert len(planted) <= 2
