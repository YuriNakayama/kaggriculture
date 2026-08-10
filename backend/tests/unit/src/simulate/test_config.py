"""Matrix expansion and seat handling."""

from __future__ import annotations

from simulate.config import ACT_TIMEOUT_SEC, MatchSpec, SimConfig


def test_agent_order_puts_case_first_by_default() -> None:
    spec = MatchSpec(case="rulebase/case1", opponent="starter", seed=0)

    assert spec.agent_order() == ("rulebase/case1", "starter")
    assert spec.case_player == 0
    assert spec.opponent_player == 1


def test_swap_sides_moves_case_to_player_one() -> None:
    spec = MatchSpec(case="rulebase/case1", opponent="starter", seed=0, swap_sides=True)

    assert spec.agent_order() == ("starter", "rulebase/case1")
    assert spec.case_player == 1
    assert spec.opponent_player == 0


def test_iter_specs_expands_opponents_and_episodes() -> None:
    config = SimConfig(
        case="rulebase/case1", opponents=("starter", "random"), episodes=3
    )

    specs = config.specs()

    assert len(specs) == 6
    assert [s.seed for s in specs if s.opponent == "starter"] == [0, 1, 2]
    assert {s.opponent for s in specs} == {"starter", "random"}


def test_seeds_start_from_seed_base() -> None:
    config = SimConfig(case="starter", opponents=("random",), episodes=2, seed_base=100)

    assert [s.seed for s in config.specs()] == [100, 101]


def test_swap_sides_plays_each_seed_from_both_seats() -> None:
    config = SimConfig(
        case="starter", opponents=("random",), episodes=2, swap_sides=True
    )

    specs = config.specs()

    assert len(specs) == 4
    # Each seed appears once per seat, so neither player benefits from acting
    # first into the shared market.
    assert sorted((s.seed, s.swap_sides) for s in specs) == [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]


def test_slow_turn_threshold_is_half_the_act_timeout() -> None:
    config = SimConfig(case="starter", opponents=("random",))

    assert config.act_timeout == ACT_TIMEOUT_SEC
    assert config.slow_turn_threshold == ACT_TIMEOUT_SEC / 2
