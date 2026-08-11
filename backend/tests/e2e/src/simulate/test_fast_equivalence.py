"""The fast engine must agree with the official one, exactly.

This is the load-bearing test for :mod:`simulate.fast`. Stripping the framework
is only safe while the results stay identical, and a divergence produces no
exception — just a different number.

The failure mode it actually catches, verified by mutation (deleting the line
makes 15 of these fail): forgetting to advance ``observation.step``, which
freezes day/hour and disables every end-of-day rule while the episode still
runs to completion and reports a plausible score.

An *extra* interpreter call past the end, by contrast, is provably harmless —
``interpreter`` early-returns once ``env.done`` — so no assertion here pretends
to guard the loop's upper bound. ``test_episode_lengths_match_official_engine``
covers short seasons because the day-boundary arithmetic differs there, not
because it detects an off-by-one.

Run this after upgrading ``kaggle-environments``: it is the detector for the
framework contract shifting under us.
"""

from __future__ import annotations

import pytest

from simulate.config import MatchSpec
from simulate.episode import run_episode
from simulate.fast import run_fast_episode

#: Deterministic agents only. The builtin `random` seeds its own RNG with no
#: argument, so it cannot be compared across two separate runs.
PAIRS = [
    ("starter", "pass"),
    ("pass", "starter"),
    ("rulebase/case1", "starter"),
    ("rulebase/case1", "pass"),
    ("starter", "starter"),
]

SEEDS = [1, 2, 7, 42, 99]


def _spec(case: str, opponent: str, seed: int, steps: int) -> MatchSpec:
    return MatchSpec(case=case, opponent=opponent, seed=seed, steps=steps)


@pytest.mark.parametrize(("case", "opponent"), PAIRS)
def test_full_season_matches_official_engine(case: str, opponent: str) -> None:
    spec = _spec(case, opponent, seed=1, steps=720)

    official = run_episode(spec, validate_actions=False)
    fast = run_fast_episode(spec, validate_actions=False)

    assert fast.case_money == official.case_money
    assert fast.opponent_money == official.opponent_money
    assert fast.steps == official.steps


@pytest.mark.parametrize("seed", SEEDS)
def test_seeds_match_official_engine(seed: int) -> None:
    # The seed drives weed spawns and shop unlocks, so agreement across seeds
    # exercises the end-of-day paths that a frozen clock would skip entirely.
    spec = _spec("starter", "pass", seed=seed, steps=720)

    official = run_episode(spec, validate_actions=False)
    fast = run_fast_episode(spec, validate_actions=False)

    assert fast.case_money == official.case_money
    assert fast.opponent_money == official.opponent_money


@pytest.mark.parametrize("steps", [24, 48, 121, 240, 720])
def test_episode_lengths_match_official_engine(steps: int) -> None:
    # Short and non-multiple-of-24 seasons exercise different day-boundary
    # arithmetic than a full season, including one (121) that ends mid-day.
    spec = _spec("starter", "pass", seed=3, steps=steps)

    official = run_episode(spec, validate_actions=False)
    fast = run_fast_episode(spec, validate_actions=False)

    assert fast.case_money == official.case_money
    assert fast.steps == official.steps == steps


def test_swapped_seats_match_official_engine() -> None:
    spec = MatchSpec(
        case="rulebase/case1",
        opponent="starter",
        seed=5,
        steps=720,
        swap_sides=True,
    )

    official = run_episode(spec, validate_actions=False)
    fast = run_fast_episode(spec, validate_actions=False)

    assert fast.case_money == official.case_money
    assert fast.opponent_money == official.opponent_money


def test_clock_actually_advances() -> None:
    # A frozen clock is the specific failure this module is most prone to. With
    # day/hour stuck at 0 nothing is ever harvested, so a real strategy would
    # end near its starting money instead of well above it.
    spec = _spec("rulebase/case1", "pass", seed=1, steps=720)

    fast = run_fast_episode(spec, validate_actions=False)

    assert fast.case_money > 5000, "no production happened — is the clock frozen?"
