"""rulebase/case1 must clear the starter baseline without breaking."""

from __future__ import annotations

import pytest

from simulate.config import SimConfig
from simulate.matrix import run_matrix
from simulate.report import ExitCode, MatrixReport, build_report

CASE = "rulebase/case1"

#: Enough episodes to be meaningful without slowing the suite. A full season is
#: roughly 1.3s and these run in parallel.
EPISODES = 4


def _report(opponent: str) -> MatrixReport:
    config = SimConfig(
        case=CASE,
        opponents=(opponent,),
        episodes=EPISODES,
        workers=EPISODES,
        swap_sides=True,
    )
    return build_report(config, run_matrix(config), elapsed_sec=0.0)


@pytest.fixture(scope="module")
def vs_starter() -> MatrixReport:
    return _report("starter")


@pytest.fixture(scope="module")
def vs_pass() -> MatrixReport:
    return _report("pass")


def test_case1_beats_starter(vs_starter: MatrixReport) -> None:
    row = vs_starter.by_opponent[0]

    # starter is the bar a new case has to clear.
    assert row.win_rate > 0.5, f"win rate {row.win_rate:.0%} vs starter"
    assert row.mean_margin > 0


def test_case1_never_crashes_or_times_out(vs_starter: MatrixReport) -> None:
    health = vs_starter.health

    assert health.crashes == 0
    assert health.timeouts == 0
    assert health.bad_statuses == 0
    assert vs_starter.exit_code is ExitCode.OK


def test_case1_wastes_no_actions(vs_starter: MatrixReport) -> None:
    # Every action the agent emits must be one the engine actually applies — a
    # silent no-op is a wasted turn that would only show up as a lower score.
    health = vs_starter.health

    assert health.invalid_actions == 0, f"discarded actions: {health.issue_kinds}"


def test_case1_stays_well_inside_the_turn_budget(vs_starter: MatrixReport) -> None:
    # The Kaggle harness allows 1s per turn. The margin here is deliberately
    # loose: these durations are wall-clock, and the suite runs episodes across
    # parallel pytest workers, so a tight bound would measure machine load
    # rather than the agent. `timeouts` is the assertion that actually matters.
    health = vs_starter.health

    assert health.timeouts == 0
    assert health.max_duration < health.act_timeout / 2


def test_case1_liquidates_its_inventory(vs_pass: MatrixReport) -> None:
    # Unsold stock scores zero, so a season that ends holding wheat has left
    # money on the table. Money well above the 3000 start implies it sold.
    assert vs_pass.by_opponent[0].mean_money > 5000
