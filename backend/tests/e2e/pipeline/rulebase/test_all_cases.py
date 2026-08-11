"""Every rulebase case must clear the starter baseline without breaking.

`test_case1.py` covers case1 in depth. This module applies the same health bar
to the whole family, so a new case cannot land silently broken.

The invalid-action assertion is the one that matters most here: the engine
discards a malformed op, an out-of-range target, or an over-cap market order
**without raising**, so such bugs surface only as a lower score. `run_matrix`
with `validate_actions` turns them into a hard failure instead.
"""

from __future__ import annotations

import pytest

from simulate.config import SimConfig
from simulate.matrix import run_matrix
from simulate.report import ExitCode, MatrixReport, build_report

#: Cases authored in this repo plus the four notebook ports. case1 has its own
#: dedicated module; it is included here so the family-wide bar covers it too.
CASES = (
    "rulebase/case1",
    "rulebase/case2",
    "rulebase/case3",
    "rulebase/case4",
    "rulebase/case5",
    "rulebase/case6",
    "rulebase/case7",
)

#: Enough episodes to be meaningful without slowing the suite.
EPISODES = 2


def _report(case: str, opponent: str) -> MatrixReport:
    config = SimConfig(
        case=case,
        opponents=(opponent,),
        episodes=EPISODES,
        workers=EPISODES,
        swap_sides=True,
    )
    return build_report(config, run_matrix(config), elapsed_sec=0.0)


@pytest.fixture(scope="module")
def reports() -> dict[str, MatrixReport]:
    return {case: _report(case, "starter") for case in CASES}


@pytest.mark.parametrize("case", CASES)
def test_case_beats_starter(case: str, reports: dict[str, MatrixReport]) -> None:
    """`starter` is the bar a case has to clear to be worth keeping."""
    row = reports[case].by_opponent[0]

    assert row.win_rate > 0.5, f"{case}: win rate {row.win_rate:.0%} vs starter"
    assert row.mean_margin > 0, f"{case}: mean margin {row.mean_margin}"


@pytest.mark.parametrize("case", CASES)
def test_case_never_crashes_or_times_out(
    case: str, reports: dict[str, MatrixReport]
) -> None:
    """An uncaught exception forfeits the episode, so this is non-negotiable."""
    health = reports[case].health

    assert health.crashes == 0, f"{case}: {health.crashes} crashes"
    assert health.timeouts == 0, f"{case}: {health.timeouts} timeouts"
    assert health.bad_statuses == 0
    assert reports[case].exit_code is ExitCode.OK


@pytest.mark.parametrize("case", CASES)
def test_case_wastes_no_actions(case: str, reports: dict[str, MatrixReport]) -> None:
    """Discarded actions are silent — only a test can catch them."""
    health = reports[case].health

    assert health.invalid_actions == 0, f"{case}: {health.issue_kinds}"
