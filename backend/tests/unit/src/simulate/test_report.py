"""Aggregation and exit-code mapping."""

from __future__ import annotations

from simulate.config import MatchSpec, SimConfig
from simulate.episode import EpisodeOutcome, TurnCost
from simulate.report import ExitCode, build_report, render
from simulate.validate import ValidationIssue


def _outcome(
    *,
    opponent: str = "starter",
    case_money: float = 4000.0,
    opponent_money: float = 3000.0,
    seed: int = 0,
    crashed: bool = False,
    timeouts: tuple[TurnCost, ...] = (),
    statuses: tuple[str, ...] = ("DONE", "DONE"),
    issues: tuple[ValidationIssue, ...] = (),
) -> EpisodeOutcome:
    return EpisodeOutcome(
        spec=MatchSpec(case="rulebase/case1", opponent=opponent, seed=seed, steps=720),
        case_money=case_money,
        opponent_money=opponent_money,
        statuses=statuses,
        steps=720,
        crashed=crashed,
        crash_repr="Traceback: boom" if crashed else None,
        timeouts=timeouts,
        issues=issues,
    )


def _config(**kwargs: object) -> SimConfig:
    defaults: dict[str, object] = {
        "case": "rulebase/case1",
        "opponents": ("starter",),
    }
    return SimConfig(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_win_loss_tie_counts() -> None:
    outcomes = [
        _outcome(case_money=4000, opponent_money=3000),
        _outcome(case_money=2000, opponent_money=3000),
        _outcome(case_money=3000, opponent_money=3000),
    ]

    report = build_report(_config(), outcomes, elapsed_sec=1.0)
    row = report.by_opponent[0]

    assert (row.wins, row.losses, row.ties) == (1, 1, 1)
    assert row.win_rate == 1 / 3


def test_margins_are_case_minus_opponent() -> None:
    outcomes = [
        _outcome(case_money=4000, opponent_money=3000),
        _outcome(case_money=1000, opponent_money=3000),
    ]

    row = build_report(_config(), outcomes, elapsed_sec=1.0).by_opponent[0]

    assert row.mean_margin == -500.0
    assert row.worst_margin == -2000.0
    assert row.mean_money == 2500.0


def test_opponents_are_grouped_separately() -> None:
    outcomes = [
        _outcome(opponent="starter"),
        _outcome(opponent="random"),
        _outcome(opponent="random"),
    ]

    report = build_report(
        _config(opponents=("starter", "random")), outcomes, elapsed_sec=1.0
    )

    assert [r.opponent for r in report.by_opponent] == ["starter", "random"]
    assert [r.episodes for r in report.by_opponent] == [1, 2]
    assert report.episodes == 3


def test_clean_run_exits_zero() -> None:
    report = build_report(_config(), [_outcome()], elapsed_sec=1.0)

    assert report.health.clean
    assert report.exit_code is ExitCode.OK


def test_crash_exits_one() -> None:
    report = build_report(_config(), [_outcome(crashed=True)], elapsed_sec=1.0)

    assert report.health.crashes == 1
    assert not report.health.clean
    assert report.exit_code is ExitCode.FAILED


def test_timeout_exits_one() -> None:
    timeouts = (TurnCost(step=5, player=0, duration=1.4),)

    report = build_report(_config(), [_outcome(timeouts=timeouts)], elapsed_sec=1.0)

    assert report.health.timeouts == 1
    assert report.exit_code is ExitCode.FAILED


def test_bad_status_exits_one() -> None:
    report = build_report(
        _config(), [_outcome(statuses=("ERROR", "DONE"))], elapsed_sec=1.0
    )

    assert report.health.bad_statuses == 1
    assert report.exit_code is ExitCode.FAILED


def test_invalid_actions_alone_do_not_fail_by_default() -> None:
    issues = (
        ValidationIssue(step=1, kind="market_orders_truncated", detail="11 > 10"),
    )

    report = build_report(_config(), [_outcome(issues=issues)], elapsed_sec=1.0)

    assert report.health.invalid_actions == 1
    assert report.health.clean
    assert report.exit_code is ExitCode.OK


def test_strict_mode_fails_on_invalid_actions() -> None:
    issues = (ValidationIssue(step=1, kind="unknown_unit_op", detail="HARVSET"),)

    report = build_report(
        _config(strict=True), [_outcome(issues=issues)], elapsed_sec=1.0
    )

    assert report.exit_code is ExitCode.INVALID_ACTIONS


def test_issue_kinds_are_counted() -> None:
    issues = (
        ValidationIssue(step=1, kind="unknown_unit_op", detail="a"),
        ValidationIssue(step=2, kind="unknown_unit_op", detail="b"),
        ValidationIssue(step=3, kind="locked_quadrant", detail="c"),
    )

    report = build_report(_config(), [_outcome(issues=issues)], elapsed_sec=1.0)

    assert dict(report.health.issue_kinds) == {
        "unknown_unit_op": 2,
        "locked_quadrant": 1,
    }


def test_render_always_shows_the_health_block() -> None:
    config = _config()
    report = build_report(config, [_outcome()], elapsed_sec=2.5)

    text = render(report, config)

    # The health block is unconditional by design: a broken agent must never be
    # reportable as a merely weak one.
    assert "health" in text
    assert "crashes" in text
    assert "timeouts" in text
    assert "invalid actions" in text


def test_render_includes_crash_traceback() -> None:
    config = _config()
    report = build_report(config, [_outcome(crashed=True)], elapsed_sec=1.0)

    text = render(report, config)

    assert "crash (first occurrence)" in text
    assert "boom" in text


def test_as_dict_is_json_serialisable() -> None:
    import json

    report = build_report(_config(), [_outcome()], elapsed_sec=1.0)

    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["case"] == "rulebase/case1"
    assert payload["episodes"] == 1
    assert payload["exit_code"] == 0
    assert payload["health"]["crashes"] == 0


def test_empty_run_is_clean() -> None:
    report = build_report(_config(), [], elapsed_sec=0.0)

    assert report.episodes == 0
    assert report.by_opponent == ()
    assert report.exit_code is ExitCode.OK
