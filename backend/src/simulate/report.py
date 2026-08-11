"""Aggregate outcomes and render them.

The report always shows a ``health`` block. The whole point of this module is
that a broken agent must not be reportable as a merely weak one, so the
crash / timeout / invalid-action counts sit next to the win rate rather than
behind a verbose flag.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .config import Engine, SimConfig
from .episode import EpisodeOutcome


class ExitCode(IntEnum):
    """Process exit status, so CI can gate on more than a win rate."""

    OK = 0
    FAILED = 1
    INVALID_ACTIONS = 2


@dataclass(frozen=True)
class OpponentReport:
    """Aggregate over every episode against one opponent."""

    opponent: str
    outcomes: tuple[EpisodeOutcome, ...]

    @property
    def episodes(self) -> int:
        return len(self.outcomes)

    @property
    def wins(self) -> int:
        return sum(1 for o in self.outcomes if o.winner_is_case)

    @property
    def ties(self) -> int:
        return sum(1 for o in self.outcomes if o.is_tie)

    @property
    def losses(self) -> int:
        return self.episodes - self.wins - self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.outcomes else 0.0

    @property
    def mean_money(self) -> float:
        if not self.outcomes:
            return 0.0
        return statistics.fmean(o.case_money for o in self.outcomes)

    @property
    def mean_margin(self) -> float:
        if not self.outcomes:
            return 0.0
        return statistics.fmean(o.margin for o in self.outcomes)

    @property
    def worst_margin(self) -> float:
        if not self.outcomes:
            return 0.0
        return min(o.margin for o in self.outcomes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": round(self.win_rate, 4),
            "mean_money": round(self.mean_money, 2),
            "mean_margin": round(self.mean_margin, 2),
            "worst_margin": round(self.worst_margin, 2),
        }


@dataclass(frozen=True)
class HealthReport:
    """Everything that says "this agent is broken", not "this agent is losing"."""

    crashes: int
    timeouts: int
    slow_turns: int
    bad_statuses: int
    invalid_actions: int
    max_duration: float
    act_timeout: float
    crash_samples: tuple[str, ...] = ()
    issue_kinds: tuple[tuple[str, int], ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.crashes or self.timeouts or self.bad_statuses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crashes": self.crashes,
            "timeouts": self.timeouts,
            "slow_turns": self.slow_turns,
            "bad_statuses": self.bad_statuses,
            "invalid_actions": self.invalid_actions,
            "max_turn_duration": round(self.max_duration, 4),
            "act_timeout": self.act_timeout,
            "issue_kinds": dict(self.issue_kinds),
        }


@dataclass(frozen=True)
class MatrixReport:
    """The full result of one ``run_matrix`` call."""

    case: str
    steps: int
    elapsed_sec: float
    by_opponent: tuple[OpponentReport, ...]
    health: HealthReport
    strict: bool = False
    engine: Engine = Engine.OFFICIAL

    @property
    def episodes(self) -> int:
        return sum(r.episodes for r in self.by_opponent)

    @property
    def exit_code(self) -> ExitCode:
        if not self.health.clean:
            return ExitCode.FAILED
        if self.strict and self.health.invalid_actions:
            return ExitCode.INVALID_ACTIONS
        return ExitCode.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "steps": self.steps,
            "episodes": self.episodes,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "engine": self.engine.value,
            "opponents": [r.as_dict() for r in self.by_opponent],
            "health": self.health.as_dict(),
            "exit_code": int(self.exit_code),
        }


def build_health(
    outcomes: list[EpisodeOutcome], *, act_timeout: float, max_samples: int = 3
) -> HealthReport:
    kinds: Counter[str] = Counter()
    for outcome in outcomes:
        for issue in outcome.issues:
            kinds[issue.kind] += 1

    samples = [o.crash_repr for o in outcomes if o.crash_repr][:max_samples]

    return HealthReport(
        crashes=sum(1 for o in outcomes if o.crashed),
        timeouts=sum(len(o.timeouts) for o in outcomes),
        slow_turns=sum(len(o.slow_turns) for o in outcomes),
        bad_statuses=sum(1 for o in outcomes if o.bad_status),
        invalid_actions=sum(o.invalid_actions for o in outcomes),
        max_duration=max((o.max_duration for o in outcomes), default=0.0),
        act_timeout=act_timeout,
        crash_samples=tuple(s for s in samples if s),
        issue_kinds=tuple(kinds.most_common()),
    )


def build_report(
    config: SimConfig, outcomes: list[EpisodeOutcome], *, elapsed_sec: float
) -> MatrixReport:
    """Group outcomes by opponent, preserving the configured opponent order."""
    grouped: dict[str, list[EpisodeOutcome]] = {name: [] for name in config.opponents}
    for outcome in outcomes:
        grouped.setdefault(outcome.spec.opponent, []).append(outcome)

    by_opponent = tuple(
        OpponentReport(opponent=name, outcomes=tuple(items))
        for name, items in grouped.items()
        if items
    )

    return MatrixReport(
        case=config.case,
        steps=config.steps,
        elapsed_sec=elapsed_sec,
        by_opponent=by_opponent,
        health=build_health(outcomes, act_timeout=config.act_timeout),
        strict=config.strict,
        engine=config.engine,
    )


def _header(report: MatrixReport, config: SimConfig) -> str:
    bits = [f"{report.episodes} ep x {report.steps} steps"]
    if config.swap_sides:
        bits.append("swap-sides")
    bits.append(f"{config.workers} worker{'s' if config.workers != 1 else ''}")
    # Name the engine whenever it is not the competition path, so a number from
    # a fast run is never mistaken for a verified one.
    if config.engine is not Engine.OFFICIAL:
        bits.append(f"engine={config.engine.value}")
    bits.append(f"{report.elapsed_sec:.1f}s")
    return f"{report.case}   ({', '.join(bits)})"


def render(report: MatrixReport, config: SimConfig) -> str:
    """Render the human-readable report, health block included."""
    lines = [_header(report, config), ""]

    lines.append(
        f"{'opponent':<16}{'W/L/T':<14}{'win%':>7}"
        f"{'mean money':>14}{'mean margin':>14}{'worst':>12}"
    )
    for row in report.by_opponent:
        wlt = f"{row.wins}/{row.losses}/{row.ties}"
        lines.append(
            f"{row.opponent:<16}{wlt:<14}{row.win_rate:>6.1%}"
            f"{row.mean_money:>14,.0f}{row.mean_margin:>+14,.0f}"
            f"{row.worst_margin:>+12,.0f}"
        )

    health = report.health
    lines += [
        "",
        "health",
        f"  crashes         : {health.crashes}",
        f"  timeouts        : {health.timeouts}"
        f"        (max turn {health.max_duration:.3f}s"
        f" / limit {health.act_timeout:.3f}s)",
        f"  slow turns      : {health.slow_turns}",
        f"  status != DONE  : {health.bad_statuses}",
        f"  invalid actions : {health.invalid_actions}",
    ]

    for kind, count in health.issue_kinds[:8]:
        lines.append(f"      {kind:<28}{count}")

    for sample in health.crash_samples:
        lines += ["", "crash (first occurrence)", *_indent(sample)]

    return "\n".join(lines)


def _indent(text: str, prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" for line in text.rstrip().splitlines()]
