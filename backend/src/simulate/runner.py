"""Backwards-compatible façade over the evaluation modules.

The original ``run_episode`` / ``run_match`` API predates the matrix runner and
is kept so existing callers keep working. New code should prefer
:mod:`simulate.episode` and :mod:`simulate.matrix`, which expose crash /
timeout / invalid-action detail that this shape cannot represent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import BUILTIN_AGENTS
from .config import FULL_SEASON_STEPS, MatchSpec
from .episode import EpisodeOutcome
from .episode import run_episode as _run_episode

__all__ = [
    "BUILTIN_AGENTS",
    "FULL_SEASON_STEPS",
    "EpisodeResult",
    "MatchSummary",
    "run_episode",
    "run_match",
]


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single episode, in player-index order."""

    rewards: list[float]
    statuses: list[str]
    steps: int
    crashed: bool = False
    timeouts: int = 0

    @property
    def winner(self) -> int | None:
        """Index of the winning player, or ``None`` on a tie."""
        if self.rewards[0] == self.rewards[1]:
            return None
        return 0 if self.rewards[0] > self.rewards[1] else 1

    @property
    def margin(self) -> float:
        """Player 0's final money minus player 1's."""
        return self.rewards[0] - self.rewards[1]

    @property
    def ok(self) -> bool:
        """True when neither agent crashed, timed out, or errored.

        ``status`` alone is not sufficient: the engine reports ``DONE`` for both
        crashes and timeouts, so those are tracked separately.
        """
        return (
            not self.crashed
            and self.timeouts == 0
            and all(s == "DONE" for s in self.statuses)
        )


@dataclass
class MatchSummary:
    """Aggregate over several episodes, from player 0's perspective."""

    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def wins(self) -> int:
        return sum(1 for e in self.episodes if e.winner == 0)

    @property
    def losses(self) -> int:
        return sum(1 for e in self.episodes if e.winner == 1)

    @property
    def ties(self) -> int:
        return sum(1 for e in self.episodes if e.winner is None)

    @property
    def errors(self) -> int:
        return sum(1 for e in self.episodes if not e.ok)

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.episodes) if self.episodes else 0.0

    @property
    def mean_money(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.fmean(e.rewards[0] for e in self.episodes)

    @property
    def mean_margin(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.fmean(e.margin for e in self.episodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "errors": self.errors,
            "win_rate": round(self.win_rate, 4),
            "mean_money": round(self.mean_money, 2),
            "mean_margin": round(self.mean_margin, 2),
        }


def _to_result(outcome: EpisodeOutcome) -> EpisodeResult:
    """Re-index a case-relative outcome back to player order."""
    rewards = [0.0, 0.0]
    rewards[outcome.spec.case_player] = outcome.case_money
    rewards[outcome.spec.opponent_player] = outcome.opponent_money
    statuses = list(outcome.statuses) or ["DONE", "DONE"]
    return EpisodeResult(
        rewards=rewards,
        statuses=statuses,
        steps=outcome.steps,
        crashed=outcome.crashed,
        timeouts=len(outcome.timeouts),
    )


def run_episode(
    agents: list[Any],
    *,
    steps: int = FULL_SEASON_STEPS,
    seed: int | None = None,
    debug: bool = True,
    replay_path: Path | None = None,
) -> EpisodeResult:
    """Run one episode between two agent specs and return its result.

    ``debug`` is accepted for compatibility and ignored: the episode always
    runs with the engine's debug mode on, because that is the only way a
    crashing agent is distinguishable from a losing one.
    """
    del debug
    if len(agents) != 2:
        raise ValueError(f"expected exactly 2 agents, got {len(agents)}")

    spec = MatchSpec(
        case=str(agents[0]),
        opponent=str(agents[1]),
        seed=seed if seed is not None else 0,
        steps=steps,
    )
    outcome = _run_episode(spec, validate_actions=False, replay_path=replay_path)
    return _to_result(outcome)


def run_match(
    agents: list[Any],
    *,
    episodes: int = 1,
    steps: int = FULL_SEASON_STEPS,
    seed: int | None = None,
    debug: bool = True,
    replay_dir: Path | None = None,
) -> MatchSummary:
    """Run ``episodes`` episodes and aggregate the results.

    When ``seed`` is given, episode *i* uses ``seed + i`` so the run is
    reproducible while still varying across episodes.
    """
    summary = MatchSummary()
    for i in range(episodes):
        replay_path = replay_dir / f"episode_{i:04d}.json" if replay_dir else None
        summary.episodes.append(
            run_episode(
                agents,
                steps=steps,
                seed=None if seed is None else seed + i,
                debug=debug,
                replay_path=replay_path,
            )
        )
    return summary
