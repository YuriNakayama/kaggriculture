"""Run Kaggriculture episodes locally and summarise the outcome."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_environments import make

#: Full season: 24 turns/day x 30 days.
FULL_SEASON_STEPS = 720

#: Agents the engine provides by name.
BUILTIN_AGENTS = ("pass", "random", "starter")


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single episode."""

    rewards: list[float]
    statuses: list[str]
    steps: int

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
        """True when neither agent errored or timed out."""
        return all(s == "DONE" for s in self.statuses)


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


def run_episode(
    agents: list[Any],
    *,
    steps: int = FULL_SEASON_STEPS,
    seed: int | None = None,
    debug: bool = True,
    replay_path: Path | None = None,
) -> EpisodeResult:
    """Run one episode and return its result.

    ``agents`` entries are either callables, paths to ``main.py`` files, or the
    name of a builtin agent (``"random"`` / ``"starter"`` / ``"pass"``).

    ``debug`` defaults to True on purpose: the engine swallows agent exceptions
    and silently substitutes a no-op, so without it a crashing agent looks like
    a merely bad one.
    """
    configuration: dict[str, Any] = {"episodeSteps": steps}
    if seed is not None:
        configuration["seed"] = seed

    env = make("kaggriculture", configuration=configuration, debug=debug)
    env.run(agents)

    final = env.steps[-1]
    result = EpisodeResult(
        rewards=[float(s["reward"] or 0.0) for s in final],
        statuses=[str(s["status"]) for s in final],
        steps=len(env.steps),
    )

    if replay_path is not None:
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(env.toJSON()), encoding="utf-8")

    return result


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
