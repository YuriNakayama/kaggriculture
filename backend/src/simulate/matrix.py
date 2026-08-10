"""Expand an evaluation matrix and run it, optionally across processes.

Workers return :class:`EpisodeOutcome` — a small dataclass — never the ``env``
itself. An env serialises to roughly 5 MB, so shipping one back per episode
would cost more in pickling than the episode costs to run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import Engine, MatchSpec, ReplayMode, SimConfig
from .episode import EpisodeOutcome, run_episode
from .fast import run_fast_episode

logger = logging.getLogger(__name__)

#: Leave a core for the OS so a long matrix does not wedge an interactive shell.
_WORKER_HEADROOM = 1


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - _WORKER_HEADROOM)


@dataclass(frozen=True)
class _Job:
    """A spec plus everything the worker needs to run it standalone."""

    spec: MatchSpec
    act_timeout: float
    slow_threshold: float
    validate_actions: bool
    replay_mode: ReplayMode
    replay_path: Path | None
    engine: Engine = Engine.OFFICIAL

    def run(self) -> EpisodeOutcome:
        if self.engine is Engine.FAST:
            return run_fast_episode(
                self.spec,
                act_timeout=self.act_timeout,
                slow_threshold=self.slow_threshold,
                validate_actions=self.validate_actions,
            )

        keep: Callable[[EpisodeOutcome], bool] | None = None
        if self.replay_mode is ReplayMode.FAILED:
            keep = lambda outcome: outcome.failed  # noqa: E731
        return run_episode(
            self.spec,
            act_timeout=self.act_timeout,
            slow_threshold=self.slow_threshold,
            validate_actions=self.validate_actions,
            replay_path=self.replay_path,
            keep_replay=keep,
        )


def _run_job(job: _Job) -> EpisodeOutcome:
    """Module-level entry point so the job is picklable for spawn."""
    return job.run()


def _replay_path_for(config: SimConfig, spec: MatchSpec, index: int) -> Path | None:
    if config.replay_mode is ReplayMode.NONE:
        return None
    if config.output_dir is None:
        return None
    seat = "p1" if spec.swap_sides else "p0"
    opponent = spec.opponent.replace("/", "_")
    name = f"ep{index:04d}_{opponent}_seed{spec.seed}_{seat}.json"
    return config.output_dir / "replays" / name


def build_jobs(config: SimConfig) -> list[_Job]:
    return [
        _Job(
            spec=spec,
            act_timeout=config.act_timeout,
            slow_threshold=config.slow_turn_threshold,
            validate_actions=config.validate_actions,
            replay_mode=config.replay_mode,
            replay_path=_replay_path_for(config, spec, i),
            engine=config.engine,
        )
        for i, spec in enumerate(config.specs())
    ]


def _run_sequential(jobs: Iterable[_Job]) -> Iterator[EpisodeOutcome]:
    for job in jobs:
        yield job.run()


def _run_parallel(jobs: list[_Job], workers: int) -> Iterator[EpisodeOutcome]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def run_matrix(
    config: SimConfig,
    *,
    on_result: Callable[[EpisodeOutcome], None] | None = None,
) -> list[EpisodeOutcome]:
    """Run every episode in ``config`` and return the outcomes.

    Results come back in completion order when parallel, so callers that care
    about ordering should sort. ``workers=1`` runs in-process, which keeps
    tracebacks and debuggers usable.
    """
    jobs = build_jobs(config)
    if not jobs:
        return []

    workers = min(config.workers, len(jobs))
    stream = _run_sequential(jobs) if workers <= 1 else _run_parallel(jobs, workers)

    outcomes: list[EpisodeOutcome] = []
    for outcome in stream:
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return outcomes
