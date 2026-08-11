"""Local evaluation against the Kaggriculture environment.

The engine hides agent failures — a crash and a timeout both finish with
``status == "DONE"``, and invalid actions are silent no-ops — so this package
runs episodes with independent crash / timeout / invalid-action detection and
reports all three next to the win rate. See :mod:`simulate.episode`.

Typical use::

    from simulate import SimConfig, run_matrix, build_report

    config = SimConfig(case="rulebase/case1", opponents=("starter",), episodes=50)
    outcomes = run_matrix(config)
    report = build_report(config, outcomes, elapsed_sec=elapsed)
"""

from __future__ import annotations

from .agents import BUILTIN_AGENTS, AgentResolutionError, resolve_agent
from .config import (
    ACT_TIMEOUT_SEC,
    FULL_SEASON_STEPS,
    Engine,
    MatchSpec,
    ReplayMode,
    SimConfig,
)
from .episode import EpisodeOutcome, TurnCost
from .episode import run_episode as run_single_episode
from .fast import run_fast_episode
from .matrix import default_workers, run_matrix
from .report import (
    ExitCode,
    HealthReport,
    MatrixReport,
    OpponentReport,
    build_report,
    render,
)
from .runner import EpisodeResult, MatchSummary, run_episode, run_match
from .validate import ActionValidator, ValidationIssue

__all__ = [
    "ACT_TIMEOUT_SEC",
    "BUILTIN_AGENTS",
    "FULL_SEASON_STEPS",
    "ActionValidator",
    "AgentResolutionError",
    "Engine",
    "EpisodeOutcome",
    "EpisodeResult",
    "ExitCode",
    "HealthReport",
    "MatchSpec",
    "MatchSummary",
    "MatrixReport",
    "OpponentReport",
    "ReplayMode",
    "SimConfig",
    "TurnCost",
    "ValidationIssue",
    "build_report",
    "default_workers",
    "render",
    "resolve_agent",
    "run_episode",
    "run_match",
    "run_fast_episode",
    "run_matrix",
    "run_single_episode",
]
