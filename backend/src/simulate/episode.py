"""Run one episode and report *why* it went the way it did.

The engine hides agent failures, which is the problem this module exists to
solve. Measured against ``kaggle-environments`` 1.32.6:

============================  ================  ==============================
agent behaviour               ``debug=False``   ``debug=True``
============================  ================  ==============================
raises an exception           ``status=DONE``   ``env.run`` re-raises
exceeds ``actTimeout``        ``status=DONE``   ``status=DONE``
============================  ================  ==============================

So ``status`` alone reports a broken agent as merely a weak one. Three
independent channels are needed instead:

1. **crash** — run with ``debug=True`` and catch what ``env.run`` re-raises.
2. **timeout** — ``env.logs[step][player]["duration"]`` is the only evidence a
   turn ran long; the status never reflects it.
3. **invalid action** — bad ops are silent no-ops, so the agent's output is
   inspected before the engine sees it (see :mod:`simulate.validate`).
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .agents import resolve_agent
from .config import ACT_TIMEOUT_SEC, SLOW_TURN_FRACTION, MatchSpec
from .validate import ActionValidator, ValidationIssue

ENV_NAME = "kaggriculture"


@dataclass(frozen=True)
class TurnCost:
    """Wall-clock a single agent turn took, as reported by ``env.logs``."""

    step: int
    player: int
    duration: float


@dataclass(frozen=True)
class EpisodeOutcome:
    """Result of one episode, always from the case-under-test's perspective."""

    spec: MatchSpec
    case_money: float
    opponent_money: float
    statuses: tuple[str, ...]
    steps: int
    crashed: bool = False
    crash_repr: str | None = None
    timeouts: tuple[TurnCost, ...] = ()
    slow_turns: tuple[TurnCost, ...] = ()
    max_duration: float = 0.0
    issues: tuple[ValidationIssue, ...] = ()
    replay_path: Path | None = None
    stderr: str = ""

    @property
    def margin(self) -> float:
        """Case money minus opponent money."""
        return self.case_money - self.opponent_money

    @property
    def winner_is_case(self) -> bool:
        return self.case_money > self.opponent_money

    @property
    def is_tie(self) -> bool:
        return self.case_money == self.opponent_money

    @property
    def bad_status(self) -> bool:
        return any(s != "DONE" for s in self.statuses)

    @property
    def invalid_actions(self) -> int:
        return len(self.issues)

    @property
    def failed(self) -> bool:
        """A failure is a broken agent, not a losing one."""
        return self.crashed or bool(self.timeouts) or self.bad_status


def _turn_costs(
    logs: Sequence[Any] | None,
    *,
    act_timeout: float,
    slow_threshold: float,
) -> tuple[tuple[TurnCost, ...], tuple[TurnCost, ...], float]:
    """Extract timeouts and slow turns from ``env.logs``.

    ``env.logs`` is indexed ``[step][player]`` and each entry carries
    ``duration`` / ``stdout`` / ``stderr``. This is the only place a
    time-limit violation is observable.
    """
    timeouts: list[TurnCost] = []
    slow: list[TurnCost] = []
    max_duration = 0.0

    for step, per_player in enumerate(logs or []):
        if not per_player:
            continue
        for player, entry in enumerate(per_player):
            if not isinstance(entry, dict):
                continue
            raw = entry.get("duration")
            if not isinstance(raw, (int, float)):
                continue
            duration = float(raw)
            max_duration = max(max_duration, duration)
            cost = TurnCost(step=step, player=player, duration=duration)
            if duration > act_timeout:
                timeouts.append(cost)
            elif duration > slow_threshold:
                slow.append(cost)

    return tuple(timeouts), tuple(slow), max_duration


def _collect_stderr(logs: Sequence[Any] | None, player: int) -> str:
    """Concatenate one player's stderr across the episode."""
    chunks: list[str] = []
    for per_player in logs or []:
        if not per_player or player >= len(per_player):
            continue
        entry = per_player[player]
        if isinstance(entry, dict):
            text = entry.get("stderr")
            if isinstance(text, str) and text.strip():
                chunks.append(text.rstrip())
    return "\n".join(chunks)


@dataclass
class _RunArtifacts:
    """What a completed (or crashed) ``env.run`` left behind."""

    rewards: tuple[float, float] = (0.0, 0.0)
    statuses: tuple[str, ...] = ()
    steps: int = 0
    logs: list[list[dict[str, Any]]] = field(default_factory=list)
    replay: dict[str, Any] | None = None
    crash_repr: str | None = None


def _harvest(env: Any) -> _RunArtifacts:
    """Pull results off an env that ran to completion."""
    final = env.steps[-1]
    return _RunArtifacts(
        rewards=(
            float(final[0]["reward"] or 0.0),
            float(final[1]["reward"] or 0.0),
        ),
        statuses=tuple(str(s["status"]) for s in final),
        steps=len(env.steps),
        logs=list(env.logs or []),
        replay=env.toJSON(),
    )


def _execute(
    agents: list[Any],
    *,
    steps: int,
    seed: int | None,
    extra_config: dict[str, Any] | None,
) -> tuple[_RunArtifacts, str]:
    """Run the episode with ``debug=True`` so crashes surface as exceptions."""
    configuration: dict[str, Any] = {"episodeSteps": steps}
    if seed is not None:
        configuration["seed"] = seed
    if extra_config:
        configuration.update(extra_config)

    # debug=True is what turns a silent DONE into a raised exception. It also
    # makes the engine print tracebacks to stderr, which is captured rather
    # than dumped over the progress output.
    buffer = io.StringIO()
    env = make(ENV_NAME, configuration=configuration, debug=True)

    try:
        with contextlib.redirect_stderr(buffer):
            env.run(agents)
    except Exception:
        artifacts = _RunArtifacts(crash_repr=traceback.format_exc())
        artifacts.logs = list(getattr(env, "logs", None) or [])
        artifacts.steps = len(getattr(env, "steps", None) or [])
        return artifacts, buffer.getvalue()

    return _harvest(env), buffer.getvalue()


def run_episode(
    spec: MatchSpec,
    *,
    act_timeout: float = ACT_TIMEOUT_SEC,
    slow_threshold: float | None = None,
    validate_actions: bool = True,
    replay_path: Path | None = None,
    keep_replay: Callable[[EpisodeOutcome], bool] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> EpisodeOutcome:
    """Run the episode described by ``spec``.

    Returns an :class:`EpisodeOutcome` even when the agent crashes — a crash is
    data about the case, not an error for the caller to handle. Only an
    unresolvable agent spec raises.
    """
    if slow_threshold is None:
        slow_threshold = act_timeout * SLOW_TURN_FRACTION

    case_spec, opponent_spec = spec.agent_order()
    agents: list[Any] = [resolve_agent(case_spec), resolve_agent(opponent_spec)]

    validator: ActionValidator | None = None
    if validate_actions:
        validator = ActionValidator(player=spec.case_player)
        agents[spec.case_player] = validator.wrap(agents[spec.case_player])

    artifacts, stderr = _execute(
        agents, steps=spec.steps, seed=spec.seed, extra_config=extra_config
    )

    timeouts, slow_turns, max_duration = _turn_costs(
        artifacts.logs, act_timeout=act_timeout, slow_threshold=slow_threshold
    )

    case_money = artifacts.rewards[spec.case_player] if artifacts.statuses else 0.0
    opponent_money = (
        artifacts.rewards[spec.opponent_player] if artifacts.statuses else 0.0
    )

    agent_stderr = _collect_stderr(artifacts.logs, spec.case_player)

    outcome = EpisodeOutcome(
        spec=spec,
        case_money=case_money,
        opponent_money=opponent_money,
        statuses=artifacts.statuses,
        steps=artifacts.steps,
        crashed=artifacts.crash_repr is not None,
        crash_repr=artifacts.crash_repr,
        timeouts=timeouts,
        slow_turns=slow_turns,
        max_duration=max_duration,
        issues=validator.issues if validator is not None else (),
        stderr=agent_stderr or stderr,
    )

    if replay_path is not None and artifacts.replay is not None:
        if keep_replay is None or keep_replay(outcome):
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_path.write_text(json.dumps(artifacts.replay), encoding="utf-8")
            outcome = EpisodeOutcome(**{**vars(outcome), "replay_path": replay_path})

    return outcome
