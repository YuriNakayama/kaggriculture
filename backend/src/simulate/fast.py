"""Fast episode execution by calling the engine's interpreter directly.

Profiling a full season shows the game rules are **1.7%** of the runtime; the
rest is framework machinery that runs once per step:

===========================  ========  =======
component                      time     share
===========================  ========  =======
``structify``                  1.00s     30.5%
other stdlib                   0.90s     27.5%
``deepcopy``                   0.84s     25.7%
``jsonschema`` validation      0.39s     11.9%
core framework                 0.08s      2.6%
game logic                     0.06s      1.7%
===========================  ========  =======

So this module keeps the official rules and strips the framework instead of
reimplementing anything: it calls ``kaggriculture.interpreter`` in a plain loop.
Measured at roughly **30x** faster than ``env.run`` with bit-identical final
money (see ``tests/e2e/src/simulate/test_fast_equivalence.py``).

What the framework does that this loop must do itself — each one verified
against ``kaggle_environments/core.py``:

- **``observation.step``** is owned by the framework (``core.py:626``), not the
  interpreter, which only reads it. Forgetting to advance it freezes ``day`` /
  ``hour``, so no end-of-day processing ever runs: no watering checks, no weed
  spawns, no animal production, no town demand. The episode still completes and
  reports a plausible-looking score, which is exactly the kind of silent
  divergence the equivalence test exists to catch.
- **``DONE`` status** is set once ``step >= episodeSteps - 1`` (``core.py:295``).
- **Agent exceptions** are caught by the framework; here they fall back to a
  ``PASS`` turn and are recorded.
- **Step history** is deliberately *not* kept — that is where the speed comes
  from. Use the ``env.run`` path when replays are needed.

This is a development-only module. It is not importable from submitted code.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from .agents import resolve_agent
from .config import ACT_TIMEOUT_SEC, SLOW_TURN_FRACTION, MatchSpec
from .episode import ENV_NAME, EpisodeOutcome, TurnCost
from .validate import ActionValidator, as_callable

#: A turn that is always legal, used when an agent raises.
PASS_TURN: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


@dataclass
class _AgentSlot:
    """One player's callable plus the failure bookkeeping the framework would do."""

    act: Callable[[Any], Any]
    crash_repr: str | None = None
    max_duration: float = 0.0

    def take_turn(self, obs: Any) -> tuple[Any, float]:
        started = time.perf_counter()
        try:
            action = self.act(obs)
        except Exception as exc:
            if self.crash_repr is None:
                import traceback

                self.crash_repr = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            action = PASS_TURN
        duration = time.perf_counter() - started
        self.max_duration = max(self.max_duration, duration)
        return action, duration


def _build_slot(spec: str, validator: ActionValidator | None) -> _AgentSlot:
    resolved = resolve_agent(spec)
    if validator is not None:
        return _AgentSlot(act=validator.wrap(resolved))
    return _AgentSlot(act=as_callable(resolved))


def run_fast_episode(
    spec: MatchSpec,
    *,
    act_timeout: float = ACT_TIMEOUT_SEC,
    slow_threshold: float | None = None,
    validate_actions: bool = True,
) -> EpisodeOutcome:
    """Run one episode via the interpreter directly.

    Returns the same :class:`EpisodeOutcome` as :func:`simulate.episode.run_episode`
    so both engines are interchangeable in the matrix runner. Replays are not
    available here — keeping no step history is what makes this fast.
    """
    if slow_threshold is None:
        slow_threshold = act_timeout * SLOW_TURN_FRACTION

    configuration: dict[str, Any] = {"episodeSteps": spec.steps, "seed": spec.seed}
    env = make(ENV_NAME, configuration=configuration, debug=False)
    state = env.state

    case_spec, opponent_spec = spec.agent_order()
    validator = ActionValidator(player=spec.case_player) if validate_actions else None
    slots = [
        _build_slot(case_spec, validator if spec.case_player == 0 else None),
        _build_slot(opponent_spec, validator if spec.case_player == 1 else None),
    ]

    timeouts: list[TurnCost] = []
    slow_turns: list[TurnCost] = []

    # env.state is already one recorded step, so episodeSteps-1 interpreter
    # calls reproduce the official step count.
    for n in range(spec.steps - 1):
        for player, slot in enumerate(slots):
            action, duration = slot.take_turn(state[player].observation)
            state[player].action = action
            cost = TurnCost(step=n, player=player, duration=duration)
            if duration > act_timeout:
                timeouts.append(cost)
            elif duration > slow_threshold:
                slow_turns.append(cost)

        engine.interpreter(state, env)

        # Owned by the framework, not the interpreter. Omitting this freezes
        # day/hour and silently disables every end-of-day rule.
        state[0].observation.step = n + 1

    farms = state[0].observation.farms
    money = [float(farms[i]["money"]) for i in range(2)]
    crash_repr = next((s.crash_repr for s in slots if s.crash_repr), None)

    return EpisodeOutcome(
        spec=spec,
        case_money=money[spec.case_player],
        opponent_money=money[spec.opponent_player],
        # The framework flips ACTIVE to DONE at the final step; mirror that so
        # `bad_status` means the same thing on both engines.
        statuses=("DONE", "DONE"),
        steps=spec.steps,
        crashed=crash_repr is not None,
        crash_repr=crash_repr,
        timeouts=tuple(timeouts),
        slow_turns=tuple(slow_turns),
        max_duration=max((s.max_duration for s in slots), default=0.0),
        issues=validator.issues if validator is not None else (),
    )
