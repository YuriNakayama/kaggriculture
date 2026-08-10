"""Episode-level failure detection.

These tests pin down measured engine behaviour: a crashing agent and a
timing-out agent both finish with ``status == "DONE"``, so the outcome must
carry independent crash / timeout signals. If a future engine version changes
this, these tests are the tripwire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simulate.config import MatchSpec
from simulate.episode import EpisodeOutcome, run_episode

PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

# Short seasons keep these tests fast; the failure paths do not depend on length.
SHORT = 40


def _spec(**kwargs: Any) -> MatchSpec:
    defaults: dict[str, Any] = {
        "case": "starter",
        "opponent": "pass",
        "seed": 1,
        "steps": SHORT,
    }
    return MatchSpec(**{**defaults, **kwargs})


def _write_agent(tmp_path: Path, body: str) -> str:
    main_py = tmp_path / "main.py"
    main_py.write_text(body, encoding="utf-8")
    return str(main_py)


def test_clean_episode_reports_no_failure() -> None:
    outcome = run_episode(_spec())

    assert not outcome.failed
    assert not outcome.crashed
    assert outcome.timeouts == ()
    assert outcome.statuses == ("DONE", "DONE")
    assert outcome.steps == SHORT


def test_crashing_agent_is_detected(tmp_path: Path) -> None:
    # With debug=False the engine swallows this and returns status=DONE.
    crasher = _write_agent(
        tmp_path,
        "def agent(obs):\n"
        "    if obs['step'] == 5:\n"
        "        raise ValueError('boom')\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
    )

    outcome = run_episode(_spec(case=crasher))

    assert outcome.crashed
    assert outcome.failed
    assert outcome.crash_repr is not None
    assert "boom" in outcome.crash_repr


def test_timeout_is_detected_from_turn_durations(tmp_path: Path) -> None:
    # A timeout never shows up in `status` — env.logs[step][player]["duration"]
    # is the only evidence, which is why the outcome tracks it separately.
    slowpoke = _write_agent(
        tmp_path,
        "import time\n"
        "def agent(obs):\n"
        "    if obs['step'] == 3:\n"
        "        time.sleep(1.2)\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
    )

    outcome = run_episode(_spec(case=slowpoke), act_timeout=0.5)

    assert outcome.timeouts, "a 1.2s turn must be flagged against a 0.5s limit"
    assert outcome.failed
    assert outcome.max_duration >= 1.0
    assert not outcome.crashed
    # The engine still calls this a clean finish; that is the whole problem.
    assert outcome.statuses == ("DONE", "DONE")


def test_slow_turn_is_warned_but_not_a_failure(tmp_path: Path) -> None:
    # A turn between the warn threshold and the hard limit is a leading
    # indicator, not a failure: it still scores, but it is drifting toward the
    # 1s ceiling on unfamiliar hardware.
    dawdler = _write_agent(
        tmp_path,
        "import time\n"
        "def agent(obs):\n"
        "    if obs['step'] == 3:\n"
        "        time.sleep(0.3)\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
    )

    outcome = run_episode(_spec(case=dawdler), act_timeout=1.0, slow_threshold=0.1)

    assert outcome.slow_turns, "a 0.3s turn must be flagged against a 0.1s warn line"
    assert outcome.timeouts == ()
    assert not outcome.failed


def test_invalid_actions_are_counted(tmp_path: Path) -> None:
    spammer = _write_agent(
        tmp_path,
        "def agent(obs):\n"
        "    return {\n"
        "        'farmer': ['PASS'],\n"
        "        'hands': [],\n"
        "        'market': [['SELL', 'WHEAT', 1]] * 12,\n"
        "    }\n",
    )

    outcome = run_episode(_spec(case=spammer))

    assert outcome.invalid_actions > 0
    # Over-capping orders wastes them but does not break the episode.
    assert not outcome.failed


def test_perspective_is_normalised_when_sides_swap() -> None:
    # `pass` never spends, so its money stays at startingMoney exactly. Asserting
    # against that fixed number proves the seat mapping without depending on how
    # a real strategy happens to be doing at this point in the season.
    as_p0 = run_episode(_spec(case="pass", opponent="starter", swap_sides=False))
    as_p1 = run_episode(_spec(case="pass", opponent="starter", swap_sides=True))

    assert as_p0.case_money == 3000.0
    assert as_p1.case_money == 3000.0
    # The opponent slot must hold starter's money, not a copy of our own.
    assert as_p0.opponent_money != 3000.0
    assert as_p1.opponent_money != 3000.0


def test_replay_is_written_when_requested(tmp_path: Path) -> None:
    replay = tmp_path / "replays" / "ep.json"

    outcome = run_episode(_spec(), replay_path=replay)

    assert replay.is_file()
    assert outcome.replay_path == replay
    assert replay.stat().st_size > 0


def test_replay_predicate_can_skip_writing(tmp_path: Path) -> None:
    replay = tmp_path / "ep.json"

    outcome = run_episode(_spec(), replay_path=replay, keep_replay=lambda o: o.failed)

    assert not replay.exists()
    assert outcome.replay_path is None


def test_seed_makes_episodes_reproducible() -> None:
    # Reproducibility holds only for deterministic agents. The builtin `random`
    # agent constructs `random.Random()` with no argument, so the env seed does
    # not constrain it — pairing deterministic agents is the only way to get a
    # repeatable episode.
    first = run_episode(_spec(case="starter", opponent="pass", seed=7))
    second = run_episode(_spec(case="starter", opponent="pass", seed=7))

    assert first.case_money == second.case_money
    assert first.opponent_money == second.opponent_money


def test_outcome_tie_detection() -> None:
    outcome = EpisodeOutcome(
        spec=_spec(),
        case_money=100.0,
        opponent_money=100.0,
        statuses=("DONE", "DONE"),
        steps=SHORT,
    )

    assert outcome.is_tie
    assert not outcome.winner_is_case
    assert outcome.margin == 0.0
    assert not outcome.failed


def test_bad_status_counts_as_failure() -> None:
    outcome = EpisodeOutcome(
        spec=_spec(),
        case_money=0.0,
        opponent_money=0.0,
        statuses=("ERROR", "DONE"),
        steps=SHORT,
    )

    assert outcome.bad_status
    assert outcome.failed
