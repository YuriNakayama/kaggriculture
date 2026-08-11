"""The legacy run_episode / run_match façade still behaves as documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulate.runner import run_episode, run_match

SHORT = 30


def test_run_episode_returns_rewards_in_player_order() -> None:
    result = run_episode(["starter", "pass"], steps=SHORT, seed=1)

    assert len(result.rewards) == 2
    # `pass` never spends, so player 1 sits at startingMoney.
    assert result.rewards[1] == 3000.0
    assert result.steps == SHORT
    assert result.ok


def test_run_episode_rejects_wrong_agent_count() -> None:
    with pytest.raises(ValueError, match="exactly 2 agents"):
        run_episode(["starter"], steps=SHORT)


def test_winner_and_margin_are_player_zero_relative() -> None:
    result = run_episode(["pass", "starter"], steps=SHORT, seed=1)

    assert result.rewards[0] == 3000.0
    assert result.margin == result.rewards[0] - result.rewards[1]
    if result.rewards[0] != result.rewards[1]:
        assert result.winner in (0, 1)


def test_ok_is_false_for_a_crashing_agent(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "def agent(obs):\n"
        "    if obs['step'] == 3:\n"
        "        raise RuntimeError('nope')\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
        encoding="utf-8",
    )

    result = run_episode([str(main_py), "pass"], steps=SHORT)

    # The engine reports DONE for a crash, so `ok` must not rely on status alone.
    assert result.crashed
    assert not result.ok


def test_run_match_aggregates_episodes() -> None:
    summary = run_match(["starter", "pass"], episodes=3, steps=SHORT, seed=1)

    assert len(summary.episodes) == 3
    assert summary.wins + summary.losses + summary.ties == 3
    assert summary.errors == 0
    assert 0.0 <= summary.win_rate <= 1.0


def test_run_match_summary_is_json_ready() -> None:
    import json

    summary = run_match(["starter", "pass"], episodes=2, steps=SHORT, seed=1)

    payload = json.loads(json.dumps(summary.as_dict()))

    assert payload["episodes"] == 2
    assert payload["errors"] == 0


def test_run_match_writes_replays(tmp_path: Path) -> None:
    run_match(["starter", "pass"], episodes=2, steps=SHORT, seed=1, replay_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "episode_0000.json",
        "episode_0001.json",
    ]


def test_empty_summary_has_neutral_aggregates() -> None:
    summary = run_match(["starter", "pass"], episodes=0, steps=SHORT)

    assert summary.episodes == []
    assert summary.win_rate == 0.0
    assert summary.mean_money == 0.0
    assert summary.mean_margin == 0.0
