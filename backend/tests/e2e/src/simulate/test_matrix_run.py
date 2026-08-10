"""Matrix execution, sequential and across processes."""

from __future__ import annotations

from pathlib import Path

from simulate.config import ReplayMode, SimConfig
from simulate.matrix import run_matrix
from simulate.report import ExitCode, build_report

SHORT = 30


def _config(**kwargs: object) -> SimConfig:
    defaults: dict[str, object] = {
        "case": "starter",
        "opponents": ("pass",),
        "steps": SHORT,
        "workers": 1,
    }
    return SimConfig(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_sequential_run_returns_one_outcome_per_spec() -> None:
    outcomes = run_matrix(_config(episodes=2, opponents=("pass", "random")))

    assert len(outcomes) == 4
    assert all(not o.crashed for o in outcomes)


def test_parallel_run_matches_sequential_totals() -> None:
    config_kwargs: dict[str, object] = {"episodes": 4, "opponents": ("pass",)}

    sequential = run_matrix(_config(workers=1, **config_kwargs))
    parallel = run_matrix(_config(workers=4, **config_kwargs))

    assert len(sequential) == len(parallel) == 4
    # `pass` never spends, so this is stable across both execution modes.
    assert {o.opponent_money for o in sequential} == {3000.0}
    assert {o.opponent_money for o in parallel} == {3000.0}


def test_parallel_run_preserves_seed_coverage() -> None:
    outcomes = run_matrix(_config(episodes=5, workers=4, seed_base=10))

    # Completion order is not submission order, so compare as a set.
    assert {o.spec.seed for o in outcomes} == {10, 11, 12, 13, 14}


def test_on_result_callback_fires_per_episode() -> None:
    seen: list[int] = []

    run_matrix(_config(episodes=3), on_result=lambda o: seen.append(o.spec.seed))

    assert len(seen) == 3


def test_swap_sides_covers_both_seats() -> None:
    outcomes = run_matrix(_config(episodes=2, swap_sides=True))

    assert len(outcomes) == 4
    assert {o.spec.swap_sides for o in outcomes} == {False, True}


def test_replay_all_writes_one_file_per_episode(tmp_path: Path) -> None:
    config = _config(episodes=2, replay_mode=ReplayMode.ALL, output_dir=tmp_path)

    run_matrix(config)

    written = list((tmp_path / "replays").glob("*.json"))
    assert len(written) == 2


def test_replay_failed_writes_nothing_for_a_clean_run(tmp_path: Path) -> None:
    config = _config(episodes=2, replay_mode=ReplayMode.FAILED, output_dir=tmp_path)

    run_matrix(config)

    assert not list((tmp_path / "replays").glob("*.json"))


def test_report_over_a_real_run_is_clean() -> None:
    config = _config(episodes=2, opponents=("pass", "starter"))

    outcomes = run_matrix(config)
    report = build_report(config, outcomes, elapsed_sec=1.0)

    assert report.episodes == 4
    assert report.health.clean
    assert report.exit_code is ExitCode.OK
    assert report.health.max_duration < config.act_timeout


def test_crashing_case_fails_the_matrix(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "def agent(obs):\n"
        "    if obs['step'] == 3:\n"
        "        raise RuntimeError('nope')\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
        encoding="utf-8",
    )
    config = _config(case=str(main_py), episodes=2, workers=2)

    outcomes = run_matrix(config)
    report = build_report(config, outcomes, elapsed_sec=1.0)

    assert report.health.crashes == 2
    assert report.exit_code is ExitCode.FAILED


def test_empty_matrix_returns_no_outcomes() -> None:
    assert run_matrix(_config(opponents=())) == []
