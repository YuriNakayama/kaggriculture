"""Job construction for the evaluation matrix (no episodes run here)."""

from __future__ import annotations

from pathlib import Path

from simulate.config import ReplayMode, SimConfig
from simulate.matrix import build_jobs, default_workers


def test_one_job_per_expanded_spec() -> None:
    config = SimConfig(
        case="starter", opponents=("pass", "random"), episodes=3, workers=1
    )

    jobs = build_jobs(config)

    assert len(jobs) == 6


def test_jobs_carry_the_configured_thresholds() -> None:
    config = SimConfig(
        case="starter", opponents=("pass",), act_timeout=0.25, validate_actions=False
    )

    job = build_jobs(config)[0]

    assert job.act_timeout == 0.25
    assert job.slow_threshold == 0.125
    assert not job.validate_actions


def test_no_replay_path_when_replays_disabled(tmp_path: Path) -> None:
    config = SimConfig(
        case="starter",
        opponents=("pass",),
        replay_mode=ReplayMode.NONE,
        output_dir=tmp_path,
    )

    assert build_jobs(config)[0].replay_path is None


def test_replay_paths_are_unique_and_descriptive(tmp_path: Path) -> None:
    config = SimConfig(
        case="rulebase/case1",
        opponents=("starter",),
        episodes=2,
        swap_sides=True,
        replay_mode=ReplayMode.ALL,
        output_dir=tmp_path,
    )

    paths = [job.replay_path for job in build_jobs(config)]

    assert all(p is not None for p in paths)
    assert len(set(paths)) == len(paths)
    names = [p.name for p in paths if p is not None]
    assert any("seed0_p0" in n for n in names)
    assert any("seed0_p1" in n for n in names)


def test_replay_path_has_no_directory_separator_from_case_name(
    tmp_path: Path,
) -> None:
    # "rulebase/case1" as an opponent must not create a nested directory.
    config = SimConfig(
        case="starter",
        opponents=("rulebase/case1",),
        replay_mode=ReplayMode.ALL,
        output_dir=tmp_path,
    )

    path = build_jobs(config)[0].replay_path

    assert path is not None
    assert "rulebase_case1" in path.name


def test_no_jobs_without_opponents() -> None:
    assert build_jobs(SimConfig(case="starter", opponents=())) == []


def test_default_workers_leaves_headroom() -> None:
    assert default_workers() >= 1
