"""Engine selection and the replay contradiction guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulate.config import Engine, ReplayMode, SimConfig
from simulate.matrix import build_jobs


def test_official_is_the_default() -> None:
    # Verification should run on the same path as the competition, so the fast
    # engine is always opt-in.
    assert SimConfig(case="starter", opponents=("pass",)).engine is Engine.OFFICIAL


def test_engine_reaches_every_job() -> None:
    config = SimConfig(
        case="starter", opponents=("pass",), episodes=3, engine=Engine.FAST
    )

    assert [job.engine for job in build_jobs(config)] == [Engine.FAST] * 3


def test_fast_engine_rejects_replays(tmp_path: Path) -> None:
    # The fast engine keeps no step history, so a replay request cannot be
    # honoured. Failing loudly beats writing nothing silently.
    with pytest.raises(ValueError, match="replays require engine=official"):
        SimConfig(
            case="starter",
            opponents=("pass",),
            engine=Engine.FAST,
            replay_mode=ReplayMode.ALL,
            output_dir=tmp_path,
        )


def test_official_engine_allows_replays(tmp_path: Path) -> None:
    config = SimConfig(
        case="starter",
        opponents=("pass",),
        engine=Engine.OFFICIAL,
        replay_mode=ReplayMode.ALL,
        output_dir=tmp_path,
    )

    assert build_jobs(config)[0].replay_path is not None


def test_fast_engine_without_replays_is_fine() -> None:
    config = SimConfig(case="starter", opponents=("pass",), engine=Engine.FAST)

    assert build_jobs(config)[0].replay_path is None
