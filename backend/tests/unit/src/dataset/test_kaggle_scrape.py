"""Unit tests for the Kaggle scrape plan/fetch logic.

No network: the API layer is exercised in the live workflow, while these cover
the parts that decide *what* gets fetched — the incremental diff and the rate
limiter — because getting those wrong means either re-downloading tens of GB or
getting throttled mid-run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dataset.kaggle_scrape import (
    PlannedEpisode,
    RateLimiter,
    ScrapePlan,
    existing_episode_ids,
)


def test_existing_ids_empty_for_missing_dir(tmp_path: Path) -> None:
    assert existing_episode_ids(tmp_path / "nope") == set()


def test_existing_ids_parses_filenames(tmp_path: Path) -> None:
    for name in ("123.json", "456.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert existing_episode_ids(tmp_path) == {123, 456}


def test_existing_ids_ignores_non_numeric(tmp_path: Path) -> None:
    """DVC drops a .gitignore in tracked dirs; it must not become an id."""
    (tmp_path / "789.json").write_text("{}", encoding="utf-8")
    (tmp_path / "episode-abc-replay.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    assert existing_episode_ids(tmp_path) == {789}


def test_plan_roundtrip(tmp_path: Path) -> None:
    plan = ScrapePlan(
        episodes=[PlannedEpisode(1, 10, "team a", 100)],
        skipped_existing=3,
        teams_scanned=2,
    )
    path = tmp_path / "plan.json"
    plan.to_json(path)

    loaded = ScrapePlan.from_json(path)
    assert loaded.skipped_existing == 3
    assert loaded.teams_scanned == 2
    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].episode_id == 1
    assert loaded.episodes[0].team_name == "team a"


def test_plan_json_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    ScrapePlan(episodes=[PlannedEpisode(1, 10, "t", 100)]).to_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["competition"] == "kaggriculture"
    assert payload["episodes"][0]["episode_id"] == 1


def test_rate_limiter_allows_burst_up_to_capacity() -> None:
    limiter = RateLimiter(capacity=5, window=60.0)
    started = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - started < 0.5


def test_rate_limiter_blocks_past_capacity() -> None:
    limiter = RateLimiter(capacity=2, window=0.3)
    limiter.acquire()
    limiter.acquire()

    started = time.monotonic()
    limiter.acquire()  # must wait for the window to roll over
    assert time.monotonic() - started >= 0.2
