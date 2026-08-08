"""Fetch leaderboard episodes (replays + agent logs) from Kaggle.

Two phases, deliberately separated:

* **plan**  — read the leaderboard, list each team's episodes, diff against
  what is already on disk, and write a JSON plan. Cheap and reviewable.
* **fetch** — download the planned episodes, checkpointing to DVC periodically
  so a mid-run failure (or an Actions timeout) still persists what it got.

Two API layers are needed, because neither covers the whole job:

* The **public SDK** (``kaggle.api``) downloads replays and logs, and serves
  the leaderboard — but its leaderboard rows carry no submission id.
* Kaggle's **internal** ``/api/i/competitions.*`` endpoints map a team id to
  its leaderboard submission id and list that submission's episodes. There is
  no public equivalent, so the team → episodes hop goes through them.

Verified against the live API: agent logs return **403 for other people's
submissions** — Kaggle only serves logs you own. Log fetching is therefore
best-effort and never fails a run; replays carry the actions and observations
that matter for training anyway.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

COMPETITION = "kaggriculture"

INTERNAL_BASE_URL = "https://www.kaggle.com/api/i/competitions."
LEADERBOARD_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/leaderboard"

#: Repo-root-relative destinations, both DVC-tracked.
REPLAY_DIR = Path("data/lake/kaggle_episodes/replays")
LOG_DIR = Path("data/lake/kaggle_episodes/logs")

#: Replays run ~29 MB each, so defaults stay small on purpose — the cron job
#: accumulates history incrementally rather than pulling hundreds of GB once.
DEFAULT_TOP = 20
DEFAULT_LIMIT_PER_TEAM = 5


class KaggleScrapeError(RuntimeError):
    """Raised when a Kaggle API call fails unrecoverably."""


# --------------------------------------------------------------------------
# plan / fetch data model
# --------------------------------------------------------------------------


@dataclass
class PlannedEpisode:
    episode_id: int
    team_id: int
    team_name: str
    submission_id: int


@dataclass
class ScrapePlan:
    competition: str = COMPETITION
    episodes: list[PlannedEpisode] = field(default_factory=list)
    skipped_existing: int = 0
    teams_scanned: int = 0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "competition": self.competition,
                    "skipped_existing": self.skipped_existing,
                    "teams_scanned": self.teams_scanned,
                    "episodes": [asdict(e) for e in self.episodes],
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> ScrapePlan:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            competition=payload.get("competition", COMPETITION),
            episodes=[PlannedEpisode(**e) for e in payload.get("episodes", [])],
            skipped_existing=int(payload.get("skipped_existing", 0)),
            teams_scanned=int(payload.get("teams_scanned", 0)),
        )


class RateLimiter:
    """Token bucket: at most ``capacity`` calls per ``window`` seconds."""

    def __init__(self, capacity: int = 45, window: float = 60.0) -> None:
        self.capacity = capacity
        self.window = window
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.window:
                self._calls.popleft()
            if len(self._calls) < self.capacity:
                self._calls.append(now)
                return
            sleep_for = self.window - (now - self._calls[0])
            if sleep_for > 0:
                logger.info("rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------


def _credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    path = Path("~/.kaggle/kaggle.json").expanduser()
    if not path.exists():
        raise KaggleScrapeError(
            f"No Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY or create {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("username") or not data.get("key"):
        raise KaggleScrapeError(f"{path} is missing username/key")
    return str(data["username"]), str(data["key"])


def build_internal_session() -> requests.Session:
    """Authenticated session for Kaggle's internal endpoints.

    Needs Basic auth *and* a session cookie + matching XSRF header, so the
    leaderboard page is fetched first purely to seed the cookie jar.
    """
    username, key = _credentials()
    session = requests.Session()
    session.auth = (username, key)
    session.headers.update(
        {
            "User-Agent": "kaggriculture-scraper/0.1",
            "Content-Type": "application/json",
        }
    )
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        ),
        pool_connections=16,
        pool_maxsize=16,
    )
    session.mount("https://", adapter)

    session.get(LEADERBOARD_URL, timeout=30)
    xsrf = session.cookies.get("XSRF-TOKEN")
    if xsrf:
        session.headers["X-XSRF-TOKEN"] = xsrf
    return session


def _post_internal(
    session: requests.Session, path: str, body: dict[str, Any], *, timeout: float = 30.0
) -> dict[str, Any]:
    url = f"{INTERNAL_BASE_URL}{path}"
    try:
        response = session.post(url, json=body, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        raise KaggleScrapeError(f"{path} failed: {exc}") from exc
    except ValueError as exc:
        raise KaggleScrapeError(f"{path} returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise KaggleScrapeError(f"{path} returned {type(payload).__name__}, not dict")
    return payload


def leaderboard_team_ids(top: int) -> list[tuple[int, str]]:
    """Top ``top`` (team_id, team_name) pairs, via the public SDK."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    rows: list[tuple[int, str]] = []
    page_token: str | None = None
    while len(rows) < top:
        page = api.competition_leaderboard_view(
            COMPETITION, page_size=min(100, top - len(rows)), page_token=page_token
        )
        if not page:
            break
        for entry in page:
            # The SDK returns snake_case attributes here, not camelCase.
            team_id = int(getattr(entry, "team_id", 0) or 0)
            if team_id:
                rows.append((team_id, str(getattr(entry, "team_name", "") or "")))
        break  # one page is enough for the small defaults; see DEFAULT_TOP

    return rows[:top]


def submission_id_for_team(session: requests.Session, team_id: int) -> int | None:
    """The team's public-leaderboard submission id, or None."""
    payload = _post_internal(session, "TeamService/GetTeam", {"teamId": team_id})
    for source in (payload, payload.get("team") or {}):
        value = source.get("publicLeaderboardSubmissionId")
        if value:
            return int(value)
    return None


def list_episodes(
    session: requests.Session, submission_id: int
) -> list[dict[str, Any]]:
    payload = _post_internal(
        session, "EpisodeService/ListEpisodes", {"submissionId": submission_id}
    )
    episodes = payload.get("episodes") or []
    return [e for e in episodes if isinstance(e, dict)]


def existing_episode_ids(replay_dir: Path) -> set[int]:
    """Episode ids already downloaded, parsed from ``<id>.json`` filenames."""
    if not replay_dir.is_dir():
        return set()
    ids: set[int] = set()
    for path in replay_dir.glob("*.json"):
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------


def build_plan(
    *,
    top: int = DEFAULT_TOP,
    limit_per_team: int = DEFAULT_LIMIT_PER_TEAM,
    replay_dir: Path = REPLAY_DIR,
    rate: RateLimiter | None = None,
    completed_only: bool = True,
) -> ScrapePlan:
    """Diff the leaderboard's episodes against what is already on disk."""
    limiter = rate or RateLimiter()
    session = build_internal_session()
    have = existing_episode_ids(replay_dir)

    plan = ScrapePlan()
    teams = leaderboard_team_ids(top)
    plan.teams_scanned = len(teams)
    logger.info("leaderboard: %d teams", len(teams))

    for team_id, team_name in teams:
        limiter.acquire()
        try:
            submission_id = submission_id_for_team(session, team_id)
        except KaggleScrapeError as exc:
            logger.warning("team %s (%d): GetTeam failed: %s", team_name, team_id, exc)
            continue
        if submission_id is None:
            continue

        limiter.acquire()
        try:
            episodes = list_episodes(session, submission_id)
        except KaggleScrapeError as exc:
            # One inaccessible team must not abort a multi-hour run.
            logger.warning("team %s: ListEpisodes failed: %s", team_name, exc)
            continue

        if completed_only:
            episodes = [e for e in episodes if e.get("state") == "COMPLETED"]

        selected = 0
        for episode in episodes:
            if limit_per_team > 0 and selected >= limit_per_team:
                break
            episode_id = int(episode.get("id") or 0)
            if not episode_id:
                continue
            if episode_id in have:
                plan.skipped_existing += 1
                continue
            plan.episodes.append(
                PlannedEpisode(
                    episode_id=episode_id,
                    team_id=team_id,
                    team_name=team_name,
                    submission_id=submission_id,
                )
            )
            have.add(episode_id)
            selected += 1

    logger.info(
        "plan: %d to fetch, %d already present, %d teams",
        len(plan.episodes),
        plan.skipped_existing,
        plan.teams_scanned,
    )
    return plan


def _run_shell(command: str, *, label: str) -> None:
    """Run a checkpoint/finalize command. Never raises — this runs in a finally."""
    if not command:
        return
    logger.info("%s: %s", label, command)
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=3600
        )
        if completed.returncode != 0:
            logger.warning(
                "%s failed (%d): %s",
                label,
                completed.returncode,
                completed.stderr[-500:],
            )
    except Exception as exc:
        logger.warning("%s raised: %s", label, exc)


def fetch_episodes(
    plan: ScrapePlan,
    *,
    replay_dir: Path = REPLAY_DIR,
    log_dir: Path = LOG_DIR,
    rate: RateLimiter | None = None,
    checkpoint_every: int = 25,
    checkpoint_interval_sec: float = 600.0,
    checkpoint_cmd: str = "",
    finalize_cmd: str = "",
    fetch_logs: bool = True,
) -> dict[str, int]:
    """Download planned episodes, checkpointing as we go.

    The finalize command runs in a ``finally``, so an interrupted or timed-out
    run still persists everything fetched up to that point.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    limiter = rate or RateLimiter()

    replay_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stats = {"fetched": 0, "failed": 0, "logs": 0, "logs_forbidden": 0}
    since_checkpoint = 0
    last_checkpoint = time.monotonic()

    try:
        for index, episode in enumerate(plan.episodes, start=1):
            target = replay_dir / f"{episode.episode_id}.json"
            if target.exists():
                continue

            limiter.acquire()
            try:
                api.competition_episode_replay(
                    episode.episode_id, path=str(replay_dir), quiet=True
                )
                # The SDK names the file itself; normalise to <id>.json so the
                # incremental diff above can recognise it next run.
                downloaded = replay_dir / f"episode-{episode.episode_id}-replay.json"
                if downloaded.exists():
                    downloaded.rename(target)
                if not target.exists():
                    raise KaggleScrapeError("replay file not written")
                stats["fetched"] += 1
            except Exception as exc:
                logger.warning("episode %d: replay failed: %s", episode.episode_id, exc)
                stats["failed"] += 1
                continue

            if fetch_logs:
                for agent_index in (0, 1):
                    limiter.acquire()
                    try:
                        api.competition_episode_agent_logs(
                            episode.episode_id,
                            agent_index,
                            path=str(log_dir),
                            quiet=True,
                        )
                        stats["logs"] += 1
                    except Exception as exc:
                        # Expected for other people's submissions (403).
                        if "403" in str(exc):
                            stats["logs_forbidden"] += 1
                        else:
                            logger.debug(
                                "episode %d agent %d: log unavailable: %s",
                                episode.episode_id,
                                agent_index,
                                exc,
                            )

            since_checkpoint += 1
            if (
                since_checkpoint >= checkpoint_every
                or time.monotonic() - last_checkpoint >= checkpoint_interval_sec
            ):
                logger.info("progress %d/%d — checkpointing", index, len(plan.episodes))
                _run_shell(checkpoint_cmd, label="checkpoint")
                since_checkpoint = 0
                last_checkpoint = time.monotonic()
    finally:
        _run_shell(finalize_cmd or checkpoint_cmd, label="finalize")

    if stats["logs_forbidden"]:
        logger.info(
            "%d agent logs were forbidden (403) — Kaggle only serves logs for "
            "your own submissions; replays carry the training signal.",
            stats["logs_forbidden"],
        )
    logger.info("fetch complete: %s", stats)
    return stats
