"""CLI: dataset acquisition.

uv run python -m dataset kaggle scrape-plan  --plan-out /tmp/plan.json
uv run python -m dataset kaggle scrape-fetch --plan-in  /tmp/plan.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .kaggle_scrape import (
    DEFAULT_LIMIT_PER_TEAM,
    DEFAULT_TOP,
    LOG_DIR,
    REPLAY_DIR,
    RateLimiter,
    ScrapePlan,
    build_plan,
    fetch_episodes,
)

app = typer.Typer(add_completion=False, help="Dataset acquisition commands.")
kaggle_app = typer.Typer(add_completion=False, help="Kaggle episode scraping.")
app.add_typer(kaggle_app, name="kaggle")


def _configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


@kaggle_app.command("scrape-plan")
def scrape_plan(
    plan_out: Path = typer.Option(..., "--plan-out"),
    top: int = typer.Option(DEFAULT_TOP, "--top", min=1),
    limit_per_team: int = typer.Option(
        DEFAULT_LIMIT_PER_TEAM, "--limit-per-team", min=0, help="0 = no limit."
    ),
    replay_dir: Path = typer.Option(REPLAY_DIR, "--replay-dir"),
    rate_capacity: int = typer.Option(45, "--rate-capacity", min=1),
    rate_window: float = typer.Option(60.0, "--rate-window", min=1.0),
    include_incomplete: bool = typer.Option(False, "--include-incomplete"),
) -> None:
    """Diff the leaderboard against local replays and write a fetch plan."""
    _configure_logging(None)
    plan = build_plan(
        top=top,
        limit_per_team=limit_per_team,
        replay_dir=replay_dir,
        rate=RateLimiter(rate_capacity, rate_window),
        completed_only=not include_incomplete,
    )
    plan.to_json(plan_out)
    typer.echo(
        json.dumps(
            {
                "plan": str(plan_out),
                "to_fetch": len(plan.episodes),
                "skipped_existing": plan.skipped_existing,
                "teams_scanned": plan.teams_scanned,
            }
        )
    )


@kaggle_app.command("scrape-fetch")
def scrape_fetch(
    plan_in: Path = typer.Option(..., "--plan-in"),
    replay_dir: Path = typer.Option(REPLAY_DIR, "--replay-dir"),
    log_dir: Path = typer.Option(LOG_DIR, "--log-dir"),
    rate_capacity: int = typer.Option(45, "--rate-capacity", min=1),
    rate_window: float = typer.Option(60.0, "--rate-window", min=1.0),
    checkpoint_every: int = typer.Option(25, "--checkpoint-every", min=1),
    checkpoint_interval_sec: float = typer.Option(
        600.0, "--checkpoint-interval-sec", min=1.0
    ),
    checkpoint_cmd: str = typer.Option("", "--checkpoint-cmd"),
    finalize_cmd: str = typer.Option("", "--finalize-cmd"),
    no_logs: bool = typer.Option(False, "--no-logs"),
    log_file: Path | None = typer.Option(None, "--log-file"),
    limit: int = typer.Option(0, "--limit", min=0, help="0 = whole plan."),
) -> None:
    """Download the planned episodes, checkpointing to DVC as it goes."""
    _configure_logging(log_file)

    plan = ScrapePlan.from_json(plan_in)
    if limit > 0:
        plan.episodes = plan.episodes[:limit]

    stats = fetch_episodes(
        plan,
        replay_dir=replay_dir,
        log_dir=log_dir,
        rate=RateLimiter(rate_capacity, rate_window),
        checkpoint_every=checkpoint_every,
        checkpoint_interval_sec=checkpoint_interval_sec,
        checkpoint_cmd=checkpoint_cmd,
        finalize_cmd=finalize_cmd,
        fetch_logs=not no_logs,
    )
    typer.echo(json.dumps(stats))


if __name__ == "__main__":
    app()
