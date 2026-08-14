"""CLI: build, verify, and submit an agent archive.

uv run python -m submit --case rulebase/case1 --dry-run
uv run python -m submit --case rulebase/case1 -m "wheat loop v1"
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import typer

from .packaging import (
    DAILY_SUBMISSION_LIMIT,
    SubmitError,
    build_archive,
    git_state,
    record_submission,
    submissions_today,
    submit_to_kaggle,
    verify_archive,
)

app = typer.Typer(add_completion=False, help="Package and submit an agent.")


@app.command()
def main(
    case: str = typer.Option(..., "--case", help="e.g. rulebase/case1"),
    message: str = typer.Option("", "-m", "--message"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build and verify only; do not upload."
    ),
    out: Path | None = typer.Option(None, "--out", help="Keep the archive here."),
    steps: int = typer.Option(720, "--steps", min=1),
    skip_verify: bool = typer.Option(
        False, "--skip-verify", help="Not recommended — see rules/backend/submit.md."
    ),
    force: bool = typer.Option(
        False, "--force", help="Submit even if the local daily count is at the cap."
    ),
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = out or Path(tmp) / "submission.tar.gz"

        try:
            typer.echo(f"== building {case} ==")
            archive = build_archive(case, archive_path)
            typer.echo(
                f"   {archive.path.name}  "
                f"{archive.size_bytes / 1024:.1f} KB  "
                f"{len(archive.members)} files"
            )
            typer.echo(f"   members: {', '.join(archive.members)}")

            if skip_verify:
                typer.echo("== verification SKIPPED ==")
            else:
                typer.echo(f"== verifying (unpack + {steps}-step episode) ==")
                verification = verify_archive(archive, steps=steps)
                raw = verification.get("rewards")
                rewards = raw if isinstance(raw, list) and len(raw) == 2 else [0.0, 0.0]
                typer.echo(
                    f"   ok — statuses {verification.get('statuses')}  "
                    f"money {float(rewards[0]):,.0f} "
                    f"vs starter {float(rewards[1]):,.0f}"
                )

            if dry_run:
                typer.echo("== dry run: not submitting ==")
                if out:
                    typer.echo(f"   archive kept at {out}")
                return

            if not message:
                raise SubmitError("-m/--message is required for a real submission")

            used = submissions_today()
            if used >= DAILY_SUBMISSION_LIMIT and not force:
                raise SubmitError(
                    f"{used}/{DAILY_SUBMISSION_LIMIT} submissions already recorded "
                    "today. Kaggle scores only the latest submissions, so spending "
                    "the last slot is rarely what you want. Pass --force to override."
                )

            _, dirty = git_state()
            if dirty:
                typer.echo(
                    "   WARNING: working tree is dirty — the recorded git sha "
                    "will not reproduce this archive. Commit first for a clean "
                    "audit trail."
                )

            typer.echo(f"== submitting ({used + 1}/{DAILY_SUBMISSION_LIMIT} today) ==")
            output = submit_to_kaggle(archive, message)
            typer.echo(f"   {output}")

            record = record_submission(
                archive, message, {} if skip_verify else verification
            )
            typer.echo(f"   recorded: {record}")
            typer.echo("   check status: dev/kaggle submissions kaggriculture")

        except SubmitError as exc:
            typer.echo(f"\nERROR: {exc}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    app()
