"""CLI: run local Kaggriculture matches.

uv run python -m simulate --case rulebase/case1 --opponent random
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

from .runner import BUILTIN_AGENTS, FULL_SEASON_STEPS, run_match

app = typer.Typer(add_completion=False, help="Run Kaggriculture matches locally.")

#: backend/ root, i.e. the parent of src/.
BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _resolve_agent(spec: str) -> str:
    """Turn a case spec or builtin name into something env.run accepts."""
    if spec in BUILTIN_AGENTS:
        return spec

    candidate = Path(spec)
    if candidate.is_file():
        return str(candidate.resolve())

    main_py = BACKEND_ROOT / "pipeline" / spec / "main.py"
    if main_py.is_file():
        return str(main_py)

    raise typer.BadParameter(
        f"{spec!r} is not a builtin agent ({', '.join(BUILTIN_AGENTS)}), "
        f"an existing file, or a case under backend/pipeline/."
    )


@app.command()
def main(
    case: str = typer.Option(..., "--case", help="Case path or builtin agent name."),
    opponent: str = typer.Option("random", "--opponent", help="Opponent agent."),
    episodes: int = typer.Option(1, "--episodes", min=1),
    steps: int = typer.Option(FULL_SEASON_STEPS, "--steps", min=1),
    seed: int | None = typer.Option(None, "--seed"),
    replay_dir: Path | None = typer.Option(None, "--replay-dir"),
    json_out: bool = typer.Option(False, "--json", help="Print JSON only."),
) -> None:
    agents = [_resolve_agent(case), _resolve_agent(opponent)]

    started = time.monotonic()
    summary = run_match(
        agents,
        episodes=episodes,
        steps=steps,
        seed=seed,
        replay_dir=replay_dir,
    )
    elapsed = time.monotonic() - started

    payload = summary.as_dict() | {
        "case": case,
        "opponent": opponent,
        "steps": steps,
        "elapsed_sec": round(elapsed, 1),
    }

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        typer.echo(f"{case}  vs  {opponent}   ({episodes} ep x {steps} steps)")
        typer.echo(
            f"  W/L/T   : {summary.wins}/{summary.losses}/{summary.ties}"
            f"   (win rate {summary.win_rate:.1%})"
        )
        typer.echo(f"  money   : mean {summary.mean_money:,.0f}")
        typer.echo(f"  margin  : mean {summary.mean_margin:+,.0f}")
        typer.echo(f"  errors  : {summary.errors}")
        typer.echo(f"  elapsed : {elapsed:.1f}s")

    # A crashed or timed-out agent is a hard failure, not a bad score.
    if summary.errors:
        for i, ep in enumerate(summary.episodes):
            if not ep.ok:
                typer.echo(f"  episode {i}: statuses={ep.statuses}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
