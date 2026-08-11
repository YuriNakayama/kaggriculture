"""CLI: run local Kaggriculture evaluation matrices.

uv run python -m simulate --case rulebase/case1 --opponent starter,random
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from .agents import AgentResolutionError, resolve_agent
from .config import (
    ACT_TIMEOUT_SEC,
    FULL_SEASON_STEPS,
    Engine,
    ReplayMode,
    SimConfig,
)
from .matrix import default_workers, run_matrix
from .report import build_report, render

app = typer.Typer(add_completion=False, help="Run Kaggriculture matches locally.")

#: repo root, i.e. the parent of backend/.
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "output" / "simulate"


def _parse_opponents(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise typer.BadParameter("at least one opponent is required")
    return names


def _validate_specs(case: str, opponents: tuple[str, ...]) -> None:
    """Resolve every agent up front.

    A typo'd case would otherwise be swapped for a silent no-op agent by the
    engine and the whole matrix would report as a legitimate loss.
    """
    for spec in (case, *opponents):
        try:
            resolve_agent(spec)
        except AgentResolutionError as exc:
            raise typer.BadParameter(str(exc)) from exc


def _run_dir(output_dir: Path | None, replay_mode: ReplayMode) -> Path | None:
    if replay_mode is ReplayMode.NONE:
        return output_dir
    if output_dir is not None:
        return output_dir
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / stamp


@app.command()
def main(
    case: str = typer.Option(..., "--case", help="Case path or builtin agent name."),
    opponent: str = typer.Option(
        "starter", "--opponent", help="Comma-separated opponents."
    ),
    episodes: int = typer.Option(1, "--episodes", min=1, help="Episodes per opponent."),
    steps: int = typer.Option(FULL_SEASON_STEPS, "--steps", min=1),
    seed: int = typer.Option(0, "--seed", help="Base seed; episode i uses seed+i."),
    swap_sides: bool = typer.Option(
        False, "--swap-sides", help="Also play each seed from the other seat."
    ),
    workers: int | None = typer.Option(None, "--workers", min=1),
    engine: Engine = typer.Option(
        Engine.OFFICIAL,
        "--engine",
        help="official = same path as the competition; fast = ~30x, no replays.",
    ),
    replay: ReplayMode = typer.Option(
        ReplayMode.NONE, "--replay", help="Replay retention (~5 MB per episode)."
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip silent-no-op detection."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero when any action would be discarded."
    ),
    act_timeout: float = typer.Option(ACT_TIMEOUT_SEC, "--act-timeout"),
    json_out: Path | None = typer.Option(None, "--json", help="Write summary JSON."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the table."),
) -> None:
    opponents = _parse_opponents(opponent)
    _validate_specs(case, opponents)

    if engine is Engine.FAST and replay is not ReplayMode.NONE:
        raise typer.BadParameter(
            "--replay requires --engine official; the fast engine keeps no step history"
        )

    config = SimConfig(
        case=case,
        opponents=opponents,
        episodes=episodes,
        steps=steps,
        seed_base=seed,
        swap_sides=swap_sides,
        workers=workers if workers is not None else default_workers(),
        engine=engine,
        replay_mode=replay,
        output_dir=_run_dir(output_dir, replay),
        validate_actions=not no_validate,
        strict=strict,
        act_timeout=act_timeout,
    )

    started = time.monotonic()
    outcomes = run_matrix(config)
    elapsed = time.monotonic() - started

    report = build_report(config, outcomes, elapsed_sec=elapsed)

    if not quiet:
        typer.echo(render(report, config))

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")

    if config.output_dir is not None and replay is not ReplayMode.NONE:
        typer.echo(f"replays under {config.output_dir / 'replays'}")

    sys.exit(int(report.exit_code))


if __name__ == "__main__":
    app()
