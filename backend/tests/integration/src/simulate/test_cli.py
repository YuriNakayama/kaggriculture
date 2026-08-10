"""CLI happy paths and exit-code contract."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from simulate.__main__ import app
from simulate.report import ExitCode

runner = CliRunner()

SHORT = ("--steps", "30")


def test_reports_a_match_and_exits_zero() -> None:
    result = runner.invoke(
        app, ["--case", "starter", "--opponent", "pass", *SHORT, "--workers", "1"]
    )

    assert result.exit_code == ExitCode.OK
    assert "starter" in result.stdout
    # The health block is unconditional by design.
    assert "health" in result.stdout
    assert "crashes" in result.stdout


def test_multiple_opponents_produce_multiple_rows() -> None:
    result = runner.invoke(
        app,
        ["--case", "starter", "--opponent", "pass,random", *SHORT, "--workers", "1"],
    )

    assert result.exit_code == ExitCode.OK
    assert "pass" in result.stdout
    assert "random" in result.stdout


def test_unknown_case_is_rejected_before_running() -> None:
    result = runner.invoke(app, ["--case", "rulebase/nope", "--opponent", "pass"])

    # Resolving up front matters: the engine would otherwise substitute a no-op
    # agent and the whole matrix would read as a legitimate loss.
    assert result.exit_code != ExitCode.OK
    assert "not a builtin agent" in result.output


def test_unknown_opponent_is_rejected_before_running() -> None:
    result = runner.invoke(app, ["--case", "starter", "--opponent", "ghost"])

    assert result.exit_code != ExitCode.OK


def test_crashing_case_exits_one(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "def agent(obs):\n"
        "    if obs['step'] == 3:\n"
        "        raise RuntimeError('nope')\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["--case", str(main_py), "--opponent", "pass", *SHORT, "--workers", "1"]
    )

    assert result.exit_code == ExitCode.FAILED
    assert "crash" in result.stdout


def test_strict_mode_exits_two_on_discarded_actions(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "def agent(obs):\n"
        "    return {\n"
        "        'farmer': ['PASS'],\n"
        "        'hands': [],\n"
        "        'market': [['SELL', 'WHEAT', 1]] * 12,\n"
        "    }\n",
        encoding="utf-8",
    )
    args = ["--case", str(main_py), "--opponent", "pass", *SHORT, "--workers", "1"]

    lenient = runner.invoke(app, args)
    strict = runner.invoke(app, [*args, "--strict"])

    assert lenient.exit_code == ExitCode.OK
    assert strict.exit_code == ExitCode.INVALID_ACTIONS


def test_json_output_is_written(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"

    result = runner.invoke(
        app,
        [
            "--case",
            "starter",
            "--opponent",
            "pass",
            *SHORT,
            "--workers",
            "1",
            "--json",
            str(out),
        ],
    )

    assert result.exit_code == ExitCode.OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["case"] == "starter"
    assert payload["episodes"] == 1
    assert payload["health"]["crashes"] == 0


def test_quiet_suppresses_the_table() -> None:
    result = runner.invoke(
        app,
        [
            "--case",
            "starter",
            "--opponent",
            "pass",
            *SHORT,
            "--workers",
            "1",
            "--quiet",
        ],
    )

    assert result.exit_code == ExitCode.OK
    assert "health" not in result.stdout


def test_replay_all_writes_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--case",
            "starter",
            "--opponent",
            "pass",
            *SHORT,
            "--workers",
            "1",
            "--replay",
            "all",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == ExitCode.OK
    assert len(list((tmp_path / "replays").glob("*.json"))) == 1
