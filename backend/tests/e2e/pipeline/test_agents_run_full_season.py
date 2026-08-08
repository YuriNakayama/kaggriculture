"""End-to-end guards: every case must survive a real season and package cleanly.

These are the two failures that cost a submission slot rather than a few
points, and neither shows up in unit tests:

1. The agent raises mid-episode (the engine substitutes a no-op and the score
   silently collapses).
2. `main.py` imports fine in the repo but not from the archive root, where it
   is a top-level module with no package.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from simulate import run_episode

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = BACKEND_ROOT / "pipeline"

CASES = ["rulebase/case1", "imitation/case1"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("opponent", ["random", "starter"])
def test_case_completes_full_season(case: str, opponent: str) -> None:
    main_py = PIPELINE_ROOT / case / "main.py"
    result = run_episode([str(main_py), opponent])

    assert result.ok, f"{case} vs {opponent}: statuses={result.statuses}"
    assert result.steps == 720


@pytest.mark.parametrize("case", CASES)
def test_case_imports_from_archive_root(case: str) -> None:
    """Pack the case the way dev/submit does, then import it flat.

    Relative imports work in the repo and fail in the archive, so this has to
    run in a separate interpreter with only the unpacked directory on the path.
    """
    case_dir = PIPELINE_ROOT / case

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "submission.tar.gz"
        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()

        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(case_dir.iterdir()):
                if entry.name in {"__pycache__", "__init__.py"}:
                    continue
                tar.add(entry, arcname=entry.name)

        with tarfile.open(archive) as tar:
            names = tar.getnames()
            assert "main.py" in names, f"main.py not at archive root: {names}"
            tar.extractall(unpacked, filter="data")

        # kaggle_environments exec()s the source and takes the LAST callable
        # defined in the module, so `agent` must be the final definition —
        # a helper added below it would be handed to the engine instead.
        probe = (
            "import main;"
            "assert callable(main.agent);"
            "src = open('main.py').read();"
            "env = {};"
            "exec(compile(src, 'main.py', 'exec'), env);"
            "last = [v for v in env.values() if callable(v)][-1];"
            "assert last is env['agent'], "
            "f'last callable is {last!r}, not agent';"
            "print('ok')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=unpacked,
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert completed.returncode == 0, (
        f"{case} failed to import from archive root:\n{completed.stderr}"
    )
