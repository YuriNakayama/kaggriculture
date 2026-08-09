"""Unit tests for submission packaging.

The archive contract fails *after* a submission slot is spent, so these guard
the two mistakes that cost a slot: `main.py` not at the archive root, and
training-only files leaking into the bundle.

The tests build their own throwaway case under a temp pipeline root rather
than packaging a real one — agent cases come and go, and a test that breaks
when a case is deleted is testing the wrong thing.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from submit.packaging import (
    EXCLUDE_NAMES,
    SubmitError,
    build_archive,
    case_dir,
    submissions_today,
)


@pytest.fixture
def fake_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A minimal multi-file case, wired in as the pipeline root."""
    import submit.packaging as pkg

    case = tmp_path / "pipeline" / "fake" / "case1"
    case.mkdir(parents=True)
    (case / "main.py").write_text("def agent(obs):\n    return {}\n", encoding="utf-8")
    (case / "policy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (case / "weights.npz").write_bytes(b"\x00binary")
    (case / "train.py").write_text("import torch\n", encoding="utf-8")
    (case / "__init__.py").write_text("", encoding="utf-8")
    (case / "__pycache__").mkdir()
    (case / "__pycache__" / "main.cpython-313.pyc").write_bytes(b"\x00")

    monkeypatch.setattr(pkg, "PIPELINE_ROOT", tmp_path / "pipeline")
    return "fake/case1"


def test_case_dir_resolves_existing_case(fake_case: str) -> None:
    assert case_dir(fake_case).is_dir()


def test_case_dir_rejects_missing_case(fake_case: str) -> None:
    with pytest.raises(SubmitError, match="no such case"):
        case_dir("fake/case999")


def test_case_dir_rejects_case_without_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit.packaging as pkg

    empty = tmp_path / "pipeline" / "fake" / "case2"
    empty.mkdir(parents=True)
    monkeypatch.setattr(pkg, "PIPELINE_ROOT", tmp_path / "pipeline")

    with pytest.raises(SubmitError, match="no main.py"):
        case_dir("fake/case2")


def test_main_py_lands_at_archive_root(fake_case: str, tmp_path: Path) -> None:
    archive = build_archive(fake_case, tmp_path / "s.tar.gz")
    with tarfile.open(archive.path) as tar:
        names = tar.getnames()
    assert "main.py" in names, f"main.py must be at the root, got {names}"
    assert not any(n.endswith("/main.py") for n in names), "main.py must not nest"


def test_archive_excludes_pycache_and_init(fake_case: str, tmp_path: Path) -> None:
    archive = build_archive(fake_case, tmp_path / "s.tar.gz")
    assert not any("__pycache__" in m for m in archive.members)
    assert "__init__.py" not in archive.members


def test_archive_excludes_train_py(fake_case: str, tmp_path: Path) -> None:
    """train.py imports torch, which the harness may not have."""
    archive = build_archive(fake_case, tmp_path / "s.tar.gz")
    assert "train.py" not in archive.members
    assert "train.py" in EXCLUDE_NAMES


def test_archive_keeps_inference_modules(fake_case: str, tmp_path: Path) -> None:
    archive = build_archive(fake_case, tmp_path / "s.tar.gz")
    for needed in ("main.py", "policy.py", "weights.npz"):
        assert needed in archive.members, f"{needed} must ship"


def test_archive_reports_size(fake_case: str, tmp_path: Path) -> None:
    archive = build_archive(fake_case, tmp_path / "s.tar.gz")
    assert archive.size_bytes > 0


def test_submissions_today_counts_only_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit.packaging as pkg

    monkeypatch.setattr(pkg, "HISTORY_DIR", tmp_path)
    (tmp_path / "old.json").write_text(
        json.dumps({"submitted_at": "2020-01-01T00:00:00+00:00"}), encoding="utf-8"
    )
    assert submissions_today() == 0


def test_submissions_today_empty_when_no_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import submit.packaging as pkg

    monkeypatch.setattr(pkg, "HISTORY_DIR", tmp_path / "nope")
    assert submissions_today() == 0
