"""Unit tests for submission packaging.

The archive contract fails *after* a submission slot is spent, so these guard
the two mistakes that cost a slot: `main.py` not at the archive root, and
training-only files leaking into the bundle.
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


def test_case_dir_resolves_existing_case() -> None:
    assert case_dir("rulebase/case1").is_dir()


def test_case_dir_rejects_missing_case() -> None:
    with pytest.raises(SubmitError, match="no such case"):
        case_dir("rulebase/case999")


def test_main_py_lands_at_archive_root(tmp_path: Path) -> None:
    archive = build_archive("rulebase/case1", tmp_path / "s.tar.gz")
    with tarfile.open(archive.path) as tar:
        names = tar.getnames()
    assert "main.py" in names, f"main.py must be at the root, got {names}"
    assert not any("/" in n and n.endswith("main.py") for n in names)


def test_archive_excludes_pycache_and_init(tmp_path: Path) -> None:
    archive = build_archive("imitation/case1", tmp_path / "s.tar.gz")
    assert not any("__pycache__" in m for m in archive.members)
    assert "__init__.py" not in archive.members


def test_archive_excludes_train_py(tmp_path: Path) -> None:
    """train.py imports torch, which the harness may not have."""
    archive = build_archive("imitation/case1", tmp_path / "s.tar.gz")
    assert "train.py" not in archive.members
    assert "train.py" in EXCLUDE_NAMES


def test_archive_keeps_inference_modules(tmp_path: Path) -> None:
    archive = build_archive("imitation/case1", tmp_path / "s.tar.gz")
    for needed in ("main.py", "features.py", "policy.py", "weights.npz"):
        assert needed in archive.members, f"{needed} must ship"


def test_archive_reports_size(tmp_path: Path) -> None:
    archive = build_archive("rulebase/case1", tmp_path / "s.tar.gz")
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
