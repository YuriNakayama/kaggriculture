"""Integration tests for the harness-faithful archive verification.

``verify_archive`` runs the unpacked archive in an isolated Python 3.11
interpreter (the measured production version) with only ``kaggle-environments``
installed, ``exec``s ``main.py`` without ``__name__`` / ``__file__``, and uses
a cwd that is not the agent dir. These tests pin the failure modes that
previously burned submission slots:

- a relative import in ``main.py`` (no package context under exec)
- an import of a dev-only ``backend/src`` package (absent in the tarball env)

and the success mode a hierarchical case relies on: subpackage absolute import
from ``main.py``, relative imports inside the package, and a data file loaded
via a package module's ``__file__``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from submit.packaging import SubmitError, build_archive, verify_archive

#: Short episode: enough to exercise the load path without a full season.
STEPS = 24


def _write_hierarchical_case(case: Path) -> None:
    case.mkdir(parents=True)
    (case / "main.py").write_text(
        textwrap.dedent(
            """\
            from pkg.core import fingerprint

            CHECK = fingerprint()  # fails at import time if data/load breaks


            def agent(obs):
                return {"farmer": ["PASS"], "hands": [], "market": []}
            """
        ),
        encoding="utf-8",
    )
    pkg = case / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "util.py").write_text(
        "def scale(x, f):\n    return x * f\n", encoding="utf-8"
    )
    (pkg / "core.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            import numpy as np

            from .util import scale


            def fingerprint():
                w = np.load(Path(__file__).parent / "data" / "w.npy")
                return scale(float(w.sum()), 1.0)
            """
        ),
        encoding="utf-8",
    )
    (pkg / "data").mkdir()
    np.save(pkg / "data" / "w.npy", np.arange(4, dtype=np.float32))


@pytest.fixture
def pipeline_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import submit.packaging as pkg

    root = tmp_path / "pipeline"
    monkeypatch.setattr(pkg, "PIPELINE_ROOT", root)
    return root


def test_hierarchical_case_verifies(pipeline_root: Path, tmp_path: Path) -> None:
    _write_hierarchical_case(pipeline_root / "fake" / "case1")

    archive = build_archive("fake/case1", tmp_path / "s.tar.gz")
    result = verify_archive(archive, steps=STEPS)

    assert result["statuses"] == ["DONE", "DONE"]
    assert str(result["python"]).startswith("3.11")


def test_relative_import_in_main_fails(pipeline_root: Path, tmp_path: Path) -> None:
    """main.py has no package context under the harness exec."""
    case = pipeline_root / "fake" / "case2"
    case.mkdir(parents=True)
    (case / "policy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (case / "main.py").write_text(
        "from .policy import VALUE\n\n\ndef agent(obs):\n    return {}\n",
        encoding="utf-8",
    )

    archive = build_archive("fake/case2", tmp_path / "s.tar.gz")
    with pytest.raises(SubmitError, match="failed verification"):
        verify_archive(archive, steps=STEPS)


def test_dev_only_import_fails(pipeline_root: Path, tmp_path: Path) -> None:
    """backend/src packages do not exist in the isolated harness env."""
    case = pipeline_root / "fake" / "case3"
    case.mkdir(parents=True)
    (case / "main.py").write_text(
        "from simulate import anything\n\n\ndef agent(obs):\n    return {}\n",
        encoding="utf-8",
    )

    archive = build_archive("fake/case3", tmp_path / "s.tar.gz")
    with pytest.raises(SubmitError, match="failed verification"):
        verify_archive(archive, steps=STEPS)
