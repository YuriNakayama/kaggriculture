"""Import support for the flat rulebase case modules.

A case directory has no ``__init__.py`` on purpose: the Kaggle harness ``exec``s
``main.py`` with the case directory on ``sys.path``, so its modules import each
other by bare name. Tests reproduce that layout rather than importing the case
as a package, so what is tested is what ships.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

CASE1_DIR = Path(__file__).resolve().parents[4] / "pipeline" / "rulebase" / "case2"

#: Modules the case defines; dropped between tests so each run re-imports them
#: against a clean ``sys.path``.
_CASE_MODULES = ("main", "config", "observe", "tasks", "market")


@pytest.fixture(scope="session", autouse=True)
def _case1_on_path() -> Iterator[None]:
    """Put the case directory on ``sys.path`` for the whole session."""
    path = str(CASE1_DIR)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)
        for name in _CASE_MODULES:
            sys.modules.pop(name, None)


@pytest.fixture
def case2() -> ModuleType:
    """The case's ``main`` module, importable as the harness would load it."""
    import main

    return main
