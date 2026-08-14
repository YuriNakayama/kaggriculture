"""Nested subpackage using a parent-relative import (``from ..util``)."""

from __future__ import annotations

from ..util import scale


def doubled(x: float) -> float:
    return scale(x, 2.0)
