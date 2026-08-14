"""Sibling relative import (``from .util``) + ``__file__``-relative data load."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .util import scale

_WEIGHTS_PATH = Path(__file__).parent / "data" / "weights.npy"


def weights_fingerprint() -> str:
    w = np.load(_WEIGHTS_PATH)
    return f"shape={w.shape} sum={scale(float(w.sum()), 1.0):.4f}"
