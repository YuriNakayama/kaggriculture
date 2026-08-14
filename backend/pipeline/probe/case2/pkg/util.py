"""Leaf module used by both ``pkg.core`` (sibling) and ``pkg.sub.deep`` (parent)."""

from __future__ import annotations


def scale(x: float, factor: float) -> float:
    return x * factor
