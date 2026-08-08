"""Local match execution against the Kaggriculture environment.

Thin wrapper over ``kaggle_environments`` so every caller (dev/simulate, tests,
evaluation) runs episodes the same way and gets the same result shape.
"""

from __future__ import annotations

from .runner import EpisodeResult, MatchSummary, run_episode, run_match

__all__ = ["EpisodeResult", "MatchSummary", "run_episode", "run_match"]
