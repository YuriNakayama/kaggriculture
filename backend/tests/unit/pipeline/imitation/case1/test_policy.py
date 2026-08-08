"""Unit tests for the imitation/case1 features and numpy policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipeline.imitation.case1.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    extract_features,
)
from pipeline.imitation.case1.policy import (
    ACTIONS,
    ACTION_DIM,
    NumpyPolicy,
    action_to_op,
)


def make_obs(
    *,
    tile: Any = None,
    seeds: dict[str, int] | None = None,
    shed: dict[str, int] | None = None,
    money: float = 3000.0,
    day: int = 0,
    prices: dict[str, int] | None = None,
) -> dict[str, Any]:
    tiles: list[list[Any]] = [[None] * 10 for _ in range(10)]
    tiles[4][4] = tile
    return {
        "player": 0,
        "day": day,
        "hour": 0,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {},
        ],
        "market": {"prices": prices or {"WHEAT": 25}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": shed or {}, "seeds": seeds or {}, "inventories": [{}]},
    }


def test_feature_names_match_dim() -> None:
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_feature_vector_shape_and_dtype() -> None:
    feats = extract_features(make_obs())
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32


def test_features_are_finite() -> None:
    """A NaN or inf silently poisons argmax into always picking slot 0."""
    feats = extract_features(
        make_obs(
            tile={"kind": "PLANT", "planted_day": 0, "watered_today": True},
            shed={"WHEAT": 999},
            seeds={"WHEAT": 99},
            money=1e9,
            day=29,
            prices={"WHEAT": 1},
        )
    )
    assert np.all(np.isfinite(feats))


def test_empty_tile_flag_set() -> None:
    feats = extract_features(make_obs(tile=None))
    assert feats[FEATURE_NAMES.index("tile_empty")] == 1.0
    assert feats[FEATURE_NAMES.index("tile_plant")] == 0.0


def test_plant_tile_flags() -> None:
    tile = {"kind": "PLANT", "planted_day": 0, "watered_today": True, "yield_units": 3}
    feats = extract_features(make_obs(tile=tile, day=2))
    assert feats[FEATURE_NAMES.index("tile_plant")] == 1.0
    assert feats[FEATURE_NAMES.index("tile_watered")] == 1.0
    assert feats[FEATURE_NAMES.index("tile_empty")] == 0.0


def test_weed_tile_flag() -> None:
    feats = extract_features(make_obs(tile={"kind": "WEED"}))
    assert feats[FEATURE_NAMES.index("tile_weed")] == 1.0


def test_price_normalised_to_one_at_base() -> None:
    feats = extract_features(make_obs(prices={"WHEAT": 25}))
    assert feats[FEATURE_NAMES.index("price_WHEAT")] == pytest.approx(1.0)


@pytest.mark.parametrize("broken", [{}, {"player": 0}, {"player": 9, "farms": [{}]}])
def test_malformed_observation_yields_zero_vector(broken: Any) -> None:
    feats = extract_features(broken)
    assert feats.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(feats))


def test_action_to_op_roundtrip() -> None:
    for i, expected in enumerate(ACTIONS):
        assert action_to_op(i) == list(expected)


@pytest.mark.parametrize("bad_index", [-1, ACTION_DIM, 999])
def test_action_to_op_out_of_range_is_safe(bad_index: int) -> None:
    assert action_to_op(bad_index) == ["PASS"]


def test_numpy_policy_forward_shape() -> None:
    rng = np.random.default_rng(0)
    policy = NumpyPolicy(
        w1=rng.normal(size=(FEATURE_DIM, 8)).astype(np.float32),
        b1=np.zeros(8, dtype=np.float32),
        w2=rng.normal(size=(8, ACTION_DIM)).astype(np.float32),
        b2=np.zeros(ACTION_DIM, dtype=np.float32),
    )
    logits = policy.logits(extract_features(make_obs()))
    assert logits.shape == (ACTION_DIM,)
    assert 0 <= policy.act(extract_features(make_obs())) < ACTION_DIM
