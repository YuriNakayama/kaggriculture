"""Observation → fixed-length feature vector.

Shared by training and inference, so the two cannot drift apart. Kept
dependency-free (pure Python + numpy) because it runs inside the submitted
agent, where only numpy is guaranteed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Crops the policy can plant, in a fixed order. The action head indexes into
#: this list, so the order is part of the model contract — never reorder it
#: without retraining.
CROPS: tuple[str, ...] = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

#: Products whose price and shed count are fed to the model, in fixed order.
PRODUCTS: tuple[str, ...] = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

#: Base prices at the market equilibrium I0, used to normalise observed prices
#: to roughly 1.0 so no single input dominates the first layer.
BASE_PRICES: dict[str, float] = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}

FEATURE_NAMES: tuple[str, ...] = (
    "day_frac",
    "hour_frac",
    "money_norm",
    "tile_empty",
    "tile_weed",
    "tile_plant",
    "tile_watered",
    "tile_age_frac",
    "tile_yield_norm",
    *(f"seed_{c}" for c in CROPS),
    *(f"shed_{p}" for p in PRODUCTS),
    *(f"price_{p}" for p in PRODUCTS),
)

FEATURE_DIM = len(FEATURE_NAMES)

_SEASON_DAYS = 30.0
_TURNS_PER_DAY = 24.0
_MONEY_SCALE = 3000.0
_MAX_CROP_AGE = 16.0
_MAX_YIELD = 6.0


def _tile_at(farm: dict[str, Any], x: int, y: int) -> Any:
    tiles = farm.get("tiles") or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def extract_features(obs: Any) -> np.ndarray:
    """Build the feature vector for the current turn.

    Every field is accessed defensively: a missing key must degrade to a zero
    feature rather than raise inside the submitted agent.
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if player >= len(farms):
        return feats
    farm = farms[player]
    private = obs.get("private") or {}

    i = 0
    feats[i] = float(obs.get("day", 0) or 0) / _SEASON_DAYS
    i += 1
    feats[i] = float(obs.get("hour", 0) or 0) / _TURNS_PER_DAY
    i += 1
    feats[i] = float(farm.get("money", 0.0) or 0.0) / _MONEY_SCALE
    i += 1

    # State of the tile the farmer is standing on. This agent, like the
    # rulebase baseline it learns from, only ever acts on its own tile.
    fx, fy = farm.get("farmer", (0, 0))
    tile = _tile_at(farm, int(fx), int(fy))
    day = float(obs.get("day", 0) or 0)

    if tile is None:
        feats[i] = 1.0  # tile_empty
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        feats[i + 1] = 1.0  # tile_weed
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
        feats[i + 2] = 1.0  # tile_plant
        feats[i + 3] = 1.0 if tile.get("watered_today") else 0.0
        age = day - float(tile.get("planted_day", day) or day)
        feats[i + 4] = min(age, _MAX_CROP_AGE) / _MAX_CROP_AGE
        feats[i + 5] = (
            min(float(tile.get("yield_units", 0) or 0), _MAX_YIELD) / _MAX_YIELD
        )
    i += 6

    seeds = private.get("seeds") or {}
    for crop in CROPS:
        feats[i] = min(float(seeds.get(crop, 0) or 0), 5.0) / 5.0
        i += 1

    shed = private.get("shed") or {}
    for product in PRODUCTS:
        feats[i] = min(float(shed.get(product, 0) or 0), 20.0) / 20.0
        i += 1

    prices = (obs.get("market") or {}).get("prices") or {}
    for product in PRODUCTS:
        base = BASE_PRICES[product]
        feats[i] = float(prices.get(product, base) or base) / base
        i += 1

    return feats
