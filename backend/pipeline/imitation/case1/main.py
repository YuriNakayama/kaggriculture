"""Imitation case1 — two-layer MLP policy, behaviour-cloned from rulebase/case1.

Inference is pure numpy (see policy.NumpyPolicy): torch is a training-time
dependency only. Weights load once at import from `weights.npz` next to this
file, so the 720 per-episode calls cost one matmul pair each.

The network chooses the farmer op. Market orders stay rule-based — selling the
harvest and topping up seed stock is not a decision worth learning, and getting
it wrong silently costs the whole episode's income.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# main.py has to import its siblings under three different loaders:
#   1. locally as the package module pipeline.imitation.case1.main  -> relative
#   2. from the unpacked archive root as a top-level module         -> flat
#   3. via kaggle_environments, which exec()s the source with no __name__ in
#      globals -> the relative form raises KeyError, not ImportError
# Catching only ImportError misses case 3 and fails at submission time, so
# catch Exception and fall through to the flat import. The e2e test loads each
# case through all three paths.
try:
    from .features import extract_features
    from .policy import NumpyPolicy, action_to_op
except Exception:  # pragma: no cover - depends on the loader, see above
    # Loader 3 defines neither __name__ nor __file__, but it does append the
    # source file's directory to sys.path before exec, so the flat import
    # resolves on its own.
    from features import extract_features  # type: ignore[no-redef]
    from policy import NumpyPolicy, action_to_op  # type: ignore[no-redef]

SAFE_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

WHEAT = "WHEAT"
WHEAT_SEED_COST = 10
MAX_MARKET_ORDERS = 10

def _weights_path() -> Path:
    """Locate weights.npz next to this file.

    Loader 3 (see above) does not define __file__, so fall back to searching
    sys.path — that loader has already appended the agent's own directory.
    """
    try:
        return Path(__file__).resolve().parent / "weights.npz"
    except NameError:  # pragma: no cover - depends on the loader
        import sys

        for entry in reversed(sys.path):
            candidate = Path(entry) / "weights.npz"
            if candidate.is_file():
                return candidate
        return Path("weights.npz")


# Loaded once at import. A missing or corrupt file must not crash the episode —
# the agent degrades to the rule-based market loop with a PASS farmer.
try:
    _POLICY: NumpyPolicy | None = NumpyPolicy.load(_weights_path())
except Exception:
    _POLICY = None


def _build_market_orders(
    money: float, shed: dict[str, Any], seeds: dict[str, Any]
) -> list[list[Any]]:
    """Sell harvested wheat, then restock seed. Sales first — see rulebase."""
    orders: list[list[Any]] = []

    wheat_in_shed = int(shed.get(WHEAT, 0) or 0)
    if wheat_in_shed > 0:
        orders.append(["SELL", WHEAT, wheat_in_shed])

    if int(seeds.get(WHEAT, 0) or 0) == 0 and money >= WHEAT_SEED_COST:
        orders.append(["BUY_SEED", WHEAT, 1])

    return orders[:MAX_MARKET_ORDERS]


def agent(obs: Any) -> dict[str, Any]:
    """Entry point called once per turn by the engine."""
    try:
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private") or {}

        shed = private.get("shed") or {}
        seeds = private.get("seeds") or {}
        money = float(farm.get("money", 0.0))

        if _POLICY is None:
            farmer_op: list[Any] = ["PASS"]
        else:
            farmer_op = action_to_op(_POLICY.act(extract_features(obs)))

        return {
            "farmer": farmer_op,
            "hands": [],
            "market": _build_market_orders(money, shed, seeds),
        }
    except Exception:
        return dict(SAFE_ACTION)
