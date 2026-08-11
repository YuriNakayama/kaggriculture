"""rulebase/case2 -- zone-partitioned wheat + melon + dairy farm.

Entry point for the Kaggle harness. ``agent(obs)`` must be the last callable
defined in this module and must never raise: an uncaught exception forfeits the
episode, so the whole body is wrapped and falls back to a legal no-op.

The Kaggle harness ``exec``s this file with neither ``__name__`` nor
``__file__`` in globals, so relative imports are impossible; it appends this
file's directory to ``sys.path`` first, so sibling modules are imported by bare
name. That works both in-place and flat at the archive root after
``tar -C <case-dir> .``.

The harness also takes the *last* callable defined in the module as the agent,
so ``agent`` must stay the final definition in this file.
"""

from __future__ import annotations

from typing import Any

from config import MAX_MARKET_ORDERS
from market import build_market
from observe import parse
from tasks import assign, build_tasks

#: Guaranteed-legal action used whenever anything goes wrong.
SAFE_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def _fallback(n_hands: int) -> dict[str, Any]:
    """A no-op action with the correct number of hand entries."""
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(n_hands)],
        "market": [],
    }


def _decide(obs: Any) -> dict[str, Any]:
    """Build one turn's action. May raise; ``agent`` is the guard."""
    snap = parse(obs)
    unit_actions = assign(snap, build_tasks(snap))

    farmer = unit_actions[0] if unit_actions else ["PASS"]
    # `hands` must hold exactly one entry per hired hand, in order: a shorter or
    # longer list silently drops or ignores actions.
    hands = unit_actions[1 : 1 + len(snap.hands)]
    while len(hands) < len(snap.hands):
        hands.append(["PASS"])

    return {
        "farmer": farmer,
        "hands": hands,
        "market": build_market(snap)[:MAX_MARKET_ORDERS],
    }


def agent(obs: Any) -> dict[str, Any]:
    """Kaggriculture agent entry point."""
    try:
        return _decide(obs)
    except Exception:
        try:
            farms = obs.get("farms", []) or []
            player = int(obs.get("player", 0))
            n_hands = len(farms[player].get("hands", []))
            return _fallback(n_hands)
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
