"""Generate a reference trace from the authoritative Python engine.

The trace drives frontend/playable/src/engine/__tests__/parity.spec.ts, which
replays the same action sequence through the TS engine port and compares
per-step observable state. Any divergence — an upstream engine change, a
porting bug — fails that spec.

Run from backend/ (so kaggle_environments resolves):

    uv run python ../frontend/scripts/gen_parity_trace.py \
        --steps 240 --seed 123 --out ../frontend/playable/parity-trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--agents", nargs=2, default=["starter", "starter"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": args.steps, "seed": args.seed},
        debug=True,
    )
    env.run(args.agents)

    steps = env.toJSON()["steps"]
    trace: dict = {
        "seed": args.seed,
        "episodeSteps": args.steps,
        "agents": args.agents,
        "config": dict(env.configuration),
        "steps": [],
    }
    for t, agent_states in enumerate(steps):
        shared = agent_states[0]["observation"]
        entry = {
            "t": t,
            # Action taken to produce this state (absent at t=0).
            "actions": [a.get("action") for a in agent_states] if t > 0 else None,
            "day": shared["day"],
            "hour": shared["hour"],
            "money": [f["money"] for f in shared["farms"]],
            "farmer": [f["farmer"] for f in shared["farms"]],
            "num_hands": [len(f["hands"]) for f in shared["farms"]],
            "market_inventory": shared["market"]["inventory"],
        }
        trace["steps"].append(entry)
    trace["rewards"] = env.toJSON()["rewards"]

    args.out.write_text(json.dumps(trace))
    print(f"parity trace: {len(steps)} steps -> {args.out}")


if __name__ == "__main__":
    main()
