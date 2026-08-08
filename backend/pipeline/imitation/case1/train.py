"""Behaviour-cloning trainer for imitation/case1.

Collects (observation, action) pairs by watching the rulebase/case1 agent play,
fits the two-layer MLP to reproduce its decisions, and exports numpy weights
that `main.py` loads at inference time.

    uv run python -m pipeline.imitation.case1.train --episodes 8 --epochs 30

Cloning a hand-written baseline is not meant to beat it — the ceiling is the
teacher. It exists to prove the whole train → export → run path works before
any real learning signal (self-play, Kaggle replays) is wired in.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import extract_features
from .policy import ACTIONS, build_torch_model, export_torch_weights

CASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = CASE_DIR / "weights.npz"

#: Map a teacher op back to its action index.
_OP_TO_INDEX: dict[tuple[str, ...], int] = {op: i for i, op in enumerate(ACTIONS)}


def _teacher_action_index(op: list[Any]) -> int | None:
    """Which action slot did the teacher pick? None if outside our action set."""
    return _OP_TO_INDEX.get(tuple(str(part) for part in op))


def collect(episodes: int, steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the rulebase teacher and record its decisions."""
    import sys

    from kaggle_environments import make

    # The teacher lives in a sibling case directory; import it by path so this
    # module does not depend on package layout.
    sys.path.insert(0, str(CASE_DIR.parents[1] / "rulebase" / "case1"))
    import main as teacher

    features: list[np.ndarray] = []
    labels: list[int] = []

    for episode in range(episodes):

        def recording_agent(obs: Any) -> dict[str, Any]:
            action: dict[str, Any] = teacher.agent(obs)
            index = _teacher_action_index(action["farmer"])
            if index is not None:
                features.append(extract_features(obs))
                labels.append(index)
            return action

        env = make(
            "kaggriculture",
            configuration={"episodeSteps": steps, "seed": seed + episode},
            debug=True,
        )
        env.run([recording_agent, "random"])

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Any:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    model = build_torch_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The teacher PASSes far more often than it acts, so an unweighted fit
    # collapses to always-PASS. Weight each class by its inverse frequency.
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    n = len(x_t)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            optimiser.zero_grad()
            loss = criterion(model(x_t[idx]), y_t[idx])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(idx)

        if epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                accuracy = float((model(x_t).argmax(1) == y_t).float().mean())
            print(f"  epoch {epoch:3d}  loss {total_loss / n:.4f}  acc {accuracy:.3f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    started = time.monotonic()

    print(f"== collecting {args.episodes} episodes from rulebase/case1 ==")
    x, y = collect(args.episodes, args.steps, args.seed)
    distribution = {
        "/".join(ACTIONS[i]): int(c)
        for i, c in enumerate(np.bincount(y, minlength=len(ACTIONS)))
    }
    print(f"  samples: {len(x)}  feature dim: {x.shape[1]}")
    print(f"  actions: {distribution}")

    print("== training ==")
    model = train(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_torch_weights(model, args.out)
    print(f"== wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB) ==")
    print(f"   total {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
