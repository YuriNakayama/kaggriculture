"""Two-layer MLP policy over a small discrete action set.

Intentionally minimal: one hidden layer, no convolutions, no sequence model.
It exists to prove the train → export → submit → run loop end to end. Anything
more expressive belongs in a later case, where its gain can be measured against
this one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Multi-loader import — see the note in main.py for why this catches Exception
# rather than ImportError.
try:
    from .features import FEATURE_DIM
except Exception:  # pragma: no cover - depends on the loader, see main.py
    from features import FEATURE_DIM  # type: ignore[no-redef]

#: The farmer ops the policy can emit, in a fixed order. The network's output
#: index maps into this tuple, so the order is part of the model contract.
ACTIONS: tuple[tuple[str, ...], ...] = (
    ("PASS",),
    ("PLANT", "WHEAT"),
    ("WATER",),
    ("HARVEST",),
    ("DIG",),
)

ACTION_DIM = len(ACTIONS)

HIDDEN_DIM = 64


def action_to_op(index: int) -> list[Any]:
    """Map a network output index to a farmer op."""
    if 0 <= index < ACTION_DIM:
        return list(ACTIONS[index])
    return ["PASS"]


class NumpyPolicy:
    """Inference-only forward pass in pure numpy.

    The submitted agent must not depend on torch — the harness environment is
    not guaranteed to have it, and importing it costs seconds we do not have
    across 720 calls. Training exports plain float32 arrays that this class
    consumes, so inference has exactly one dependency: numpy.
    """

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)

    @classmethod
    def load(cls, path: Any) -> NumpyPolicy:
        """Load weights from a .npz produced by the trainer."""
        with np.load(path) as data:
            return cls(data["w1"], data["b1"], data["w2"], data["b2"])

    def logits(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(features @ self.w1 + self.b1, 0.0)  # ReLU
        return np.asarray(hidden @ self.w2 + self.b2, dtype=np.float32)

    def act(self, features: np.ndarray) -> int:
        return int(np.argmax(self.logits(features)))


def build_torch_model(hidden_dim: int = HIDDEN_DIM) -> Any:
    """Construct the trainable torch model.

    Imported lazily so that this module stays importable inside the submitted
    agent, which has numpy but not necessarily torch.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(FEATURE_DIM, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, ACTION_DIM),
    )


def export_torch_weights(model: Any, path: Any) -> None:
    """Write a trained torch model out as the .npz the agent loads."""
    linear1, _, linear2 = model[0], model[1], model[2]
    np.savez(
        path,
        # torch Linear stores weight as (out, in); the numpy forward pass uses
        # x @ W, so transpose on the way out.
        w1=linear1.weight.detach().cpu().numpy().T.astype(np.float32),
        b1=linear1.bias.detach().cpu().numpy().astype(np.float32),
        w2=linear2.weight.detach().cpu().numpy().T.astype(np.float32),
        b2=linear2.bias.detach().cpu().numpy().astype(np.float32),
    )
