"""Training case registry and path helpers for RunPod launches.

Each entry names one trainable case and tells the pod how to run it: which
DVC stage it corresponds to, which module to invoke, and where the resulting
weights belong. Registry keys do not have to equal the on-disk directory —
see :func:`case_subdir` for how variants of a case share one output tree.

Add an entry here when a case becomes worth training on a GPU. Cases that
train fine on CPU (the whole `rulebase` family, and `imitation/case1` — a
two-layer MLP that fits in seconds) do not need one.
"""

from __future__ import annotations

from pathlib import Path

import typer

#: Registry of GPU-trainable cases.
#:
#: Keys:
#:   family            - pipeline family directory (default "imitation")
#:   stage             - DVC stage name for this training run
#:   train_module      - python -m target, run from backend/
#:   config_arg        - extra args appended to the train invocation
#:   preprocess_cmd    - python -m target run before training; empty means the
#:                       mart is supplied by `dvc pull` and needs no pod-side
#:                       preprocessing (which also avoids the stall watcher)
#:   canonical_weights - repo path the trained weights are copied back to
CASE_DEFAULTS: dict[str, dict[str, str]] = {
    # imitation/case1 is behaviour cloning onto a 64-unit MLP; it trains in
    # ~20s on a laptop CPU. Registered so the RunPod path has a smoke-test
    # target that does not require a real GPU workload to exist yet.
    "case1": {
        "family": "imitation",
        "stage": "train_imitation_case1",
        "train_module": "pipeline.imitation.case1.train",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "backend/pipeline/imitation/case1/weights.npz",
    },
}


def case_subdir(case: str) -> str:
    """Map a registry key to its on-disk case subdirectory.

    Variants of one case (different head modes, sweep points, ...) are
    registered under separate keys but share a single output tree, so the
    `_<variant>` suffix is stripped here. Reinforce and bench keys carry a
    prefix that is likewise not part of the directory name.
    """
    if case.startswith("reinforce_"):
        return case[len("reinforce_") :]
    if case.startswith("bench_"):
        return case[len("bench_") :]
    # Variant suffixes: caseN_<variant> -> caseN. Only applied when the part
    # before the underscore is itself a caseN token, so unrelated keys are
    # left alone.
    head, _, tail = case.partition("_")
    if tail and head.startswith("case") and head[4:].isdigit():
        return head
    return case


def case_family(case: str) -> str:
    """Return the top-level pipeline family directory for a registry key."""
    return CASE_DEFAULTS.get(case, {}).get("family", "imitation")


def runs_root_for(case: str) -> Path:
    return Path(f"data/output/models/{case_family(case)}/{case_subdir(case)}/runs")


def case_defaults(case: str) -> dict[str, str]:
    if case not in CASE_DEFAULTS:
        raise typer.BadParameter(
            f"unknown case={case!r}; supported: {sorted(CASE_DEFAULTS)}"
        )
    return CASE_DEFAULTS[case]


__all__ = [
    "CASE_DEFAULTS",
    "case_defaults",
    "case_family",
    "case_subdir",
    "runs_root_for",
]
