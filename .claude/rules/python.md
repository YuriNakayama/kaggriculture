---
paths:
  - "**/*.py"
  - "**/*.ipynb"
---

# Python Rules

**General Python rules** for editing `.py` / `.ipynb` files in this repository. Auto-loaded across every region that contains Python code (`backend/`, `pipeline/`, tests, notebooks).

For agent-case directory structure see `.claude/rules/backend/pipeline.md`. For pytest conventions see `.claude/rules/backend/tests.md`. For Kaggle submission packaging see `.claude/rules/backend/submit.md`.

`pyproject.toml` / `uv.lock` / `.python-version` sit at `backend/` root, and `uv run ...` commands are expected to execute from `backend/`. From the repo root, prefer the `dev/*` wrappers.

## Backend Module Architecture (`backend/src/**`)

`backend/src/` holds the shared **development** libraries (these are *not* submitted to Kaggle).
Each subdirectory is exposed as a top-level package via `[tool.hatch.build.targets.wheel] packages`, so imports are bare (`from simulate import ...`, `from submit import ...`):

```
backend/src/
  submit/          Kaggle submission packaging / validation / quota (python -m submit)
  simulate/        kaggle-environments wrapper: run matches, dump replays (python -m simulate)
  dataset/         Replay → training-data pipeline (python -m dataset)
  evaluate/        Cross-case evaluation (win rate, final-money margin, aggregation)
  gpu/             GPU provider CLIs (one subpackage per provider):
    runpod/        RunPod pod control CLI (python -m gpu.runpod)
    kaggle/        Kaggle Notebook GPU training CLI (python -m gpu.kaggle)
```

`backend/pipeline/` holds the **agent families** — the code that actually gets submitted:

```
backend/pipeline/
  rulebase/case1/  Hand-written heuristic agents
  imitation/       Behaviour-cloning agents (added when the data pipeline exists)
  reinforce/       RL agents (added later)
```

## Style

- **Type hints are required** on all function signatures and module-level constants. `mypy` runs in CI with the config in `backend/pyproject.toml`.
- Use `from __future__ import annotations` at the top of modules that need forward references.
- Format and lint with Ruff — never hand-format. `dev/format` writes, `dev/lint` checks + autofixes.
- Prefer `pathlib.Path` over `os.path`.
- Prefer f-strings; no `%`-formatting or `.format()`.
- Dataclasses (or Pydantic models where validation is genuinely needed) over ad-hoc dicts for structured state.

## Logging

- Use the `logging` module, never bare `print`, in `backend/src/**`.
- **Exception**: submitted agent code under `backend/pipeline/**` should avoid logging entirely on the hot path — see the agent-robustness section below.

## Agent Code Robustness (`backend/pipeline/**`)

The submitted agent runs inside the Kaggle harness. Two properties dominate everything else:

> **The agent must never raise.** An uncaught exception forfeits the episode.
> **Every action must be cheap.** The agent is called 720 times per episode, per player.

Concretely:

- Wrap the top-level `agent(obs)` body in a `try/except Exception` that falls back to a guaranteed-legal no-op:
  ```python
  {"farmer": ["PASS"], "hands": [], "market": []}
  ```
- Never index into `obs` with `[]` on a key that might be absent — use `.get()` with a default. The observation shape is documented in `docs/competition/abstract.md`, but defensive access costs nothing.
- **Invalid actions are silent no-ops, not errors.** A typo'd op name or an out-of-range coordinate will not raise — it will quietly waste a turn. This means bugs do not surface as crashes; they surface as a lower score. Cover action construction with unit tests.
- `hands` must have exactly one entry per hired hand, in order. Sending fewer/more silently drops or ignores actions.
- Market orders beyond `maxMarketOrdersPerTurn` (default 10) are **silently discarded**. Build the order list with an explicit cap and prioritise deliberately — don't let an unbounded list-comprehension decide what gets dropped.
- No file IO, no network, no imports of anything not bundled in the submission tarball.

## Imports in submitted code

`main.py` must sit at the **root** of the submission archive and must be importable standalone. Inside `backend/pipeline/<family>/case<N>/`, use **relative imports** between the case's own modules so the same directory works both locally and after packaging. Do not import from `backend/src/**` in submitted code — those are dev-only libraries and will not be present in the tarball.

## Numeric work

- Prefer `numpy` for vectorised state featurisation. Avoid per-tile Python loops over the 10×10 board when a numpy view will do — the agent is called 720× per episode.
- Use `polars` for offline dataset work (replay parsing, aggregation), `pandas` only when a dependency forces it.
