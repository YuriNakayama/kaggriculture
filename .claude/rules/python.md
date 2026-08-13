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
  simulate/        kaggle-environments wrapper: run matches, aggregate results,
                   dump replays (python -m simulate)
  dataset/         Replay → training-data pipeline (python -m dataset)
  gpu/             GPU provider CLIs (one subpackage per provider):
    runpod/        RunPod pod control CLI (python -m gpu.runpod)
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

Measured on the real harness (probe/case1+2, 2026-08-13 — see `.claude/rules/backend/submit.md`): `main.py` is `exec`'d with **no `__name__` / `__file__` / `__package__`**, the agent dir is appended to `sys.path`, and cwd is elsewhere. Production Python is **3.11** (local dev is 3.13 — avoid 3.12+ syntax in `backend/pipeline/**`). Therefore:

- **In `main.py`**: no relative imports (`from .x import` fails — no package context), no `__file__`. Import top-level sibling modules by bare name (`from tasks import assign`), subpackages absolutely (`from pkg.core import f`). Keep `agent` as the **last** callable defined — the harness takes the last one.
- **Inside a subpackage** (`pkg/…`): relative imports (`from .util import`, `from ..util import`) work, and `__file__` is available — load bundled data files with `Path(__file__).parent / …` from a package module, never from `main.py` and never with a bare relative `open()`.
- `__init__.py` is stripped from the archive (`EXCLUDE_NAMES`), so subpackages run as namespace packages — measured to work; don't put code in `__init__.py`.
- Do not import from `backend/src/**` in submitted code — those are dev-only libraries and will not be present in the tarball.

## Numeric work

- Prefer `numpy` for vectorised state featurisation. Avoid per-tile Python loops over the 10×10 board when a numpy view will do — the agent is called 720× per episode.
- Use `polars` for offline dataset work (replay parsing, aggregation), `pandas` only when a dependency forces it.
