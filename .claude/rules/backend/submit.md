---
paths:
  - "backend/src/submit/**"
  - "dev/submit"
---

# Kaggle Submission Rules (`backend/src/submit/**`, `dev/submit`)

Packaging and submitting an agent to the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) competition.

## Archive contract

The submission is either a bare `main.py` or a `tar.gz`. In both cases:

> **`main.py` must be at the ROOT of the archive**, not nested inside a directory.
> It must define a module-level `agent(obs)` function.

```bash
# single file
kaggle competitions submit kaggriculture -f main.py -m "message"

# multi-file — note the files are added at the archive root, not the parent dir
tar -czf submission.tar.gz -C backend/pipeline/rulebase/case1 .
kaggle competitions submit kaggriculture -f submission.tar.gz -m "message"
```

The `-C <dir> .` form is what keeps `main.py` at the root. `tar -czf out.tar.gz backend/pipeline/...` would nest it and the harness would fail to import.

## Measured harness constraints (probe/case1 + probe/case2, 2026-08-13)

Ground truth from the real validation image (submissions 55481573 / 55481654);
`verify_archive` mirrors these exactly:

| Constraint | Measured value | Consequence for cases |
|---|---|---|
| Python | **3.11.13** (local dev: 3.13) | No 3.12+-only syntax in submitted code |
| `main.py` loading | `exec` with **no** `__name__` / `__file__` / `__package__`; the **last callable** defined becomes the agent | No relative imports and no `__file__` in `main.py`; keep `agent` as the final definition |
| cwd | `/kaggle/working` — **not** the agent dir | Bare relative `open()` fails; load data via a *package module's* `__file__` |
| Agent dir | `/kaggle_simulations/agent`, appended to `sys.path` | Top-level siblings import by bare name; subpackages by `import pkg.mod` |
| Hierarchy | Subpackages work, incl. relative imports inside them and namespace packages (`__init__.py` is stripped by `build_archive`) | Hierarchical cases are fine; don't rely on `__init__.py` side effects |
| Libraries | numpy 2.4.6, polars, pandas, scipy, torch 2.6.0+cu124, kaggle_environments 1.32.6 | numpy/torch usable at inference; still keep archives lean |

## Rules a case must follow to be submittable

The single authoritative list of constraints on `backend/pipeline/<family>/case<N>/` code, derived from the measurements above (this section overrides any older import guidance elsewhere):

- **Python 3.11 compatible** — local dev is 3.13; no 3.12+-only syntax in submitted code. `dev/submit --dry-run` verifies under a real 3.11 interpreter.
- **`main.py`: no relative imports, no `__file__`.** It is `exec`'d with no package context. Import top-level sibling modules by bare name (`from tasks import assign`), subpackages absolutely (`from pkg.core import f`).
- **`agent` must be the LAST callable defined in `main.py`** — the harness takes the last one.
- **Subpackages (`pkg/…`) are allowed.** Relative imports (`from .util`, `from ..util`) work *inside* them, and their modules do have `__file__`.
- **Data files (weights etc.) load via `Path(__file__).parent` of a subpackage module** — never from `main.py` (no `__file__`) and never with a bare relative `open()` (cwd is not the agent dir).
- **`__init__.py` never ships** (stripped by `EXCLUDE_NAMES`; subpackages run as namespace packages) — don't put code in it. Same for `train.py`: training-only code, never imported at inference.
- **No imports from `backend/src/**`** — dev-only libraries, absent from the tarball.

## Pre-submit checklist

Enforce these in `backend/src/submit/` rather than relying on discipline:

- [ ] `main.py` exists at archive root and defines `agent`
- [ ] The archive imports cleanly in a **fresh interpreter with only `kaggle-environments` installed** — no `backend/src/**` imports, no dev-only dependencies
- [ ] A full 720-turn episode completes against `random` **and** `starter` without raising
- [ ] No absolute paths; any bundled weights load relative to `main.py`
- [ ] No credentials, no `.env`, no `data/` contents in the archive

## Dry run first

```bash
dev/submit --case rulebase/case1 --dry-run
```

This builds the archive, unpacks it into a temp dir, and replays the measured harness conditions in an **isolated Python 3.11 interpreter** (provisioned by uv, only `kaggle-environments` installed): `exec`s `main.py` with no `__name__`/`__file__`, asserts `agent` is the last callable defined, uses a cwd that is *not* the agent dir, and runs a full 720-turn season against `starter`. No Kaggle API call.

Do this before every real submission. It catches the nested-`main.py`, stray-`backend/src`-import, and relative-import failures that otherwise burn a slot.

`train.py` is **excluded from the archive** by `EXCLUDE_NAMES`: it imports torch, a training-only dependency. A module needed at inference must not be named `train.py`.

## Submission history

Every real submission is recorded under `data/output/submit/` (DVC-managed) with the case, message, **git sha + dirty flag**, timestamp, archive size, member list, and verification result. This is the audit trail linking a leaderboard score back to the exact code that produced it — **commit before submitting**, or the recorded sha will not reproduce the archive (the CLI warns on a dirty tree).

## Quota

**5 submissions/day**, and only the latest are scored on the ladder. `dev/submit` counts today's entries in `data/output/submit/` and refuses at the cap (`--force` overrides). Kaggle enforces the real limit; this is a local guard so an over-quota attempt fails immediately instead of producing a confusing API error.

Treat slots as scarce: prefer a batch of local episodes (`dev/simulate --episodes 50`) over "submit and see".

## Monitoring after submit

```bash
kaggle competitions submissions kaggriculture       # status + submission IDs
kaggle competitions episodes <SUBMISSION_ID>        # games played
kaggle competitions replay <EPISODE_ID> -p data/lake/kaggle_episodes/replays
kaggle competitions logs <EPISODE_ID> 0    # your own submissions only
kaggle competitions leaderboard kaggriculture -s
```

Downloaded replays land under `data/lake/` (raw, DVC-managed) — see `.claude/rules/data.md`.

Agent logs are available only for **your own** submissions, and only via the CLI above; `dev/scrape` does not collect them.
