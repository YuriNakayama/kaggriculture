---
paths:
  - "backend/pipeline/**"
---

# Agent Pipeline Rules (`backend/pipeline/**`)

`backend/pipeline/` holds the **agent families** — the only code that gets submitted to Kaggle. Everything under `backend/src/` is dev-only tooling and is never bundled.

## Family / case layout

```
backend/pipeline/
  <family>/            rulebase | imitation | reinforce
    case<N>/           one submittable agent variant
      main.py          REQUIRED — defines `agent(obs)`; lands at the archive root
      <module>.py      supporting modules, imported relatively from main.py
      README.md        what this case does and what changed vs the previous case
```

- **family** = the approach. `rulebase` (hand-written heuristics), `imitation` (behaviour cloning from replays), `reinforce` (RL).
- **case** = one concrete, submittable variant within a family. Cases are append-only: `case1`, `case2`, … A case is a historical record of what was submitted, so **do not rewrite an old case in place** once it has been submitted — create the next case instead.

## Rules for a case directory

- `main.py` **must** define a module-level `agent(obs)` function. The Kaggle harness imports it from the archive root.
- Use **relative imports** (`from .policy import choose_action`) between the case's own modules so the directory works both in-place and after packaging.
- **Never import from `backend/src/**`** — those packages do not exist inside the submission tarball.
- Keep the case self-contained: any model weights must live inside the case directory and be loaded with a path relative to `main.py`.
- Each case's `README.md` records the hypothesis it tests and the observed result, and links to the corresponding `docs/experiment/` entry.

## Agent contract

See `.claude/rules/python.md` for the full robustness rules. The two that dominate:

> The agent must never raise — wrap the top level and fall back to `{"farmer": ["PASS"], "hands": [], "market": []}`.
> Invalid actions are **silent no-ops**, so bugs show up as a low score, not a crash. Unit-test action construction.

Return shape:

```py
{
  "farmer": [op, *args],          # exactly one op
  "hands":  [[op, *args], ...],   # exactly one per hired hand, in order
  "market": [[op, *args], ...],   # capped at maxMarketOrdersPerTurn (default 10)
}
```

Market orders past the cap are dropped **silently and from the tail**, so order the list by priority deliberately rather than letting it be incidental.

## Local verification before submitting

```bash
dev/simulate --case rulebase/case1 --opponent random
dev/simulate --case rulebase/case1 --opponent starter --episodes 10
```

A case is not submittable until it completes a full 720-turn season against both `random` and `starter` without raising.
