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

This builds the archive, unpacks it into a temp dir, and in a **separate interpreter** with only that dir importable: imports `main.py` flat, asserts `agent` is the last callable defined (the harness takes the last one), and runs a full 720-turn season against `starter`. No Kaggle API call.

Do this before every real submission. It catches the nested-`main.py`, stray-`backend/src`-import, and relative-import failures that otherwise burn a slot.

`train.py` is **excluded from the archive** by `EXCLUDE_NAMES`: it imports torch, a training-only dependency. A module needed at inference must not be named `train.py`.

## Submission history

Every real submission is recorded under `data/output/submit/` (DVC-managed) with the case, message, **git sha**, timestamp, archive size, member list, and verification result. This is the audit trail linking a leaderboard score back to the exact code that produced it.

## Quota

**5 submissions/day**, and only the latest are scored on the ladder. `dev/submit` counts today's entries in `data/output/submit/` and refuses at the cap (`--force` overrides). Kaggle enforces the real limit; this is a local guard so an over-quota attempt fails immediately instead of producing a confusing API error.

Treat slots as scarce: prefer a batch of local episodes (`dev/simulate --episodes 50`) over "submit and see".

## Monitoring after submit

```bash
kaggle competitions submissions kaggriculture       # status + submission IDs
kaggle competitions episodes <SUBMISSION_ID>        # games played
kaggle competitions replay <EPISODE_ID> -p data/lake/kaggle_episodes/replays
kaggle competitions logs <EPISODE_ID> 0 -p data/lake/kaggle_episodes/logs
kaggle competitions leaderboard kaggriculture -s
```

Downloaded replays and logs land under `data/lake/` (raw, DVC-managed) — see `.claude/rules/data.md`.
