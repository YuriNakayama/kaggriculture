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

`dev/submit --dry-run` must build the archive, unpack it into a temp dir, import `main.py` from that temp dir, and run one episode — without touching the Kaggle API. Do this before every real submission; it catches the nested-`main.py` and stray-import failures that otherwise burn a submission slot.

## Submission history

Record every real submission under `data/output/submit/` (DVC-managed): the archive itself, the message, the case, the timestamp, and later the resulting score. This is the audit trail linking a leaderboard score back to the exact code that produced it.

## Quota

Kaggle enforces a per-day submission limit on this competition. Before submitting, check remaining quota via `kaggle competitions submissions kaggriculture` and refuse the submission locally if the day's budget is spent — an over-quota API call wastes time and produces a confusing error. Treat submission slots as a scarce resource: prefer a batch of local episodes over "submit and see".

## Monitoring after submit

```bash
kaggle competitions submissions kaggriculture       # status + submission IDs
kaggle competitions episodes <SUBMISSION_ID>        # games played
kaggle competitions replay <EPISODE_ID> -p data/lake/kaggle_episodes/replays
kaggle competitions logs <EPISODE_ID> 0 -p data/lake/kaggle_episodes/logs
kaggle competitions leaderboard kaggriculture -s
```

Downloaded replays and logs land under `data/lake/` (raw, DVC-managed) — see `.claude/rules/data.md`.
