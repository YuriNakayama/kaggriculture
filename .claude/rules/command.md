---
paths:
  - "dev/**"
---

# Command Catalog (`dev/**`)

`dev/*` are the canonical entry points. Each one `cd`s to the repo root and runs `uv` under `backend/` internally, so **they work from any directory**. Prefer them over invoking `uv` / `dvc` / `kaggle` directly — the wrappers pin the working directory and the interpreter.

## Development

| Command | What it does |
|---|---|
| `dev/setup` | `uv sync` under `backend/`. Run once after clone and after dependency changes. `--nn` adds torch, `--gpu` adds the RunPod SDK, `--all` adds both |
| `dev/format` | `ruff format` — writes |
| `dev/lint` | `ruff check --fix` + `mypy`. Non-zero exit on any failure |
| `dev/test [args...]` | `pytest` under `backend/`. Extra args pass through (`dev/test tests/unit -x`) |

## Agent work

| Command | What it does |
|---|---|
| `dev/simulate --case <family>/<caseN> [--opponent random\|starter\|pass] [--episodes N]` | Run episodes locally via `kaggle-environments` and report final money / win rate |
| `dev/submit --case <family>/<caseN> --dry-run` | Build + verify only. **Always run this first.** |
| `dev/submit --case <family>/<caseN> -m MSG` | Build, verify, submit, and record to `data/output/submit/`. Refuses at 5/day |

## Kaggle data

| Command | What it does |
|---|---|
| `dev/kaggle submissions kaggriculture` | Submission status + IDs |
| `dev/kaggle episodes <SUBMISSION_ID>` | Games played by a submission |
| `dev/kaggle leaderboard kaggriculture -s` | Current standings |
| `dev/scrape [--top N] [--limit-per-team K] [--limit N]` | Fetch top-team match replays into `data/lake/kaggle_episodes/replays/`, checkpointing to DVC as it goes |

`dev/scrape` is the local form of the `Scrape Kaggle Episodes` GitHub Actions workflow. It is **incremental**: it pulls the existing DVC state first and only fetches episodes it does not already have. It checkpoints (DVC commit + push) periodically, so a mid-run failure still persists everything fetched up to that point.

Two things worth knowing, both verified against the live API:

- **Replays are ~29 MB each.** Defaults are deliberately small (top 20 teams, 5 episodes each). Raise them with the S3 bill in mind.
- **Replays are the training signal.** Each one holds every turn's action, observation, and reward for both players. Agent *logs* (stderr) are not fetched: Kaggle 403s them for any submission you do not own, and they contain no gameplay data.

## Worktrees

| Command | What it does |
|---|---|
| `dev/create-worktree <branch>` | Create `<repo>.worktrees/<branch>/`, symlink `data/` to the main repo, share the DVC cache, copy `.env` files |
| `dev/delete-worktree [<branch>]` | Remove a worktree (no arg = the current one). Warns on uncommitted/unpushed work, offers to delete the branch |
| `dev/sync-data` | Merge a worktree's real `data/` back into the main repo and replace it with a symlink |

**Always use `dev/create-worktree`, not `git worktree add`.** A bare `git worktree add`
gives the new tree its own `data/` and its own empty DVC cache, so `dvc pull`
re-downloads everything from S3 and the replays end up duplicated on disk.

The wrapper also marks the tracked `data/**` stubs `skip-worktree`. Replacing that
directory with a symlink otherwise makes git see them as deleted — including
`replays.dvc`, the pointer to everything in S3 — and a `git commit -a` would drop it.

## Data (DVC)

| Command | What it does |
|---|---|
| `dev/dvc setup` | One-time per machine: shared cache dir + AWS profile into `.dvc/config.local` (gitignored) |
| `dev/dvc pull [targets]` | Fetch real data from S3 |
| `dev/dvc push [targets]` | Upload to S3 |
| `dev/dvc status --cloud` | Is my local cache in sync with the remote? |
| `dev/dvc add <path>` | Start tracking a directory |
| `dev/dvc <anything>` | Pass-through to `dvc` |

**Do not run two `dvc push` / `dvc repro` concurrently** — they contend on the cache lock. Reads (`pull`, `status`) are fine in parallel.

## GPU

| Command | What it does |
|---|---|
| `dev/runpod ...` | RunPod pod lifecycle (train / dev / ps / stock / pull / promote / cost-report) |

Needs the optional GPU deps: `dev/setup --gpu`. Trainable cases are registered in
`backend/src/gpu/runpod/config/cases.py`; add an entry there before launching a new one.

Not needed yet — the rulebase family is pure CPU, and `imitation/case1` trains
in about 20 seconds on a laptop.

## Infrastructure

Terraform lives under `infra/`. `terraform apply` and `destroy` are **denied in `.claude/settings.json`** — infrastructure changes are a human decision. Claude may run `fmt`, `validate`, and `plan` to check correctness. See `.claude/rules/infra.md`.
