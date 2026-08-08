---
paths:
  - "dev/**"
---

# Command Catalog (`dev/**`)

`dev/*` are the canonical entry points. Each one `cd`s to the repo root and runs `uv` under `backend/` internally, so **they work from any directory**. Prefer them over invoking `uv` / `dvc` / `kaggle` directly — the wrappers pin the working directory and the interpreter.

## Development

| Command | What it does |
|---|---|
| `dev/setup` | `uv sync` under `backend/`. Run once after clone and after dependency changes |
| `dev/format` | `ruff format` — writes |
| `dev/lint` | `ruff check --fix` + `mypy`. Non-zero exit on any failure |
| `dev/test [args...]` | `pytest` under `backend/`. Extra args pass through (`dev/test tests/unit -x`) |

## Agent work

| Command | What it does |
|---|---|
| `dev/simulate --case <family>/<caseN> [--opponent random\|starter\|pass] [--episodes N]` | Run episodes locally via `kaggle-environments` and report final money / win rate |
| `dev/submit --case <family>/<caseN> [--dry-run] [-m MSG]` | Build the tar.gz and submit. **Always `--dry-run` first** — it unpacks to a temp dir and runs an episode, catching the nested-`main.py` and stray-import failures that would otherwise burn a submission slot |

## Kaggle data

| Command | What it does |
|---|---|
| `dev/kaggle submissions` | Submission status + IDs |
| `dev/kaggle episodes <SUBMISSION_ID>` | Games played by a submission |
| `dev/kaggle leaderboard` | Current standings |
| `dev/scrape [--top N] [--limit-per-team K] [--workers W]` | Fetch leaderboard replays + agent logs into `data/lake/kaggle_episodes/`, checkpointing to DVC as it goes |

`dev/scrape` is the local form of the `Scrape Kaggle Episodes` GitHub Actions workflow. It is **incremental**: it pulls the existing DVC state first and only fetches episodes it does not already have. It checkpoints (DVC commit + push) periodically, so a mid-run failure still persists everything fetched up to that point.

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
| `dev/runpod ...` | RunPod pod lifecycle (create / status / ssh / terminate) |

Only needed once training enters the picture; the rulebase family runs fine on CPU.

## Infrastructure

Terraform lives under `infra/`. `terraform apply` and `destroy` are **denied in `.claude/settings.json`** — infrastructure changes are a human decision. Claude may run `fmt`, `validate`, and `plan` to check correctness. See `.claude/rules/infra.md`.
