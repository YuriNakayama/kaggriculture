# Kaggle Kaggriculture: Farming-Sim Competition Agents

Project for the **[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)** simulation competition on Kaggle ($50,000, deadline **2026-09-30**). We develop an agent that plays a two-player **farming-sim** on the `kaggle-environments` engine and climbs the leaderboard. See [`docs/competition/abstract.md`](../docs/competition/abstract.md) for the full spec summary; the verbatim official rules are in [`docs/competition/README.md`](../docs/competition/README.md).

Two players each run a farm for a **720-turn season** (24 turns/day × 30 days), buying seeds and livestock, planting, watering, harvesting, tending animals, hiring hands, and trading on a **dynamic market**. Whoever has the most money in the bank at the end wins.

The core difficulty is **not** the farming — it's the market. Prices move as a function of inventory, with a different shape function above and below the equilibrium `I0`. Premium goods (strawberry, melon, milk, wool) have `above_target > 1`, so a modest glut drives them straight to the $1 floor. Both players sell into the same book concurrently. So the real problem is a joint **production-scheduling and market-timing** problem under partial information about the opponent's stock.

## Agent Pipeline

The engine calls one entry point, `agent(obs) -> dict`, once per turn:

1. **Observation** (`obs`): `day` / `hour`, both players' farms (public), the shared market book, unlocked town shops, and your own private shed / seeds / unit inventories.
2. **State featurisation**: parse the 10×10 tile grid, unit positions, shed contents, and market prices.
3. **Policy**: decide one op per unit (farmer + each hired hand) and an ordered list of market orders.
4. **Action assembly**: return the dict below.

```py
{
  "farmer": [op, *args],          # exactly one op
  "hands":  [[op, *args], ...],   # exactly one per hired hand, in order
  "market": [[op, *args], ...],   # ordered; capped at maxMarketOrdersPerTurn (default 10)
}
```

> The agent must **never raise** — an uncaught exception forfeits the episode. Wrap the top level and fall back to `{"farmer": ["PASS"], "hands": [], "market": []}`.
>
> **Invalid actions are silent no-ops, not errors.** A bad op name, an out-of-range target, or the 11th market order this turn is discarded without any signal. Bugs therefore surface as a *low score*, never as a crash — so local verification and unit tests on action construction carry the weight that exceptions normally would.

## Technology Stack

- **Language**: Python 3.13
- **Engine**: `kaggle-environments` (`make("kaggriculture")`)
- **Data**: numpy / polars, DVC + S3 for replays and datasets
- **Testing**: Pytest, Ruff, Mypy
- **Package management**: uv
- **Infra**: Terraform (S3 DVC remote + GitHub Actions OIDC role)

## Folder Structure

```
backend/                Python implementation (pyproject.toml / uv.lock live here)
  src/                  Dev-only shared libs — NEVER imported by submitted code
    submit/             Submission packaging / validation / quota (python -m submit)
    simulate/           kaggle-environments wrapper, replay dumping (python -m simulate)
    dataset/            Replay → training data, Kaggle scraping (python -m dataset)
    evaluate/           Cross-case evaluation
    gpu/common/         Provider-agnostic: credentials, run metadata
    gpu/runpod/         RunPod pod control (python -m gpu.runpod)
    gpu/kaggle/         Kaggle Kernel GPU training (python -m gpu.kaggle)
  pipeline/             Agent families — this IS the submitted code
    rulebase/case1/     Wheat loop (hand-written heuristic)
    imitation/case1/    2-layer MLP, behaviour-cloned; numpy-only at inference
  tests/                Pytest (unit / integration / e2e)
infra/                  Terraform (S3 DVC remote, GitHub Actions OIDC)
data/                   4 layers (lake / processed / mart / output) — DVC-managed, gitignored
dev/                    Command wrappers (each cd's to root and runs uv under backend/)
.github/workflows/      ci.yml (ruff/mypy/pytest) + scrape-kaggle.yml (daily replay fetch)
docs/
  competition/          Official rules (README.md / AGENTS.md) + abstract.md summary
  experiment/           Experiment plans and results — see .claude/rules/docs.md
```

`uv run ...` is expected to run under `backend/`. From the repo root, use `dev/*`. See [`.claude/rules/command.md`](rules/command.md) for the full command catalog.

## Submission

`main.py` must sit at the **root** of the archive (not nested) and define `agent(obs)`. Build from a `backend/pipeline/<family>/case<N>/` directory with `tar -czf out.tar.gz -C <case-dir> .`. Submitted code must not import from `backend/src/**` — those packages are not in the tarball. **Always `dev/submit --dry-run` first**; it unpacks to a temp dir and runs an episode, catching the two failures that otherwise burn a submission slot.

## Glossary

| Term | Description |
|------|-------------|
| Quadrant | One of four 5×5 sections of the 10×10 board. Only NW starts unlocked; NE/SW/SE cost $1k/$2k/$4k via `BUY_LAND` |
| Shed | Central inventory, capacity 100 non-seed items. Not a tile — "adjacent" means standing on `(4,4)`, `(5,4)`, `(4,5)`, or `(5,5)`. Overflow at end-of-day is **discarded** |
| Farm hand | Day-labourer hired via `HIRE`; cost `fib(n)` for the n-th hire that day, resets daily, disappears at day end |
| Weed | Spawns on empty tiles (`weedSpawnChance`, default 0.005/day) or from a neglected plant. Cleared with `DIG` |
| `I0` | Market equilibrium inventory, 10,000 for every product. Price equals `base` here |
| `T` | Anchor throughput — one 5×5 field's 24-day output. `target` means "moving `T` past `I0` shifts price by `target × base`" |
| `pending_care_bonus` | Yield banked by `CARE` on fed days, paid out in full on the next scheduled production |
| Town shop | Unlocks every 3 days (random), consumes its demanded products every 4 turns forever after. Demand grows monotonically |
| `starter` | Built-in deterministic baseline agent; the bar a new case must clear |

## Win Conditions

Most money in the bank at the end of the season. Ties are possible. **Unsold inventory does not count** — produce sitting in the shed at turn 720 is worth zero, so the season ends with a liquidation problem, not a production one.

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/python.md` | `**/*.py`, `**/*.ipynb` | Module architecture, agent robustness contract |
| `.claude/rules/backend/pipeline.md` | `backend/pipeline/**` | Family/case layout, what may be imported in submitted code |
| `.claude/rules/backend/submit.md` | `backend/src/submit/**`, `dev/submit` | Archive contract, pre-submit checklist, quota |
| `.claude/rules/backend/tests.md` | `backend/tests/**` | unit / integration / e2e classification |
| `.claude/rules/data.md` | `data/**` | 4-layer structure, DVC conventions |
| `.claude/rules/infra.md` | `infra/**` | Terraform, OIDC, required repo secrets |
| `.claude/rules/command.md` | `dev/**` | Command catalog |
| `.claude/rules/docs.md` | `docs/**` | Experiment doc naming |
| `.claude/rules/security.md` | Always loaded | Commits, secrets, CI/CD |

## This Repository Is Public

`github.com/YuriNakayama/kaggriculture` is a **public** repository. Before any commit, verify no `terraform.tfstate`, `terraform.tfvars`, `.env`, `kaggle.json`, AWS account IDs, or `data/` contents are staged. A leaked secret here is a rotate-and-rewrite-history incident, not a follow-up fix.

## Response Language And Interface

- Answer user questions concisely, organizing the response as a table, chart, list, short sentence, ASCII art, or similar structured format.
- Keep user-facing replies under 800 characters, excluding tables, charts, code blocks, and ASCII art (which can exceed the limit when needed).
- Use the `AskUserQuestion` tool when asking questions to the user
- Internal reasoning, tool calls, and intermediate notes: English.
- User-facing output (final replies, reports, summaries): Japanese.(全てのユーザー向けの出力は日本語で行うこと)
