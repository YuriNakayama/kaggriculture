# rulebase/case2 — baseline from public notebooks

**Date**: 2026-08-10
**Case**: `backend/pipeline/rulebase/case2`
**Engine**: `kaggle-environments==1.32.6` (pinned this iteration)

## Goal

Stand up a multi-crop `rulebase` case. When this work started `backend/pipeline/`
held only empty `imitation` / `reinforce` stubs; `case1` (a single-crop wheat
loop) landed on `main` in parallel via PR #1, so this case was renumbered from
case1 to case2 on rebase. Target: comfortably clear `starter` and establish a
floor for later cases.

## Approach

Surveyed the 19 public notebooks under `backend/notebook/20260809/`. They split
into three groups:

| Group | Count | Content |
|---|---|---|
| Replay-tape agents | 10 | A 719-entry precomputed action table (base85 + zlib) replayed by step index, plus thin overlays. Four are byte-identical. |
| Genuine live `agent(obs)` | 5 | Real per-turn heuristics |
| Meta-analysis, no agent | 4 | Replay corpus statistics |

The replay-tape group is the actual top of the ladder, but it was **not** used:
it is not a rule-based policy, it does not generalise off the recorded
distribution, and the strongest ladder agent (Seb, #1) is only 35–66%
self-identical, i.e. genuinely adaptive.

case2 is built on the live-heuristic group: **semalytics** for zone-partitioned
workers, **dariushafshar** for CARE/animal economics, **raykkretzschmar** for the
town-shop demand correction.

## Verifying the notebooks against the engine

The engine source (1073 lines) was read directly rather than trusting notebook
claims. This mattered:

| Claim | Verdict |
|---|---|
| Market parameter table (base / T / curve shapes) | **Confirmed** exactly |
| Step 719's action never executes | **Confirmed** (`step >= episodeSteps - 2` → `DONE`) |
| Selling at the $1 floor adds no market supply | **Confirmed** (`_commit_unit`) |
| Melon-only cropping is optimal (semalytics) | **Rejected.** MELON is demanded by **0** town shops; WHEAT by 5, STRAWBERRY 4, MILK 3. raykkretzschmar's measurement — realized price is governed by shop demand, not curve steepness — beats semalytics' static-curve reasoning. Crop mix is WHEAT + MELON. |
| CARE is a percentage multiplier | **Rejected.** It banks `pending_care_bonus`, paid out only on a day the animal was also fed. |

Two engine details no notebook mentioned, both of which caused real bugs here:

- **Bought animals land in the shed**, not on the board. They must be `PICKUP`'d
  and `PLACE`'d. The first implementation left 2 cows in the shed for the entire
  season.
- **Atomic PLANT validation**: if a turn requests more seeds of a crop than are
  held, *every* PLANT request for that crop is dropped, not just the excess.

## Results

Progression as bugs were found by inspecting end-of-season state — note that
every one of these was invisible in win/loss, since all versions beat `starter`
100%:

| Version | Mean money (10 ep vs `starter`) |
|---|---|
| Initial | 22,636 |
| + herd actually placed on pastures | — |
| + crop targets scale with unlocked land | 32,096 |
| + feed-fetching as a first-class task | **35,105** |

Final validation — 15 episodes per seat against each builtin, seats swapped:

| Opponent | Seat 0 | Seat 1 | Total | Errors | Mean money |
|---|---|---|---|---|---|
| `starter` | 15/15 | 15/15 | **30/30** | 0 | 34,846 |
| `random` | 15/15 | 15/15 | **30/30** | 0 | 33,773 |
| `pass` | 15/15 | 15/15 | **30/30** | 0 | 34,564 |

**90/90 overall, zero errored episodes.** `dev/submit --dry-run` passes
(5 files, 11.6 KB, full 720-step episode in a fresh interpreter).

Seat asymmetry is real in this engine — market orders resolve in player order —
so both seats were evaluated; results are symmetric at this strength level.

### The methodological point

Win rate was 100% against `starter` from the very first run, and stayed there
through every fix. All three bugs were found by dumping the **end-of-season farm
state** — 72 empty tiles, cows sitting in the shed, animals escaping 6× per
season — not by the scoreboard. Against a weak baseline, score is nearly useless
as a debugging signal. Later cases should diff farm state, not just money.

## Testing

24 unit tests covering action-shape invariants, malformed-observation
robustness, and policy behaviour. Since invalid actions are silent no-ops, the
suite was **mutation-tested** rather than assumed adequate:

| Mutation | Caught |
|---|---|
| Remove market-order cap | ✅ |
| `assign` returns wrong number of unit actions | ✅ |
| MELON base price drift in `config.py` | ✅ |
| Liquidation starts after the last executed step | ✅ |
| Oversell beyond shed stock | ✅ |

The first two initially **survived**, because `main._decide` re-caps the market
list and pads `hands` — masking bugs in the layer beneath. Tests were added
against `tasks.assign` and `market.build_market` directly. A test asserting
`market.price_at` matches `engine.market_price` across products and inventory
levels guards the mirrored constants against engine drift.

## Repository changes beyond the case

- `kaggle-environments` pinned `>=1.17` → `==1.32.6`. Engine behaviour is
  strongly version-dependent (a published ablation gives 30.5× on 1.29.3 vs 7.1×
  on 1.32.6), so an unpinned engine makes results incomparable.
- Removed `__init__.py` from `backend/pipeline/`, `imitation/`, `reinforce/`.
  Cases are `exec`'d flat by the harness and import siblings by bare name;
  the package files made mypy resolve each module under two names.
- Added `mypy_path = "src:pipeline/rulebase/case2"` + `explicit_package_bases`.
  Without it, `ignore_missing_imports` silently resolved every intra-case import
  to `Any`, so the case was effectively untyped. Fixing this surfaced one
  pre-existing error in `gpu/runpod` (now also resolved).

> Note: `.claude/rules/backend/pipeline.md` prescribes **relative imports**
> between a case's modules. That is not possible: the harness `exec`s `main.py`
> with no `__name__`, so relative imports raise. Bare-name imports are used
> instead, and the rule should be updated.

## Next

Ranked by expected value:

1. **Scale the herd.** 2 cows vs a meta modal farm of 8 cows + 6 sheep. Needs
   the pasture and feed pipeline to scale first.
2. **Fix feed logistics properly.** Escapes are down from ~6 to 1–2 per season
   but should be 0; fetching is still reactive.
3. **Opponent-aware market timing.** `Snapshot` already carries
   `opponent_tiles` / `opponent_money` and the policy ignores both.
4. **Weed control.** ~10 tiles idle mid-season because `DIG` sits below `PLANT`.
5. **Marginal-return land purchase** instead of a fixed day-5 / day-9 schedule.

Items 1–2 are the natural content of case3.
