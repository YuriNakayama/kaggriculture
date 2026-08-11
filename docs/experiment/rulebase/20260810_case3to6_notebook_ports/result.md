# rulebase case3–case6 — porting the public live-heuristic notebooks

**Date**: 2026-08-10
**Cases**: `backend/pipeline/rulebase/case{2,3,4,5}`
**Engine**: `kaggle-environments==1.32.6`

## Scope

Of the 19 public notebooks under `backend/notebook/20260809/`, four contain a
genuine live `agent(obs)` heuristic that was not already absorbed into case2:

| Case | Notebook | Lines | Character |
|---|---|---|---|
| case3 | `pilkwang__...structured-economic-policy` | 2005 | Phase machine + job matching |
| case4 | `dariushafshar__care-is-a-4-2x-multiplier...` | 152 | Husbandry only, CARE-first |
| case5 | `bovard__kaggriculture-getting-started` | 97 | Official tutorial baseline |
| case6 | `delayedkarma__whatwheatwrought...` | 44 | Single-tile wheat loop |

`semalytics` is not a separate case — it is the structural basis of case2.

**Excluded by decision:** the 10 replay-tape notebooks (a base85+zlib 719-entry
action table replayed by step index). They are the actual top of the public
ladder but are not rule-based policies, do not generalise off the recorded
distribution, and the strongest ladder agent (Seb, #1) is only 35–66%
self-identical — i.e. genuinely adaptive, so the tape approach does not even
describe the leader. The 4 pure meta-analysis notebooks contain no agent.

## Results

### Against builtins — 10 episodes per seat, seats swapped

| Case | starter | random | pass | Mean money | Errors |
|---|---|---|---|---|---|
| **case3** | 20/20 | 20/20 | 20/20 | **137,175** | 0 |
| case2 | 20/20 | 20/20 | 20/20 | 33,779 | 0 |
| case4 | 20/20 | 20/20 | 20/20 | 29,671 | 0 |
| case5 | 20/20 | 20/20 | 20/20 | 5,957 | 0 |
| case6 | 20/20 | 20/20 | 20/20 | 5,505 | 0 |

**100/100 builtin matches won, zero errored episodes.**

### Head-to-head — 6 episodes per seat, seats swapped

| Matchup | Result | Mean margin (seat 0) |
|---|---|---|
| case3 vs case5 | 12/12 case3 | +144,472 |
| case3 vs case6 | 12/12 case3 | +122,479 |
| case3 vs case4 | 12/12 case3 | +114,987 |
| case3 vs case2 | 12/12 case3 | +97,242 |
| case2 vs case5 | 12/12 case2 | +29,279 |
| case2 vs case6 | 12/12 case2 | +27,399 |
| case4 vs case6 | 12/12 case4 | +22,485 |
| case4 vs case5 | 12/12 case4 | +20,395 |
| case2 vs case4 | 10/12 case2 | +7,479 |
| case5 vs case6 | 12/12 case5 | +891 |

A clean total order: **case3 ≫ case2 > case4 ≫ case5 ≈ case6**.

The only non-sweep is case2 vs case4 (10/12), which is also the closest pair on
mean money — the crop-and-dairy generalist and the pure-husbandry specialist are
genuinely comparable at this strength.

## The headline finding

**case3 banks 4.1× case2 and beats it 12/12.** A hand-built agent informed by the
notebooks is decisively worse than the best notebook itself. Its edge is not one
trick but density: a five-phase policy machine, a job/value/travel matching
score, per-product reserve fractions, and — uniquely among all ported cases —
**opponent-aware sizing** (`_melon_target` reads the opponent's melon tile count
and the current price before choosing 8, 10 or 12 tiles).

case2's post-mortem listed "no opponent modelling" as a known limitation. case3
demonstrates what that limitation is worth.

## Verification

Every port was checked against the engine rather than trusted:

- **case3**: `CROPS` / `ANIMALS` / `MARKET` / `SHOPS` compared field-by-field
  (its field names differ from the engine's; its values do not) — **0 diffs**.
  `_price_at` compared against `engine.market_price` over 9 products × 7
  inventory levels — **0 diffs**. This port needed no correction.
- **case4**: fixed a latent bug — the original mutated `priv["shed"]` during
  worker assignment to stop two workers claiming the same shed animal. Replaced
  with a local tally; mutating the observation is fragile and would break if the
  harness ever shared or reused that dict.
- **case5**: the notebook imports `CROPS` from `kaggle_environments`. The two
  constants it needs are inlined — submitted code cannot rely on engine
  internals being importable.
- **case3 global state**: carries three module-level mutables that persist across
  turns. Verified that repeated same-seed episodes in one interpreter reproduce
  bit-identical results, so local benchmarks are not order-dependent.

All four received a never-raise wrapper and defensive key access.

## Repository changes

- **mypy**: `pipeline/rulebase/case[2-5]/` excluded. case3 alone produced 177
  errors under `strict`, 144 of them "untyped function" noise. These are
  upstream code kept byte-comparable to their source so a future upstream
  revision can be diffed against them; annotating 2000 lines would fork them
  permanently for no behavioural gain. Correctness is pinned by e2e episodes and
  explicit engine constant/price checks instead. **case2 stays fully strict.**
- **ruff**: `pipeline/rulebase/case3` excluded for the same reason (it is
  verbatim). case4–case6 are lightly adapted and are linted normally.
- **e2e tests**: `test_rulebase_seasons.py` now parametrises over all five cases
  — full season without erroring, and beats `starter` 3/3. 16 tests.

## Reading of the family

| Case | Keep for |
|---|---|
| case3 | The strongest agent available, and the reference for what a mature heuristic looks like |
| case2 | The only case we can safely tune — 5 small modules, strict typing, mutation-tested unit suite |
| case4 | The CARE/animal-economics argument, with its own statistical backing |
| case5, case6 | Floors. Useful to confirm a change is not accidentally catastrophic |

case3 is the strongest but the hardest to attribute: at 2005 lines, a score
change cannot easily be traced to one mechanism. The productive path is to lift
individual mechanisms from case3 into case2 — where they can be measured
one at a time — starting with opponent-aware crop sizing and per-product reserve
fractions.

## Caveats

- All numbers are against builtins and each other, **not the leaderboard**. The
  real ladder is dominated by replay-tape agents that none of these cases faces
  here.
- Engine behaviour is strongly version-dependent (case4's own source ablation
  gives 30.53× on 1.29.3 vs 7.12× on 1.32.6). These results hold for 1.32.6 only.
- Head-to-head used 6 episodes per seat; the margins are large enough that the
  ordering is unambiguous, but the case2/case4 pair (10/12) would need more
  episodes to pin down precisely.
