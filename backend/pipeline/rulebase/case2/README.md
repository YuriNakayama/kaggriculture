# rulebase/case2 — zone-partitioned wheat + melon + dairy

The first multi-crop `rulebase` case, and the baseline the later cases here are
measured against. Built from the public Kaggle notebooks (see *Provenance*),
with every game constant re-derived from the engine source at
`kaggle-environments==1.32.6` rather than trusted from the notebooks.

Where this sits relative to [`case1`](../case1/README.md): case1 is a
single-crop wheat loop chosen for wheat's shallow glut curve. This case keeps
that reasoning for its wheat component but adds melon, dairy, land purchases and
a workforce, banking roughly 3.3× as much against `starter` (33.8k vs 10.2k).

## Hypothesis

A hand-written heuristic that (a) keeps every hired hand busy, (b) treats the
two end-of-day deadlines as hard constraints, and (c) sells in small batches
instead of dumping, clears the `starter` baseline by a wide margin and provides
a stable floor for later cases.

## Result

15 episodes per seat against each builtin, seats swapped:

| Opponent | Seat 0 | Seat 1 | Total | Mean money |
|---|---|---|---|---|
| `starter` | 15/15 | 15/15 | **30/30** | 34,846 |
| `random` | 15/15 | 15/15 | **30/30** | 33,773 |
| `pass` | 15/15 | 15/15 | **30/30** | 34,564 |

**90/90, zero errored episodes.** `dev/submit --dry-run` passes: 5 files, 11.6 KB.

Full write-up: [`docs/experiment/rulebase/20260810_case2_baseline/result.md`](../../../../docs/experiment/rulebase/20260810_case2_baseline/result.md)

## Strategy

| Element | Choice | Why |
|---|---|---|
| Labour | 6 hands hired every hour 0 | Fibonacci cost `1+1+2+3+5+8 = 20/day`; not hiring is the most expensive mistake in the meta |
| Assignment | Column-band zones, greedy by (priority, distance) | Two units never claim the same tile, so no collision handling is needed |
| Crops | MELON 10 + WHEAT 8 **per unlocked quadrant** | MELON has the highest base ($250); WHEAT feeds the herd and is demanded by 5 shop types |
| Animals | 2 COW | MILK is demanded by 3 shop types; COW's 2-day interval pays back faster than SHEEP's 3-day |
| Land | NE on day 5, SW on day 9; SE skipped | SE costs $4,000 and is bought by ~0% of top players |
| Selling | Per-item batch caps, ranked by proceeds ÷ price damage | Premium goods hit the $1 floor after only 60–80 units |
| Endgame | Liquidate from step 700 | Unsold stock scores zero |

### Priority order

`FEED` → `FETCH_FEED` → `RESCUE_WATER` → `HARVEST` → `FETCH_ANIMAL` →
`PLACE_ANIMAL` → `BUILD_PASTURE` → `WATER` → `CARE` → `PLANT` → `DIG` →
`COLLECT_FERTILIZER`

The first three encode engine deadlines: `consecutive_unwatered >= 2` turns a
plant into a weed, and `consecutive_unfed >= 2` makes an animal escape. Growth
work only ever uses labour those deadlines did not need.

## Engine facts this case depends on

Read from `kaggriculture.py` at 1.32.6, and worth re-checking on any engine bump:

- **Step 719 never executes.** `interpreter` marks `DONE` when
  `step >= episodeSteps - 2`, so everything must be sold by step 718.
- **Selling at the $1 floor does not add market supply** (`_commit_unit`), so a
  floored product cannot be pushed down further.
- **Market orders resolve in per-unit lockstep across both players**, and `SELL`
  placed before a `BUY` in the same list funds that buy on the same turn.
- **Bought animals land in the shed**, not on the board; someone must `PICKUP`
  and `PLACE` them. Missing this leaves the herd in the shed all season.
- **Atomic PLANT validation**: if a turn requests more seeds of a crop than are
  held, *every* PLANT for that crop is dropped — over-requesting wastes them all.
- **CARE banks `pending_care_bonus`**, paid out only on a day the animal was
  also fed.
- `PICKUP` / `PLACE` / `DROP` work from any of the four shed-access tiles
  `(4,4) (5,4) (4,5) (5,5)`, which are *not* the shed itself.

## Layout

| File | Role |
|---|---|
| `main.py` | `agent(obs)` entry point and the never-raise guard |
| `config.py` | Engine constants (mirrored from 1.32.6) + strategy tuning |
| `observe.py` | Defensive `obs` → `Snapshot` parsing |
| `tasks.py` | Task generation, zone assignment, BFS pathing |
| `market.py` | Price model, sell ranking, purchase scheduling |

Modules import each other **by bare name**, not relatively: the Kaggle harness
`exec`s `main.py` with neither `__name__` nor `__file__` defined, having first
appended the case directory to `sys.path`. The case is therefore a plain
directory with no `__init__.py`, and `agent` must stay the **last** callable
defined in `main.py` because the harness takes the last one.

## Standing in the family

case3 (a verbatim port of `pilkwang`) banks **4.1×** what this case does and
beats it **12/12** head-to-head. case2 remains the case to *tune*, though: five
small modules, fully strict typing, and a mutation-tested unit suite, versus
case3's 2005 unannotated lines where a score change cannot be attributed to any
one mechanism.

The productive path is to lift mechanisms from case3 into case2 one at a time —
opponent-aware crop sizing and per-product reserve fractions first. See
[`docs/experiment/rulebase/20260810_case3to6_notebook_ports/result.md`](../../../../docs/experiment/rulebase/20260810_case3to6_notebook_ports/result.md).

## Known limitations (candidates for case7)

1. **Animals still escape 1–2×/season.** Feed logistics are reactive; a unit
   fetches wheat only once animals are already hungry.
2. **Weeds accumulate** (~10 mid-season). `DIG` sits below `PLANT`, so weeded
   tiles are reclaimed late.
3. **Herd is tiny.** 2 cows against a meta modal farm of 8 cows + 6 sheep. The
   pasture/feed pipeline needs to scale before the herd can.
4. **No opponent modelling.** `Snapshot` carries `opponent_tiles` and
   `opponent_money`, but the policy ignores both; the market is a joint problem.
5. **Land purchase is unconditional on schedule** rather than on marginal return.

## Provenance

- **semalytics** — zone-partitioned workers, task-priority structure.
- **dariushafshar** — CARE mechanics (`1 + interval` units/cycle, not a
  percentage) and animal economics.
- **raykkretzschmar** — the correction that realized price is governed by *how
  many town shops demand a product*, not glut-curve steepness. This is why the
  crop mix is not melon-only: MELON is demanded by **0** shops, WHEAT by 5.
- **cjlcjlcjl** — units-to-floor measurements behind the sell batch caps.
- **romantamrazov** — step 719 is never executed.

Notebook claims were verified against the engine before use; several
(melon-only cropping, `SEED_BUFFER` sizing) did not survive that check.
