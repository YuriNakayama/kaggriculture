# rulebase/case7 — transferring case3's mechanisms into a measurable case

**Date**: 2026-08-11
**Case**: `backend/pipeline/rulebase/case7`
**Engine**: `kaggle-environments==1.32.6`

## Hypothesis

case3 (a 2005-line verbatim notebook port) banks 4.1× case2 but cannot be tuned:
no score change can be attributed to any single mechanism. Lifting its
mechanisms into case2 — small, strict-typed, mutation-tested — one flag at a
time should transfer some of that advantage *and* say which parts carry it.

Three candidates were chosen from a revenue breakdown of case3 against case2:

| Mechanism | Rationale |
|---|---|
| Herd expansion + labour scaling | case3 ends on 15 animals / 13 hands vs case2's 2 / 6; MILK+WOOL is 39% of its revenue vs case2's 23% |
| Opponent-aware melon sizing | case3's only opponent-aware logic; case2's own post-mortem listed "no opponent modelling" as a limitation |
| Strawberry | 22% of case3's revenue, 0% of case2's |

## Result

**Two of the three hypotheses were wrong.** The change that paid was a case2 bug
the experiment exposed.

### Ablation, 30 episodes per configuration, seats swapped

| Configuration | Mean money | Delta |
|---|---|---|
| zone-free deadlines + herd expansion (shipped) | **49,275** | — |
| without herd expansion | 45,768 | −3,507 |
| **without zone-free deadlines** | 34,561 | **−14,714** |
| with strawberry | 47,356 | −1,919 |
| with opponent-aware melon | 47,356 | ±0 |

### Final standing

| Opponent | Total | Mean money |
|---|---|---|
| `starter` | 30/30 | 44,977 |
| `random` | 30/30 | 48,972 |
| `pass` | 30/30 | 43,546 |

**+28% over case2 against builtins** — but see the head-to-head caveat below.

## The finding: a latent bug worth more than every ported mechanism

case2 partitions the board into per-unit column zones and penalises out-of-zone
work by +100 priority. Correct at 2 animals; crippling at 10+, because
**livestock clusters into a few columns**. With 8 animals across 4 of 8 zones,
only 4 units could ever feed — the rest walked to their own zones while animals
starved beside them.

Measured: **30 animals lost per season**, 12 empty pastures at the final step,
and at day 16, *9 animals with only 1 fed* despite 26 wheat in the shed and 9
carried. Feed was never the constraint; labour reaching the animals was.

Making deadline work (FEED, rescue-WATER, HARVEST, CARE) claimable farm-wide
dropped animal losses **30 → 5** and was worth **+14,714** on its own.

This was invisible in case2 because a 2-cow herd fits inside one zone. It became
a defect only when something else grew — the kind of bug that has no symptom
until an unrelated change removes the condition hiding it.

## Negative results, recorded rather than discarded

**Strawberry: −1,919.** Strawberry needs 10 days to first yield and then
produces on a 2-day interval. 6–7 tiles sat occupied for most of the season and
returned **24 units total**, displacing melon that earns more on the same
ground. case3 can afford it because it runs three quadrants of crops and can
absorb a slow-maturing line; case7 cannot.

**Opponent-aware melon sizing: exactly 0** — 47,356 with and without, to the
dollar. The code never fires. `starter` grows only carrots; case5 (a melon
monocrop) peaks at **one** melon tile against a branch needing 12. Left in,
flagged off, with the reason recorded: it is untested code, not measured-bad
code, and the distinction matters for whoever revisits it.

## The result that matters most

**case7 loses to case2 head-to-head, 7/16**, despite banking 33% more against
every builtin.

Playing each other, both collapse from ~45k to ~20k:

| Day | MELON | MILK |
|---|---|---|
| 10 | 271 | 195 |
| 25 | 140 | 82 |
| 29 | **4** | **26** |

Sell volumes come out near-identical (case7: 151 MILK / 109 MELON; case2: 151 /
108). case7's larger herd buys it **no extra product** — it pays the feed and
purchase cost without the revenue, and its extra supply deepens the glut it is
selling into.

This is the competition's stated core difficulty, reproduced in miniature:
*production capacity is worth nothing without somewhere to sell it.* Against a
passive opponent, more production is strictly better. Against a live one selling
the same goods, it is close to self-harm.

## Methodological notes

- **Benchmark against builtins is misleading on its own.** Every case in this
  family beats `starter` 100%; the metric that separated case7 from case2 was
  head-to-head, and it pointed the other way from mean bank.
- **Flags earned their cost.** Without per-mechanism gating, case7 would have
  shipped strawberry (−1,919) and inert melon logic bundled into a "+28%" result,
  and the +14,714 zone fix would have been invisible inside the aggregate.
- **First build was worse than case2** (29,894 vs 35,105): it bought 4 cows on
  day 0 for $1,600 off a $3,000 start, pinning the farm near $0 for 20 days.
  Cash reserve now scales with herd size.

## Repository changes

- `dev/lint` type-checks each authored case (case2, case7) in its own mypy pass.
  Flat cases both define `config`/`tasks`, so a single run cannot resolve them —
  previously mypy silently checked case7's code against **case2's** modules.
- `pyproject.toml`: `mypy_path = "src"`; `pipeline/rulebase/` excluded from the
  default pass and handled per-case by `dev/lint`.
- `test_case7_assignment.py`: 8 unit tests, loading case7 via importlib so both
  cases can be unit-tested in one pytest run. The zone test was **verified to
  fail** when the fix is reverted (1 fed instead of 3).

## Next

The productive direction is no longer more production:

1. **Differentiate.** Read the opponent's crop mix and grow what they are not.
   The machinery exists (`Snapshot.opponent_crop_tiles`) and is unused.
2. **Time sales against opponent supply** rather than by fixed batch caps —
   what case3's melon mechanism was reaching for, and what needs an opponent
   that can actually exercise it.
3. **Build a sparring opponent** that grows melon at scale, so opponent-aware
   logic can be measured at all. Its absence is why one mechanism here is
   unfalsifiable rather than merely unproven.
