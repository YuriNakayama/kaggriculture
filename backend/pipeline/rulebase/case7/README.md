# rulebase/case7 — case2 + measured mechanisms from case3

case2 with three mechanisms lifted from case3, each behind a flag so its
contribution could be measured rather than assumed. **Two of the three were
measured and turned off.** The change that actually paid was a latent case2 bug
the experiment exposed.

## Results

| Opponent | Seat 0 | Seat 1 | Total | Mean money |
|---|---|---|---|---|
| `starter` | 15/15 | 15/15 | 30/30 | 44,977 |
| `random` | 15/15 | 15/15 | 30/30 | 48,972 |
| `pass` | 15/15 | 15/15 | 30/30 | 43,546 |

**Against builtins: +28% over case2** (44,977 vs 33,779).

Head-to-head, 8 episodes per seat:

| Matchup | Result |
|---|---|
| vs case4 | **16/16** for case7 |
| vs case2 | **7/16** for case7 — a loss |
| vs case3 | 0/16 |

> **Read the head-to-head, not the mean bank.** case7 banks 33% more than case2
> against a passive baseline yet loses to it directly. See *The joint-market
> problem* below. Do not treat this case as a straight upgrade.

## Ablation (30 episodes per configuration)

| Configuration | Mean money | Delta |
|---|---|---|
| zone-free deadlines + herd expansion | **49,275** | — |
| without herd expansion | 45,768 | −3,507 |
| **without zone-free deadlines** | 34,561 | **−14,714** |
| with strawberry | 47,356 | −1,919 |
| with opponent-aware melon | 47,356 | ±0 |

### What shipped, and why

| Mechanism | State | Evidence |
|---|---|---|
| **Zone-free deadlines** | on | **+14,714.** Not a ported mechanism — a case2 bug this work exposed. |
| Herd expansion + labour scaling | on | +3,507 |
| Strawberry | **off** | −1,919 |
| Opponent-aware melon sizing | **off** | exactly 0 — never fires |

## The bug worth more than everything else

case2 partitions the board into per-unit column zones, and penalises
out-of-zone work by +100 priority. That is fine at 2 animals. At 10+ it is
crippling: **livestock clusters into a few columns**, so with 8 animals spread
across 4 of 8 zones, only 4 units could ever feed — the rest walked to their own
zones while animals starved beside them. Measured at **30 animals lost per
season**, with 12 empty pastures at the final step.

Deadline work (FEED, rescue-WATER, HARVEST, CARE) is now `zone_free` and
claimable farm-wide; only growth work stays zone-bound. Animal losses fell
**30 → 5** and the herd held at 10–11 instead of collapsing.

This was invisible in case2 because a 2-cow herd fits in one zone. It only
became a defect once something else grew.

## Negative results (kept behind flags, deliberately)

**Strawberry: −1,919.** The hypothesis was that strawberry supplies 22% of
case3's revenue against 0% of case2's, so adding it should pay. It does not:
strawberry needs 10 days to first yield and then produces on a 2-day interval,
so 6–7 tiles sat occupied for most of the season and returned **24 units total**
while displacing melon that earns more on the same ground.

**Opponent-aware melon sizing: exactly 0.** 47,356 with and without, to the
dollar, over 30 episodes — the code never fires. No opponent available here trips
a threshold: `starter` grows only carrots, and case5 (a melon monocrop) peaks at
**one** melon tile against a branch needing 12. Sound in principle, but untested
on this ladder, so it stays off.

## The joint-market problem

case7 banks 45k against builtins and 20k against case2 — and case2 also drops to
20k. Both crowd the same products, and prices collapse for both:

| Day | MELON | MILK |
|---|---|---|
| 10 | 271 | 195 |
| 25 | 140 | 82 |
| 29 | **4** | **26** |

Sell volumes are near-identical (case7: 151 MILK / 109 MELON; case2: 151 / 108),
so case7's larger herd buys it no extra product — it pays the feed and animal
cost without the revenue. This is the competition's core difficulty stated
plainly: **production capacity is worth nothing without somewhere to sell it.**

The next real gain is not more production. It is either differentiating into
products the opponent is not dumping, or timing sales against the opponent's
supply — which is what case3's untestable melon mechanism was reaching for.

## Changes from case2

| File | Change |
|---|---|
| `config.py` | Three feature flags; herd ramp, labour bounds, cash reserves |
| `observe.py` | `Snapshot.opponent_crop_tiles()` — farms are public |
| `tasks.py` | `Task.zone_free`; herd ramp; both pasture species; melon sizing |
| `market.py` | Workload-scaled hiring; species-aware, herd-scaled buying |

Cash reserve now scales with the herd (`CASH_RESERVE + 120/animal`). The first
build bought 4 cows on day 0 for $1,600 off a $3,000 start, pinning the farm at
~$0 for 20 days and starving both hiring and feed.
