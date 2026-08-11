# rulebase/case4 — CARE-first animal husbandry

Port of `dariushafshar__care-is-a-4-2x-multiplier-animal-economics`. Grows no
crops and runs no market model: it keeps a mixed COW/SHEEP herd fed, cared for
and harvested every day, buys its feed, and drips product into the market.

## Strategy

| Element | Choice |
|---|---|
| Herd | 4 COW + 3 SHEEP, capped |
| Crops | none — wheat feed is **bought**, not grown |
| Hands | 3, hired from day 1 once money > $2,000 |
| Selling | MILK / WOOL in batches of ≤ 4, never dumped |
| Feed | keep ~3 days in stock (`max(8, n_alive × 3)`) |

Two species on purpose: MILK and WOOL sit on independent price curves, so a glut
in one does not drag the other down.

## Why CARE dominates the design

`CARE` is **not** a percentage multiplier — a common misreading. From
`_daily_refresh_animals` in the engine: a day where the animal is both fed and
cared for increments `pending_care_bonus`, and that bonus pays out in full on
the next production tick, but only if the animal was also fed that day.

Steady state is therefore `1 + interval` units per cycle, so the **slower the
animal, the more CARE is worth**. Measured over 30 days: GOOSE 2.07×,
COW 3.25×, SHEEP 4.22×.

## Why the herd is deliberately small

The notebook ran 288 paired episodes with McNemar tests. Every attempt to grow
the herd was *significantly worse*:

| Change | Win rate | z |
|---|---|---|
| baseline (8 cow, 6 sheep) | 19.1% | — |
| sheep 6 → 8 | 8.7% | **−4.52** |
| cows 8 → 10 | 12.5% | **−3.53** |
| cows 8 → 6 | 18.4% | −0.33 (free) |

Product price collapses faster than volume grows — WOOL falls off a cliff
(`sq`, above_target 3.20, T=105). Its conclusion: *"CARE decides which animal
you keep. The town sink decides how many."*

## Changes from the notebook

1. Never-raise wrapper, with hand-count-correct fallback.
2. Defensive key access throughout.
3. **Fixed a latent bug**: the original decremented `priv["shed"]` while
   assigning workers, mutating the observation to stop two workers claiming the
   same shed animal. Replaced with a local tally — mutating `obs` is fragile and
   would break if the harness ever shared or reused that dict.
4. Dropped the `CARE_ENABLED` ablation flag (always on here).

## Caveat on the headline number

The notebook's "7.12× final bank" is CARE-on vs CARE-off against the weak
`starter` baseline, not a leaderboard claim — its author notes their tuned
agents "sit mid-table". It is also strongly version-dependent: the same ablation
gives **30.53× on 1.29.3** but **7.12× on 1.32.6**.
