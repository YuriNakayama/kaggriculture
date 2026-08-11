# rulebase/case3 — structured economic policy

Port of `pilkwang__kaggriculture-structured-economic-policy` (2005 lines). The
strongest from-scratch heuristic in the public notebook set, and by a wide
margin the strongest case in this repo.

**~149k mean bank vs `starter`, against case2's ~35k.**

## Verification

Unlike every other port, this one needed **no correction**. Its `CROPS`,
`ANIMALS`, `MARKET` and `SHOPS` tables and its `_price_at` were checked
field-by-field against `kaggriculture.py` at 1.32.6:

- table field diffs: **0** (its field names differ, its values do not)
- price diffs across 9 products × 7 inventory levels: **0**

## Architecture

Not a state machine — a per-turn six-stage pipeline:

```
observation → phase + role assignment → job generation
            → priority/value/travel matching → inventory shadow → market
```

Five phases (`_policy_phase`), checked in order:

| Phase | Trigger |
|---|---|
| `LIQUIDATE` | final 22 turns |
| `CRISIS` | at-risk animals + crops > workers, or shed ≥ 95 |
| `BOOTSTRAP` | day ≤ 4 |
| `COMPOUND` | day ≤ 21 |
| `REALIZE` | otherwise |

Jobs match to workers by `S = priority_bonus + value − 8 × manhattan`, with
`PRIORITY_BONUS` spanning 120000 down to −100 so feeding, rescue watering and
terminal harvests are lexicographically protected. Growth work only consumes
labour those deadlines did not need — the same principle as case2, but with a
much richer job/value model.

## Mechanisms worth stealing

- **Adaptive melon sizing** (`_melon_target`): 12 tiles when MELON ≥ $300 and the
  opponent holds ≤ 5 melon tiles; 8 when price ≤ 170 or opponent ≥ 12; else 10.
  This is the **only opponent-aware logic in any ported case**.
- **Herd ramp**: `COW, COW, COW, SHEEP` core → 11 by day 7 → 15 by day 11, no
  animal purchases after day 18.
- **Reserve fractions** (WOOL 0.40, MILK 0.42, MELON 0.58, WHEAT 0.68) withhold
  stock instead of selling everything available.
- **Labour scaling**: `H* = min(cap, max(floor, ceil((J + 2R)/7)))`, cap 12
  before day 20 and 13 from day 20 (the 13th hand costs 233/day marginal).

## Market theory from the notebook

Two results worth carrying into later cases:

1. Ordering ahead of a non-reacting block of `q` units moves the **objective** by
   **2×** the price impact, because the margin is a difference of two banks.
2. SELL slots should be ranked by **self-impact** `I = q × (p(I) − p(I+q))`, not
   gross proceeds — gross is dominated by base price and defers cheap-but-steep
   products until after they are crushed.

It also notes that withholding inventory lifts the price path for *both* seats,
so it is a transfer to the opponent whenever they are the larger supplier in
that window.

## Changes from the notebook

Only the `%%agentfile` cell magic was stripped. The code is otherwise verbatim:
it is stdlib-only, already carries a never-raise wrapper, and already defines
`agent` last (which the harness requires).

## Caveats

- **Three module-level mutable globals** (`_SIGNATURE_ACTIVE`,
  `_SIGNATURE_LAST_STEP`, `_SCHEDULE_WHEAT_REQUESTED`) persist across turns.
  Verified that repeated same-seed episodes in one interpreter reproduce
  bit-identical results, but any future change to how they reset would silently
  make local benchmarks order-dependent.
- **2005 lines makes attribution hard.** A score change cannot easily be traced
  to one mechanism. Treat this as a reference implementation to learn from, not
  a codebase to tune blind — that is what case2 is for.
- The notebook claims no leaderboard result; it only asserts a deterministic
  720-turn run vs `starter` on seed 217 in both seats.
