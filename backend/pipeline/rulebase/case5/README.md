# rulebase/case5 — "Melon Maxxer" (tutorial baseline)

Port of the official getting-started notebook
(`bovard__kaggriculture-getting-started`). Included as a **reference point**, not
as a competitive entry: it answers "what is the tutorial strategy actually
worth?"

## Strategy

Single farmer, melon monocrop, no hands, no land, no animals. Priority is
harvest → water → plant, walking greedily toward the nearest tile that needs
work. Sells the entire shed in one order whenever MELON ≥ $200.

## Changes from the notebook

1. The notebook does `from kaggle_environments...kaggriculture import CROPS`.
   The two constants it uses (`MELON` seed cost 80, `max_yield_day` 12) are
   inlined — submitted code cannot depend on engine internals being importable.
2. Top level wrapped in `try/except`; an uncaught exception forfeits the episode.

Otherwise the logic is unchanged, including its known weaknesses.

## Known weaknesses

The tutorial names most of these itself:

- **Never hires.** Consistently identified across the meta notebooks as the
  single most expensive mistake — 4 hands cost `1+1+2+3 = 7`/day for 96 extra
  worker-turns.
- **One tile at a time.** A lone farmer caps throughput regardless of crop.
- **Dumps the whole shed in one order**, walking its own price down the curve.
- **Never liquidates.** Unsold produce at step 720 scores zero, and the
  `SELL_THRESHOLD = 200` gate can hold stock past the end of the season.
- **Melon monocrop.** MELON is demanded by **zero** town shops (WHEAT by 5), so
  its realized price decays with nothing draining the glut.
