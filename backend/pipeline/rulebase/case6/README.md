# rulebase/case6 — single-tile wheat loop

Port of `delayedkarma__whatwheatwrought-a-kaggriculture-tale`, the smallest
coherent agent in the notebook set (44 lines). A floor, not a strategy.

## Strategy

Buy one wheat seed → plant → water → harvest on day 2 → sell → repeat. No hands,
no land, no animals, no market model.

The premise is cycle speed: wheat first yields on day 2 against melon's day 10,
so a single farmer keeps money turning over continuously instead of waiting out
a long maturation.

## Why it plateaus

One farmer can work exactly one tile per turn, so the fast cycle buys nothing —
throughput is bounded by actions, not by crop speed. Wheat is also the
cheapest product in the game ($25 base), so each cycle realizes very little.

Its one genuine virtue: wheat's glut curve is shallow (`log`, T=400), so dumping
the whole stock every cycle costs almost nothing. The same code applied to a
premium good would crater its own price — the notebook's implicit point.

## Changes from the notebook

Never-raise wrapper and defensive key access only. The original's movement scan
is preserved verbatim, including its quirk of walking toward tiles that are
already planted (it targets "empty **or** PLANT" tiles, so it often walks to a
tile it cannot act on).
