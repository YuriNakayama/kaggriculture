"""The RNG-driven rules, checked by distribution rather than by sequence.

``test_jaxenv_equivalence.py`` disables weed spawning and shop unlocks so it can
compare the port against the engine turn by turn. It has to: the engine draws
from a per-day ``random.Random`` and consumes one draw per empty tile, so the
stream depends on board occupancy in a way a batched device program cannot
reproduce. The port uses a per-environment JAX key instead.

That makes these two rules untestable by equality, so they are tested for the
properties that actually matter to a learning agent: the right spawn rate, the
right unlock cadence, the documented caps, per-environment independence, and
reproducibility from a seed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

jax = pytest.importorskip("jax", reason="JAX is an optional dependency")
jnp = pytest.importorskip("jax.numpy", reason="JAX is an optional dependency")

from simulate.jaxenv import env as E  # noqa: E402
from simulate.jaxenv.state import (  # noqa: E402
    KIND_EMPTY,
    KIND_WEED,
    MAX_SHOP_INSTANCES,
    MAX_UNITS,
    N_SHOPS,
    TURNS_PER_DAY,
    EnvState,
    initial_state,
)

StepFn = Callable[..., EnvState]


def _idle_actions(batch: int) -> tuple[Any, ...]:
    """Everyone passes; no market orders."""
    unit = jnp.zeros((batch, 2, MAX_UNITS), dtype=jnp.int32)
    market = jnp.zeros((batch, 2, E.MAX_MARKET_ORDERS), dtype=jnp.int32)
    return unit, unit, unit, market, market, market


def _run_idle(state: EnvState, days: int, step_fn: StepFn) -> EnvState:
    actions = _idle_actions(state.step.shape[0])
    for _ in range(days * TURNS_PER_DAY):
        state = step_fn(state, *actions)
    return state


def test_weed_spawn_rate_matches_configured_chance() -> None:
    """Over many environments the empty-tile spawn rate tracks the parameter.

    One day, one spawn opportunity per empty tile. With 25 unlocked empty tiles
    per player and a large batch the observed rate should sit close to the
    configured chance; the tolerance is wide enough that only a genuinely wrong
    rate (a missing draw, a per-board instead of per-tile draw) fails.
    """
    chance = 0.1
    batch = 512
    step_fn = E.make_step(weed_chance=chance, shop_unlock_interval=10**6)

    state = initial_state(batch, seed=0)
    n_empty_before = int((state.kind == KIND_EMPTY).sum())

    state = _run_idle(state, days=1, step_fn=step_fn)
    n_weeds = int((state.kind == KIND_WEED).sum())

    observed = n_weeds / n_empty_before
    assert chance * 0.85 < observed < chance * 1.15, (
        f"weed rate {observed:.4f} is far from the configured {chance}"
    )


def test_no_weeds_spawn_when_chance_is_zero() -> None:
    step_fn = E.make_step(weed_chance=0.0, shop_unlock_interval=10**6)
    state = _run_idle(initial_state(8, seed=3), days=5, step_fn=step_fn)
    assert int((state.kind == KIND_WEED).sum()) == 0


def test_weeds_only_replace_empty_tiles() -> None:
    """Locked ground never grows weeds, so the NW quadrant stays the only risk."""
    step_fn = E.make_step(weed_chance=1.0, shop_unlock_interval=10**6)
    state = _run_idle(initial_state(4, seed=5), days=1, step_fn=step_fn)

    half = 10 // 2
    # Everything outside NW starts LOCKED and must remain untouched.
    locked_region = state.kind[:, :, half:, :]
    assert int((locked_region == KIND_WEED).sum()) == 0
    locked_region_e = state.kind[:, :, :, half:]
    assert int((locked_region_e == KIND_WEED).sum()) == 0


def test_shops_unlock_on_schedule_and_respect_the_cap() -> None:
    """One shop per unlock interval, never exceeding MAX_SHOP_INSTANCES."""
    interval = 3
    step_fn = E.make_step(weed_chance=0.0, shop_unlock_interval=interval)

    state = initial_state(16, seed=11)
    for day in range(1, 31):
        state = _run_idle(state, days=1, step_fn=step_fn)
        expected = min(day // interval, MAX_SHOP_INSTANCES)
        counts = state.shops.sum(axis=-1)
        assert set(counts.tolist()) == {expected}, (
            f"after day {day}: expected {expected} shops, got {set(counts.tolist())}"
        )

    assert int(state.shops.sum(axis=-1).max()) == MAX_SHOP_INSTANCES


def test_shop_draws_cover_the_whole_catalogue() -> None:
    """Shops are drawn uniformly with replacement, so all 8 should appear."""
    step_fn = E.make_step(weed_chance=0.0, shop_unlock_interval=3)
    state = _run_idle(initial_state(256, seed=17), days=24, step_fn=step_fn)

    drawn = state.shops.sum(axis=0) > 0
    assert int(drawn.sum()) == N_SHOPS, (
        f"only {int(drawn.sum())} of {N_SHOPS} shops were ever drawn"
    )


def test_same_seed_reproduces_the_same_episode() -> None:
    """A seed fully determines the stochastic stream, so runs are replayable."""
    step_fn = E.make_step(weed_chance=0.05, shop_unlock_interval=3)

    a = _run_idle(initial_state(4, seed=99), days=6, step_fn=step_fn)
    b = _run_idle(initial_state(4, seed=99), days=6, step_fn=step_fn)

    assert bool(jnp.array_equal(a.kind, b.kind))
    assert bool(jnp.array_equal(a.shops, b.shops))


def test_environments_in_a_batch_have_independent_streams() -> None:
    """Distinct seeds per batch element, so a batch is not one episode repeated."""
    step_fn = E.make_step(weed_chance=0.05, shop_unlock_interval=3)
    state = _run_idle(initial_state(32, seed=123), days=9, step_fn=step_fn)

    # Shop draws should differ across the batch rather than moving in lockstep.
    rows = {tuple(row.tolist()) for row in state.shops}
    assert len(rows) > 1, "every environment drew an identical shop sequence"

    # Same for weeds: identical boards would mean a shared stream.
    boards = {bytes(state.kind[b].tobytes()) for b in range(state.kind.shape[0])}
    assert len(boards) > 1, "every environment grew identical weeds"
