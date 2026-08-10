"""Declarative description of what to evaluate.

A :class:`MatchSpec` is one cell of the evaluation matrix: a single episode of
one case against one opponent on one seed. :class:`SimConfig` holds the
knobs that apply to a whole run and expands into the spec list.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

#: Full season: 24 turns/day x 30 days.
FULL_SEASON_STEPS = 720

#: Per-turn wall-clock budget the Kaggle harness enforces (`actTimeout`).
ACT_TIMEOUT_SEC = 1.0

#: Fraction of ACT_TIMEOUT_SEC above which a turn is reported as "slow".
#: Slow turns are not failures, but they are the only early warning that an
#: agent is drifting toward the limit on the competition's hardware.
SLOW_TURN_FRACTION = 0.5


class ReplayMode(StrEnum):
    """How much replay JSON to keep. Each episode serialises to roughly 5 MB."""

    NONE = "none"
    FAILED = "failed"
    ALL = "all"


class Engine(StrEnum):
    """Which execution path runs the episode.

    ``OFFICIAL`` goes through ``env.run`` — the same path the competition uses,
    and the only one that can produce replays or observe the framework's own
    timing. ``FAST`` calls the engine's interpreter directly, roughly 30x
    quicker, for parameter sweeps and large evaluations.

    The default stays ``OFFICIAL`` on purpose: verification should run on the
    same path as the real thing, and ``FAST`` is an opt-in that takes on the
    duty of tracking future ``kaggle-environments`` releases.
    """

    OFFICIAL = "official"
    FAST = "fast"


@dataclass(frozen=True)
class MatchSpec:
    """One episode: a case, an opponent, and a seed.

    ``swap_sides`` runs the case as player 1 instead of player 0. Seat order
    matters (player 0 acts first into the shared market), so a fair evaluation
    plays both seats.
    """

    case: str
    opponent: str
    seed: int
    steps: int = FULL_SEASON_STEPS
    swap_sides: bool = False

    @property
    def case_player(self) -> int:
        """Index the case under test occupies in the episode."""
        return 1 if self.swap_sides else 0

    @property
    def opponent_player(self) -> int:
        return 0 if self.swap_sides else 1

    def agent_order(self) -> tuple[str, str]:
        """The two agent specs in the order ``env.run`` expects them."""
        if self.swap_sides:
            return (self.opponent, self.case)
        return (self.case, self.opponent)

    @property
    def label(self) -> str:
        seat = "p1" if self.swap_sides else "p0"
        return f"{self.case} vs {self.opponent} seed={self.seed} {seat}"


@dataclass(frozen=True)
class SimConfig:
    """Knobs for a whole evaluation run."""

    case: str
    opponents: tuple[str, ...]
    episodes: int = 1
    steps: int = FULL_SEASON_STEPS
    seed_base: int = 0
    swap_sides: bool = False
    workers: int = 1
    engine: Engine = Engine.OFFICIAL
    replay_mode: ReplayMode = ReplayMode.NONE
    output_dir: Path | None = None
    validate_actions: bool = True
    strict: bool = False
    act_timeout: float = ACT_TIMEOUT_SEC
    extra_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The fast engine keeps no step history — that is precisely what makes
        # it fast — so it cannot produce replays. Failing here beats silently
        # writing nothing and leaving the caller to wonder where the files went.
        if self.engine is Engine.FAST and self.replay_mode is not ReplayMode.NONE:
            raise ValueError(
                "replays require engine=official; the fast engine keeps no step history"
            )

    @property
    def slow_turn_threshold(self) -> float:
        return self.act_timeout * SLOW_TURN_FRACTION

    def iter_specs(self) -> Iterator[MatchSpec]:
        """Expand into one :class:`MatchSpec` per episode.

        Seeds are ``seed_base + i`` so a run is reproducible from the two
        numbers recorded in the summary. With ``swap_sides`` every seed is
        played twice, once from each seat, so the pairing stays balanced.
        """
        for opponent in self.opponents:
            for i in range(self.episodes):
                seed = self.seed_base + i
                yield MatchSpec(
                    case=self.case,
                    opponent=opponent,
                    seed=seed,
                    steps=self.steps,
                    swap_sides=False,
                )
                if self.swap_sides:
                    yield MatchSpec(
                        case=self.case,
                        opponent=opponent,
                        seed=seed,
                        steps=self.steps,
                        swap_sides=True,
                    )

    def specs(self) -> list[MatchSpec]:
        return list(self.iter_specs())
