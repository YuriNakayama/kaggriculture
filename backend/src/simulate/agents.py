"""Resolve agent specs into something ``env.run`` accepts.

Three forms are supported, checked in this order:

1. a builtin agent name (``"pass"`` / ``"random"`` / ``"starter"``)
2. a filesystem path to a ``main.py``
3. a case under ``backend/pipeline/`` (``"rulebase/case1"``)
"""

from __future__ import annotations

from pathlib import Path

#: Agents the engine provides by name.
BUILTIN_AGENTS: tuple[str, ...] = ("pass", "random", "starter")

#: backend/ root, i.e. the parent of src/.
BACKEND_ROOT = Path(__file__).resolve().parents[2]

#: Where agent families live.
PIPELINE_ROOT = BACKEND_ROOT / "pipeline"


class AgentResolutionError(ValueError):
    """Raised when an agent spec matches none of the supported forms."""


def resolve_agent(spec: str) -> str:
    """Turn an agent spec into a builtin name or an absolute ``main.py`` path.

    Raises :class:`AgentResolutionError` when the spec cannot be resolved.
    Resolving eagerly matters: a typo'd case name would otherwise reach the
    engine, which substitutes a no-op agent without complaining, and the run
    would look like a legitimate loss.
    """
    if spec in BUILTIN_AGENTS:
        return spec

    candidate = Path(spec)
    if candidate.is_file():
        return str(candidate.resolve())

    main_py = PIPELINE_ROOT / spec / "main.py"
    if main_py.is_file():
        return str(main_py)

    raise AgentResolutionError(
        f"{spec!r} is not a builtin agent ({', '.join(BUILTIN_AGENTS)}), "
        f"an existing file, or a case under backend/pipeline/."
    )


def is_builtin(spec: str) -> bool:
    return spec in BUILTIN_AGENTS


def available_cases() -> list[str]:
    """Every ``<family>/<case>`` under ``backend/pipeline/`` holding a main.py."""
    if not PIPELINE_ROOT.is_dir():
        return []
    cases = [
        f"{main_py.parent.parent.name}/{main_py.parent.name}"
        for main_py in sorted(PIPELINE_ROOT.glob("*/*/main.py"))
    ]
    return cases
