"""Agent spec resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulate.agents import (
    BUILTIN_AGENTS,
    AgentResolutionError,
    is_builtin,
    resolve_agent,
)


@pytest.mark.parametrize("name", BUILTIN_AGENTS)
def test_builtin_names_resolve_to_themselves(name: str) -> None:
    assert resolve_agent(name) == name
    assert is_builtin(name)


def test_file_path_resolves_to_absolute_path(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text("def agent(obs):\n    return {}\n", encoding="utf-8")

    resolved = resolve_agent(str(main_py))

    assert resolved == str(main_py.resolve())
    assert Path(resolved).is_absolute()


def test_unknown_spec_raises() -> None:
    # An unresolvable spec must fail loudly: the engine would silently swap in
    # a no-op agent and the episode would read as a legitimate loss.
    with pytest.raises(AgentResolutionError, match="not a builtin agent"):
        resolve_agent("rulebase/case_does_not_exist")


def test_unknown_spec_is_not_builtin() -> None:
    assert not is_builtin("rulebase/case1")
