"""probe/case1 -- runtime-environment probe, PASS-only agent.

Prints (to stdout, retrievable via ``kaggle competitions logs``) everything we
need to know about how the harness loads ``main.py``: interpreter version,
``cwd``, ``sys.path``, which of ``__name__`` / ``__file__`` / ``__package__``
exist in globals, and which third-party libraries are importable.

The agent itself always returns the legal no-op, so any episode failure is
attributable to the probe prints, not to farming logic.
"""

import json
import sys
from pathlib import Path
from typing import Any


def _probe_environment() -> dict[str, Any]:
    g = globals()
    libs = {}
    for lib in ("numpy", "polars", "pandas", "scipy", "torch", "kaggle_environments"):
        try:
            mod = __import__(lib)
            libs[lib] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            libs[lib] = f"UNAVAILABLE: {type(exc).__name__}"
    return {
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "sys_path": sys.path,
        "argv": sys.argv,
        "globals_name": g.get("__name__", "<absent>"),
        "globals_file": g.get("__file__", "<absent>"),
        "globals_package": g.get("__package__", "<absent>"),
        "libs": libs,
    }


try:
    print("KAGGRICULTURE_PROBE " + json.dumps(_probe_environment()), flush=True)
except Exception as exc:  # noqa: BLE001 - probe must never raise
    print(f"KAGGRICULTURE_PROBE_FAILED {type(exc).__name__}: {exc}", flush=True)


SAFE_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    try:
        n_hands = len((obs.get("farms", [{}])[obs.get("player", 0)]).get("hands", []))
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}
    except Exception:  # noqa: BLE001
        return SAFE_ACTION
