"""probe/case2 -- hierarchical-package probe, PASS-only agent.

Verifies, on the real Kaggle harness, the four things a hierarchical case
needs (each step is reported independently so a partial failure is still
diagnosable from ``kaggle competitions logs``):

1. the tar.gz directory hierarchy survives extraction (``pkg/sub`` importable)
2. ``main.py`` can absolute-import a subpackage (``import pkg.core``)
3. relative imports work *inside* the package (``from .util``, ``from ..util``)
4. a data file inside the package loads via ``Path(__file__)`` of a pkg module
   (main.py itself has no ``__file__`` under the harness exec)
"""

import json
import traceback
from typing import Any

RESULTS: dict[str, str] = {}

try:
    from pkg.core import weights_fingerprint

    RESULTS["absolute_import_and_intra_pkg_relative"] = "ok"
    try:
        RESULTS["data_file_via_pkg_file"] = f"ok: {weights_fingerprint()}"
    except Exception:  # noqa: BLE001
        RESULTS["data_file_via_pkg_file"] = traceback.format_exc(limit=2)
except Exception:  # noqa: BLE001
    RESULTS["absolute_import_and_intra_pkg_relative"] = traceback.format_exc(limit=2)

try:
    from pkg.sub.deep import doubled

    RESULTS["nested_subpackage_parent_relative"] = f"ok: doubled(21)={doubled(21)}"
except Exception:  # noqa: BLE001
    RESULTS["nested_subpackage_parent_relative"] = traceback.format_exc(limit=2)

print("KAGGRICULTURE_PROBE2 " + json.dumps(RESULTS), flush=True)


SAFE_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    try:
        n_hands = len((obs.get("farms", [{}])[obs.get("player", 0)]).get("hands", []))
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}
    except Exception:  # noqa: BLE001
        return SAFE_ACTION
