# probe/case2 — hierarchical-package probe

**Not a strategy agent.** PASS-only agent verifying that a case with a
subpackage hierarchy survives packaging and the Kaggle harness:

1. tar.gz hierarchy survives extraction (`pkg/sub` importable)
2. `main.py` absolute-imports a subpackage (`import pkg.core`)
3. relative imports inside the package (`from .util`, `from ..util`)
4. data file loaded via `Path(__file__)` of a package module
   (`pkg/data/weights.npy` — main.py itself has no `__file__` under the
   harness exec)

Each check reports independently, prefixed `KAGGRICULTURE_PROBE2`, retrievable
via `kaggle competitions logs`.

Note: `build_archive` currently strips `__init__.py` (EXCLUDE_NAMES), so in
the archive `pkg` becomes a namespace package — whether that still supports
relative imports on the harness is exactly what this probe measures.
