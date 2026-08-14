# probe/case1 — runtime-environment probe

**Not a strategy agent.** PASS-only agent whose sole purpose is to print, at
import time, how the Kaggle harness loads `main.py`: interpreter version, cwd,
`sys.path`, presence of `__name__` / `__file__` / `__package__` in globals, and
which third-party libraries are importable. Output is prefixed
`KAGGRICULTURE_PROBE` and retrieved via `kaggle competitions logs`.

Submitted once to establish the ground truth that `verify_archive` and the
import rules in `.claude/rules/python.md` must mirror. See
`docs/experiment/` submit-infrastructure notes.
