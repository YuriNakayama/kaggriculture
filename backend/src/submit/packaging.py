"""Build and validate Kaggle submission archives.

The archive contract is unforgiving in two specific ways, and both fail *after*
the submission slot is spent:

1. **`main.py` must be at the archive root.** `tar -czf out.tgz <case-dir>`
   nests it one level down and the harness cannot import it.
2. **The archive must be self-contained.** Anything imported from
   `backend/src/**` exists locally but not in the tarball.

:func:`verify_archive` reproduces the harness's loading conditions, which were
measured on the real validation image via ``probe/case1`` / ``probe/case2``
(2026-08-13, submissions 55481573 / 55481654):

- Python **3.11** (local dev is 3.13 — 3.12+ syntax fails only in production)
- ``main.py`` is ``exec``'d with **no** ``__name__`` / ``__file__`` /
  ``__package__`` in globals, and the harness takes the **last** callable
  defined in the module as the agent
- cwd is ``/kaggle/working``, *not* the agent dir; the agent dir
  (``/kaggle_simulations/agent``) is appended to ``sys.path`` — so bare
  relative ``open()`` fails, and data files must be resolved through the
  ``__file__`` of a bundled *package module* (main.py has none)
- subpackage hierarchies work, including relative imports inside packages and
  namespace packages (``__init__.py`` is stripped from the archive)
- available libs: numpy 2.4.6, polars, pandas, scipy, torch 2.6.0+cu124,
  kaggle_environments 1.32.6

The verification episode therefore runs in an **isolated Python 3.11
interpreter** (provisioned by uv) with only ``kaggle-environments`` installed,
from a cwd that is not the unpacked dir.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
PIPELINE_ROOT = BACKEND_ROOT / "pipeline"

#: Submission history lives under the DVC-managed output layer.
HISTORY_DIR = REPO_ROOT / "data" / "output" / "submit"

#: Python version of the Kaggle validation image (measured by probe/case1).
#: Local dev runs 3.13, so verification must run under this version or a
#: 3.12+-only construct passes dry-run and dies in production.
PROD_PYTHON = "3.11"

#: kaggle-environments version on the validation image (measured). Kept in
#: sync with the pin in ``backend/pyproject.toml``.
PROD_KAGGLE_ENVIRONMENTS = "1.32.6"

#: Never ship these into the archive.
#:
#: `train.py` is deliberately excluded: it is training-only code. (The
#: validation image does ship torch — measured 2.6.0+cu124 — but bundling
#: training modules bloats the archive and risks import-time failures for
#: code the agent never calls.) A module needed at inference must not be
#: named `train.py`.
#:
#: `__init__.py` is stripped, so subpackages travel as namespace packages —
#: measured to work on the harness, including relative imports inside them.
EXCLUDE_NAMES = frozenset(
    {"__pycache__", "__init__.py", ".DS_Store", ".pytest_cache", "train.py"}
)

#: Kaggle's per-day submission cap for this competition.
DAILY_SUBMISSION_LIMIT = 5


class SubmitError(RuntimeError):
    """Raised when an archive is invalid or a submission is refused locally."""


@dataclass(frozen=True)
class Archive:
    path: Path
    case: str
    members: tuple[str, ...]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


def case_dir(case: str) -> Path:
    """Resolve ``<family>/<caseN>`` to its pipeline directory."""
    path = PIPELINE_ROOT / case
    if not path.is_dir():
        raise SubmitError(f"no such case directory: {path}")
    if not (path / "main.py").is_file():
        raise SubmitError(f"{case} has no main.py")
    return path


def build_archive(case: str, out_path: Path) -> Archive:
    """Pack a case directory with ``main.py`` at the archive root."""
    source = case_dir(case)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    members: list[str] = []
    with tarfile.open(out_path, "w:gz") as tar:
        for entry in sorted(source.rglob("*")):
            if any(part in EXCLUDE_NAMES for part in entry.relative_to(source).parts):
                continue
            if entry.name in EXCLUDE_NAMES:
                continue
            arcname = str(entry.relative_to(source))
            tar.add(entry, arcname=arcname, recursive=False)
            members.append(arcname)

    if "main.py" not in members:
        raise SubmitError(f"main.py is not at the archive root; members={members[:10]}")
    return Archive(path=out_path, case=case, members=tuple(members))


def verify_archive(archive: Archive, *, steps: int = 720) -> dict[str, object]:
    """Unpack into a temp dir and run one episode the way the harness would.

    Faithful to the measured production conditions (see module docstring):
    Python 3.11 with only ``kaggle-environments`` installed, ``main.py``
    ``exec``'d with no ``__name__`` / ``__file__``, the unpacked dir
    *appended* to ``sys.path``, and a cwd that is **not** the unpacked dir.
    A stray ``backend/src`` import, a relative import in ``main.py``, a bare
    relative ``open()``, or 3.12+-only syntax all surface here rather than
    on Kaggle.
    """
    with tempfile.TemporaryDirectory() as tmp:
        unpacked = Path(tmp) / "agent"  # ~ /kaggle_simulations/agent
        working = Path(tmp) / "working"  # ~ /kaggle/working (harness cwd)
        unpacked.mkdir()
        working.mkdir()
        with tarfile.open(archive.path) as tar:
            tar.extractall(unpacked, filter="data")

        probe = f"""
import json, sys
sys.path.append({str(unpacked)!r})

# the harness execs main.py with no __name__/__file__/__package__ and takes
# the LAST callable defined in the module as the agent
src = open({str(unpacked / "main.py")!r}).read()
env = {{}}
exec(compile(src, "main.py", "exec"), env)
agent = env.get("agent")
assert callable(agent), "main.py must define agent(obs)"
last = [v for v in env.values() if callable(v)][-1]
assert last is agent, f"last callable is {{last!r}}, not agent"

# a full episode against the built-in baseline
from kaggle_environments import make
e = make("kaggriculture", configuration={{"episodeSteps": {steps}}}, debug=True)
e.run([agent, "starter"])
final = e.steps[-1]
print(json.dumps({{
    "rewards": [s["reward"] for s in final],
    "statuses": [s["status"] for s in final],
    "steps": len(e.steps),
    "python": ".".join(map(str, sys.version_info[:3])),
}}))
"""
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                PROD_PYTHON,
                "--with",
                f"kaggle-environments=={PROD_KAGGLE_ENVIRONMENTS}",
                "python",
                "-c",
                probe,
            ],
            cwd=working,
            capture_output=True,
            text=True,
            timeout=1800,
        )

    if completed.returncode != 0:
        raise SubmitError(
            f"archive failed verification:\n{completed.stderr.strip()[-2000:]}"
        )

    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise SubmitError(f"could not parse verification output: {exc}") from exc

    statuses = result.get("statuses") or []
    if statuses and statuses[0] != "DONE":
        raise SubmitError(f"agent did not finish cleanly: status={statuses[0]}")
    return dict(result)


def submissions_today() -> int:
    """How many submissions were recorded locally today (UTC).

    Kaggle enforces the real cap; this is a local guard so an over-quota
    attempt fails immediately instead of producing a confusing API error.
    """
    if not HISTORY_DIR.is_dir():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    for record in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(data.get("submitted_at", "")).startswith(today):
            count += 1
    return count


def git_state() -> tuple[str, bool]:
    """Current HEAD sha and whether the working tree is dirty.

    A dirty tree means the recorded sha does not reproduce the archive, so
    the CLI warns before a real submission and the flag is persisted.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "", False
    return sha, dirty


def record_submission(
    archive: Archive, message: str, verification: dict[str, object]
) -> Path:
    """Append a submission to the local history.

    This is the audit trail linking a leaderboard score back to the exact code
    that produced it, so it records the git sha (and whether the tree was
    dirty at submit time) alongside the case.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sha, dirty = git_state()

    payload = {
        "case": archive.case,
        "message": message,
        "git_sha": sha,
        "git_dirty": dirty,
        "submitted_at": now.isoformat(),
        "archive_bytes": archive.size_bytes,
        "members": list(archive.members),
        "verification": verification,
    }
    stem = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{archive.case.replace('/', '_')}"
    path = HISTORY_DIR / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def submit_to_kaggle(archive: Archive, message: str) -> str:
    """Upload the archive. Returns the CLI's stdout."""
    completed = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "kaggriculture",
            "-f",
            str(archive.path),
            "-m",
            message,
        ],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    if completed.returncode != 0:
        raise SubmitError(f"kaggle submit failed:\n{completed.stderr.strip()}")
    return completed.stdout.strip()
