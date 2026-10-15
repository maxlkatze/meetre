"""Self-update via ``git pull``.

Pulls the latest commits for the meetre checkout and reinstalls if the code
changed. Used by the menu-bar app ("Check for updates" + optional auto-update
on launch) and the install script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def repo_root() -> Optional[Path]:
    """The git checkout containing meetre, or None if not a git repo."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _git() -> Optional[str]:
    """Path to a usable git: one from PATH, else a locally-installed one."""
    found = shutil.which("git")
    if found:
        return found
    root = repo_root()
    if root:
        for cand in (root / ".runtime" / "conda" / "bin" / "git",
                     root / ".runtime" / "git" / "bin" / "git"):
            if cand.exists():
                return str(cand)
    return None


def _run(git: str, root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([git, "-C", str(root), *args], capture_output=True, text=True)


def head() -> Optional[str]:
    """Commit currently checked out (None if unknown)."""
    git, root = _git(), repo_root()
    if not git or not root:
        return None
    return _run(git, root, "rev-parse", "HEAD").stdout.strip() or None


def check(since: Optional[str] = None) -> dict:
    """Fetch and report whether the upstream branch has commits not in ``since``.

    ``since`` is the commit the running app was started from (default: HEAD),
    so an update that was already pulled but not yet restarted into still
    counts as available. Returns ``{available, behind, error}``.
    """
    git, root = _git(), repo_root()
    if not git or not root or not has_remote():
        return {"available": False, "behind": 0, "error": "no git checkout/remote"}
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never block on credentials
    try:
        fetch = subprocess.run([git, "-C", str(root), "fetch", "--quiet"],
                               capture_output=True, text=True, env=env, timeout=60)
    except subprocess.TimeoutExpired:
        return {"available": False, "behind": 0, "error": "git fetch timed out"}
    if fetch.returncode != 0:
        return {"available": False, "behind": 0,
                "error": fetch.stderr.strip() or "git fetch failed"}
    count = _run(git, root, "rev-list", "--count", f"{since or 'HEAD'}..@{{u}}")
    if count.returncode != 0:
        return {"available": False, "behind": 0, "error": count.stderr.strip()}
    behind = int(count.stdout.strip() or 0)
    return {"available": behind > 0, "behind": behind, "error": None}


def venv_python(root: Path) -> str:
    """The venv interpreter. Never ``sys.executable``: inside meetre.app that is
    the bundle symlink, which runs as the *base* runtime python (it can't see
    the venv's pyvenv.cfg) — pip there uninstalls meetre from the venv (taking
    .venv/bin/meetre and the login item's import with it) and installs it into
    .runtime/python instead."""
    for name in ("python3.12", "python3", "python"):
        cand = root / ".venv" / "bin" / name
        if cand.exists():
            return str(cand)
    return sys.executable


def reinstall_editable(root: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [venv_python(root), "-m", "pip", "install", "-q", "-e", f"{root}[menubar]"],
        capture_output=True, text=True, env=env,
    )


def has_remote() -> bool:
    git, root = _git(), repo_root()
    if not git or not root:
        return False
    return bool(_run(git, root, "remote").stdout.strip())


def update(reinstall: bool = True) -> dict:
    """Fast-forward pull the checkout.

    Returns ``{updated, before, after, error}``. ``updated`` is True only when
    new commits arrived. Reinstalls the package when the code changed.
    """
    git, root = _git(), repo_root()
    if not git:
        return {"updated": False, "error": "git not found"}
    if not root:
        return {"updated": False, "error": "not a git checkout"}
    if not has_remote():
        return {"updated": False, "error": "no git remote configured"}

    before = _run(git, root, "rev-parse", "HEAD").stdout.strip()
    pull = _run(git, root, "pull", "--ff-only")
    if pull.returncode != 0:
        return {"updated": False, "error": pull.stderr.strip() or "git pull failed"}
    after = _run(git, root, "rev-parse", "HEAD").stdout.strip()

    updated = before != after
    # meetre is an editable install, so new code is live after the pull; only
    # dependency / entry-point changes need pip.
    changed = _run(git, root, "diff", "--name-only", before, after).stdout.split()
    if updated and reinstall and any(f in ("pyproject.toml", "setup.py", "setup.cfg")
                                     for f in changed):
        reinstall_editable(root)
    return {"updated": updated, "before": before, "after": after, "error": None}
