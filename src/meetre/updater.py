"""Self-update via ``git pull``.

Pulls the latest commits for the meetre checkout and reinstalls if the code
changed. Used by the menu-bar app ("Check for updates" + optional auto-update
on launch) and the install script.
"""

from __future__ import annotations

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
    """Path to a usable git: a bundled local one, else from PATH."""
    root = repo_root()
    if root:
        local = root / ".runtime" / "git" / "bin" / "git"
        if local.exists():
            return str(local)
    return shutil.which("git")


def _run(git: str, root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([git, "-C", str(root), *args], capture_output=True, text=True)


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
    if updated and reinstall:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(root)],
            capture_output=True, text=True,
        )
    return {"updated": updated, "before": before, "after": after, "error": None}
