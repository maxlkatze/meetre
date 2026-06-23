"""Full uninstall: remove meetre's login item, CLI shim, config/cache and the
install folder itself.

The tricky part is that we run *from inside* the install folder (the venv /
local runtime live there) and ``launchctl`` will SIGTERM this very process when
the login item is unloaded. So the destructive steps can't run in-process —
``run()`` hands them to a short detached ``/bin/sh`` that waits for us to quit,
then tears everything down (including the folder we were running from).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List


def install_root() -> Path:
    """The meetre install/checkout folder (src/meetre/uninstall.py -> parents[2])."""
    return Path(__file__).resolve().parents[2]


def _cli_shim() -> Path:
    return Path(os.path.expanduser("~/.local/bin/meetre"))


def targets() -> List[Path]:
    """Everything ``run()`` will delete, for showing the user before they confirm."""
    from . import autostart, config

    paths: List[Path] = []
    root = install_root()
    shim = _cli_shim()
    # Only claim the CLI shim if it actually points back into this install — a
    # user may have their own `meetre` on PATH from elsewhere.
    try:
        if shim.is_symlink() and os.path.realpath(shim).startswith(str(root) + os.sep):
            paths.append(shim)
    except OSError:
        pass
    paths.append(autostart.PLIST_PATH)        # login item
    paths.append(config.CONFIG_DIR)           # ~/.config/meetre
    paths.append(Path(os.path.expanduser("~/.cache/meetre")))  # logs
    paths.append(root)                        # the install folder (last)
    return paths


def run() -> None:
    """Uninstall meetre. Spawns a detached cleanup that survives this process so
    it can stop the login item and delete the folder we're running from.

    The caller should quit the app right after calling this.
    """
    from . import autostart

    root = install_root()
    plist = autostart.PLIST_PATH
    uid = os.getuid()
    rm_paths = targets()

    quoted = " ".join(shlex.quote(str(p)) for p in rm_paths)
    # sleep lets the app finish quitting first; bootout/unload stop the login
    # item (either form may be a no-op depending on macOS version); then we
    # delete the plist and every target path, including the install folder.
    script = (
        "sleep 1; "
        f"launchctl bootout gui/{uid}/{shlex.quote(autostart.LABEL)} 2>/dev/null; "
        f"launchctl unload {shlex.quote(str(plist))} 2>/dev/null; "
        f"rm -rf {quoted}"
    )
    subprocess.Popen(
        ["/bin/sh", "-c", script],
        start_new_session=True,  # detach so it outlives this process
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd="/",  # never hold a handle on the folder we're about to delete
    )
