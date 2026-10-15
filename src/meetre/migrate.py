"""Move existing installs from the legacy Python menu bar to Meetre.app.

Existing users never re-run install.sh: the old menu bar pulls updates itself
and restarts. So on startup it calls :func:`to_native_app`, which either hands
over to an installed Meetre.app right away, or runs install.sh detached in the
background — that builds the app into ~/Applications, removes the old login
item and Launchpad entries, quits this process and opens the app (whose
onboarding takes over).

Absolute tool paths throughout: under launchd PATH may be empty.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_LOG = Path(os.path.expanduser("~/.cache/meetre/migrate.log"))
_MARK = Path(os.path.expanduser("~/.cache/meetre/migrate.attempt"))
_RETRY_AFTER = 24 * 3600  # a failed build is retried at most once a day


def _swift_available() -> bool:
    try:
        return subprocess.run(["/usr/bin/xcrun", "--find", "swift"],
                              capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def to_native_app() -> bool:
    """Hand over to Meetre.app. True when this legacy process should exit."""
    from . import bundle

    if os.environ.get("MEETRE_LEGACY") == "1":  # explicit opt-out for debugging
        return False
    from . import instances

    if instances.busy():  # never swap apps under a running recording
        return False
    app = Path.home() / "Applications" / "Meetre.app"
    if (app / "Contents" / "MacOS" / "Meetre").exists():
        subprocess.run(["/usr/bin/open", str(app)])
        return True

    root = bundle._root()
    script = root / "install.sh"
    if not script.exists() or not (root / "mac" / "build.sh").exists():
        return False  # an old checkout without the native app
    if _MARK.exists() and time.time() - _MARK.stat().st_mtime < _RETRY_AFTER:
        return False  # tried recently and it didn't finish; keep running
    if not _swift_available():
        return False  # can't build; the legacy app keeps working

    _LOG.parent.mkdir(parents=True, exist_ok=True)
    _MARK.touch()
    env = {**os.environ,
           "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
           "PYTHONPATH": ""}
    log = open(_LOG, "a")
    log.write(f"\n=== migrate {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    # Detached: install.sh quits this process (pkill meetre.menubar) near
    # the end and must survive that.
    subprocess.Popen(["/bin/bash", str(script)], cwd=str(root), env=env,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     start_new_session=True)
    return False  # keep serving until the installer swaps us out
