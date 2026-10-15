"""Find, replace and restart running menu-bar instances.

Everything here uses absolute tool paths: under launchd ("start at login") or
after a restart ``PATH`` may be empty (see CLAUDE.md).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import List, Optional

_LOG = os.path.expanduser("~/.cache/meetre/menubar.err.log")


def others() -> List[int]:
    """PIDs of this user's other meetre menu-bar processes."""
    try:
        out = subprocess.run(["/bin/ps", "-xo", "pid=,command="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    me = {os.getpid(), os.getppid()}
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        # The interpreter itself (venv python or the meetre.app symlink) — not
        # any shell/grep whose command line merely mentions the module.
        exe = os.path.basename(argv[0]) if argv else ""
        if (argv[1:3] == ["-m", "meetre.menubar"]
                and (exe.startswith("python") or exe == "meetre")
                and pid.isdigit() and int(pid) not in me):
            pids.append(int(pid))
    return pids


def terminate(pids: List[int], timeout: float = 4.0) -> None:
    """SIGTERM ``pids``, then SIGKILL whatever is still alive after ``timeout``."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    alive = list(pids)
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = [p for p in alive if _alive(p)]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def is_launchd_job() -> bool:
    """True when this process is the "start at login" LaunchAgent job."""
    from . import autostart

    return os.environ.get("XPC_SERVICE_NAME") == autostart.LABEL


def launch_command() -> List[str]:
    """Command that starts a fresh menu-bar instance.

    Prefers the LaunchAgent (the exact login launch, so it stays the managed
    job); otherwise the meetre.app bundle with the venv on PYTHONPATH.
    """
    from . import autostart, bundle

    if autostart.is_enabled():
        return ["/bin/launchctl", "kickstart", f"gui/{os.getuid()}/{autostart.LABEL}"]
    exe = bundle.launch_args() or [str(bundle._venv_python()), "-m", "meetre.menubar"]
    return ["/usr/bin/env", f"PYTHONPATH={bundle._pythonpath()}", *exe]


def relaunch_after_exit(pid: int) -> None:
    """Spawn a detached watcher that starts meetre once ``pid`` has exited.

    Waiting for the old process to be fully gone avoids two instances fighting
    over the status item / audio devices, which is what made restarts flaky.
    If ``pid`` hangs for 15 s it is killed.
    """
    script = (
        'pid="$1"; shift; i=0; '
        'while /bin/kill -0 "$pid" 2>/dev/null; do '
        '  i=$((i+1)); [ "$i" -ge 75 ] && /bin/kill -9 "$pid" 2>/dev/null; '
        '  /bin/sleep 0.2; '
        'done; exec "$@"'
    )
    os.makedirs(os.path.dirname(_LOG), exist_ok=True)
    log = open(_LOG, "a")
    subprocess.Popen(["/bin/sh", "-c", script, "sh", str(pid), *launch_command()],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     start_new_session=True, close_fds=True)


def app_path():
    """The installed Meetre.app (``~/Applications``), else the repo build."""
    from pathlib import Path

    from . import bundle

    for cand in (Path.home() / "Applications" / "Meetre.app",
                 bundle._root() / "mac" / "build" / "Meetre.app"):
        if (cand / "Contents" / "MacOS" / "Meetre").exists():
            return cand
    return None


def app_pids() -> List[int]:
    """PIDs of running Meetre.app processes (the native app)."""
    try:
        out = subprocess.run(["/bin/ps", "-xo", "pid=,command="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    return [int(line.split(None, 1)[0]) for line in out.splitlines()
            if "/Meetre.app/Contents/MacOS/Meetre" in line and line.split(None, 1)[0].isdigit()]


def busy() -> Optional[str]:
    """What meetre is doing if a recording is running/processing, else None.

    The engine keeps ~/.cache/meetre/busy while recording or processing
    (content: "<engine pid> <what>"); a stale file from a dead engine is
    ignored."""
    from pathlib import Path

    f = Path(os.path.expanduser("~/.cache/meetre/busy"))
    try:
        pid, _, what = f.read_text().strip().partition(" ")
    except OSError:
        return None
    return (what or "recording") if pid.isdigit() and _alive(int(pid)) else None


def restart_from_outside() -> None:
    """``meetre restart``: stop every running instance, then start one."""
    doing = busy()
    if doing:
        raise SystemExit(f"meetre is busy ({doing}) — stop the recording first, then restart.")
    app = app_path()
    if app is not None:
        terminate(app_pids() + others())  # the app also stops its engine
        subprocess.run(["/usr/bin/open", str(app)])
        return
    terminate(others())
    from . import autostart

    if autostart.is_enabled():
        # -k also covers a job launchd still thinks is running.
        subprocess.run(["/bin/launchctl", "kickstart", "-k",
                        f"gui/{os.getuid()}/{autostart.LABEL}"],
                       capture_output=True, text=True)
        return
    log = open(_LOG, "a")
    subprocess.Popen(launch_command(), stdin=subprocess.DEVNULL, stdout=log,
                     stderr=log, start_new_session=True)
