"""Manage a macOS LaunchAgent so the menu-bar app starts at login.

No admin needed — LaunchAgents live in the user's ``~/Library/LaunchAgents``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

LABEL = "com.agendaro.meetre.menubar"
_AGENT_DIR = Path(os.path.expanduser("~/Library/LaunchAgents"))
PLIST_PATH = _AGENT_DIR / f"{LABEL}.plist"
_LOG_DIR = Path(os.path.expanduser("~/.cache/meetre"))


def is_enabled() -> bool:
    return PLIST_PATH.exists()


def app_target() -> Optional[str]:
    """The Meetre.app the login item launches, or None for the Python menu bar."""
    if not is_enabled():
        return None
    m = re.search(r"<string>(.+?\.app)/Contents/MacOS/Meetre</string>", PLIST_PATH.read_text())
    return m.group(1) if m else None


def _plist_xml(app: Optional[str] = None) -> str:
    # Always the venv interpreter — sys.executable may be the meetre.app
    # bundle symlink, which can't import meetre without our PYTHONPATH.
    from .bundle import _venv_python

    python = str(_venv_python())
    if app:  # the native app (it starts the engine itself)
        exe = Path(app) / "Contents" / "MacOS" / "Meetre"
        args = f"    <string>{exe}</string>"
    else:
        args = (f"    <string>{python}</string>\n    <string>-m</string>\n"
                "    <string>meetre.menubar</string>")
    # Include common dirs + a bundled local git on PATH for auto-update.
    root = Path(__file__).resolve().parents[2]
    path = (f"{Path(python).parent}:{root}/.runtime/conda/bin:{root}/.runtime/git/bin:"
            "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
    out_log = _LOG_DIR / "menubar.out.log"
    err_log = _LOG_DIR / "menubar.err.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>{path}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{out_log}</string>
  <key>StandardErrorPath</key><string>{err_log}</string>
</dict>
</plist>
"""


def enable(app: Optional[str] = None) -> None:
    """Install + load the LaunchAgent (starts now and at every login).

    ``app`` = path to Meetre.app to start that instead of the Python menu bar."""
    _AGENT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(_plist_xml(app))
    # Reload so changes take effect immediately.
    subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                   capture_output=True, text=True)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)],
                   capture_output=True, text=True)


def repair() -> None:
    """Rewrite a stale/broken login item in place (picked up at next login)."""
    if is_enabled():
        app = app_target()  # keep whichever app the login item targets
        if PLIST_PATH.read_text() != _plist_xml(app):
            PLIST_PATH.write_text(_plist_xml(app))


def disable() -> None:
    """Unload + remove the LaunchAgent."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                       capture_output=True, text=True)
        PLIST_PATH.unlink(missing_ok=True)
