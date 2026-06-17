"""Manage a macOS LaunchAgent so the menu-bar app starts at login.

No admin needed — LaunchAgents live in the user's ``~/Library/LaunchAgents``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LABEL = "com.agendaro.meetre.menubar"
_AGENT_DIR = Path(os.path.expanduser("~/Library/LaunchAgents"))
PLIST_PATH = _AGENT_DIR / f"{LABEL}.plist"
_LOG_DIR = Path(os.path.expanduser("~/.cache/meetre"))


def is_enabled() -> bool:
    return PLIST_PATH.exists()


def _plist_xml() -> str:
    # Run the menu-bar app with the same interpreter that installed it.
    python = sys.executable
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
    <string>{python}</string>
    <string>-m</string>
    <string>meetre.menubar</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>{path}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{out_log}</string>
  <key>StandardErrorPath</key><string>{err_log}</string>
</dict>
</plist>
"""


def enable() -> None:
    """Install + load the LaunchAgent (starts now and at every login)."""
    _AGENT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(_plist_xml())
    # Reload so changes take effect immediately.
    subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                   capture_output=True, text=True)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)],
                   capture_output=True, text=True)


def disable() -> None:
    """Unload + remove the LaunchAgent."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                       capture_output=True, text=True)
        PLIST_PATH.unlink(missing_ok=True)
