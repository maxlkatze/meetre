"""A minimal ``meetre.app`` wrapper so macOS attributes the app correctly.

Run as ``python -m meetre.menubar`` the process has no app bundle, so the
notification center (and the app switcher) call it "python3.12" with a generic
icon. macOS derives that name/icon/identifier from the *main bundle* of the
running executable — so if we launch the same interpreter from inside a tiny
``meetre.app`` (an ``Info.plist`` plus a symlink to the venv python), everything
is attributed to "meetre" with our icon and the ``net.cubedpixels.meetre`` id.

The bundle is generated on demand next to the install; :func:`relaunch_into_bundle`
re-execs into it once at startup. If anything fails we just keep running in
place (still works, only the cosmetic attribution is lost).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

BUNDLE_ID = "net.cubedpixels.meetre"
_MARKER = "/meetre.app/Contents/MacOS/"


def _root() -> Path:
    # src/meetre/bundle.py -> parents[2] == repo root (editable install)
    return Path(__file__).resolve().parents[2]


def app_path() -> Path:
    return _root() / "meetre.app"


def running_inside_bundle() -> bool:
    return _MARKER in (sys.executable or "")


def _venv_python() -> Path:
    """Stable path to the venv interpreter, regardless of how we were launched."""
    binp = _root() / ".venv" / "bin"
    for c in ("python3.12", "python3", "python"):
        p = binp / c
        if p.exists():
            return p
    return Path(sys.executable)


def _pythonpath() -> str:
    """venv site-packages + src so the bundle's python finds deps and meetre.

    PYTHONPATH dirs aren't scanned for .pth files, so the editable-install path
    (src) must be added explicitly alongside site-packages.
    """
    parts = [str(p) for p in (_root() / ".venv" / "lib").glob("python*/site-packages")]
    parts.append(str(_root() / "src"))
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return ":".join(parts)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("meetre")
    except Exception:  # noqa: BLE001
        return "0"


def _info_plist() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>meetre</string>
  <key>CFBundleDisplayName</key><string>meetre</string>
  <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
  <key>CFBundleExecutable</key><string>meetre</string>
  <key>CFBundleIconFile</key><string>meetre</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>{_version()}</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""


def _write_icns(resources: Path) -> None:
    """Render the app icon into the bundle (best-effort).

    sips only converts standard icon sizes to .icns, so resize the source PNG to
    512×512 first, then convert.
    """
    try:
        from . import icon

        png = icon.icon_path()
        if not png:
            return
        big = resources / "_icon512.png"
        subprocess.run(["sips", "-z", "512", "512", png, "--out", str(big)],
                       capture_output=True, timeout=20)
        if big.exists():
            subprocess.run(
                ["sips", "-s", "format", "icns", str(big),
                 "--out", str(resources / "meetre.icns")],
                capture_output=True, timeout=20,
            )
            try:
                big.unlink()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def ensure_bundle() -> Path:
    """Create/refresh meetre.app and return the path to its executable symlink."""
    app = app_path()
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    exe = macos / "meetre"
    target = _venv_python()
    try:
        if exe.is_symlink() or exe.exists():
            exe.unlink()
        exe.symlink_to(target)
    except Exception:  # noqa: BLE001
        pass

    _write_icns(resources)
    (app / "Contents" / "Info.plist").write_text(_info_plist())
    return exe


def launch_args() -> Optional[list]:
    """ProgramArguments to launch meetre from the bundle (for the LaunchAgent),
    or None if the bundle can't be built."""
    try:
        exe = ensure_bundle()
        return [str(exe), "-m", "meetre.menubar"]
    except Exception:  # noqa: BLE001
        return None


def relaunch_into_bundle() -> None:
    """Re-exec the menu bar from inside meetre.app for correct attribution.

    No-op (returns) if already inside the bundle or if it can't be built — the
    caller then just runs in place.
    """
    if running_inside_bundle():
        return
    try:
        exe = ensure_bundle()
        env = dict(os.environ)
        env["PYTHONPATH"] = _pythonpath()
        os.execve(str(exe), [str(exe), "-m", "meetre.menubar"], env)
    except Exception:  # noqa: BLE001
        pass  # fall through: run in place, attributed to the interpreter
