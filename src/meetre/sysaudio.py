"""Native macOS system-audio capture via a bundled ScreenCaptureKit helper.

macOS does not let an app read the system output directly, but
**ScreenCaptureKit** (macOS 13+) can capture it natively — no BlackHole or
other loopback driver required. We ship a tiny Swift CLI (``helpers/syscap.swift``)
that does exactly that; this module compiles it on first use and hands back the
path to the binary so :mod:`meetre.recorder` can run it as a capture source.

System-audio capture is gated behind the **Screen Recording** permission
(System Settings ▸ Privacy & Security ▸ Screen Recording) — the same one used
for screen capture. The first attempt prompts for it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).parent / "helpers" / "syscap.swift"
_CACHE_DIR = Path(os.path.expanduser("~/.cache/meetre"))
_BINARY = _CACHE_DIR / "syscap"


def swiftc_path() -> Optional[str]:
    """Locate the Swift compiler, or None if the toolchain is absent."""
    found = shutil.which("swiftc")
    if found:
        return found
    # Command Line Tools install swiftc under xcrun but not always on PATH.
    try:
        out = subprocess.run(
            ["xcrun", "--find", "swiftc"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def available() -> bool:
    """True if native system-audio capture can be built on this machine."""
    return _SRC.exists() and swiftc_path() is not None


def _needs_build() -> bool:
    if not _BINARY.exists():
        return True
    # Rebuild if the Swift source has changed since the last compile.
    return _SRC.stat().st_mtime > _BINARY.stat().st_mtime


def helper_path(force: bool = False) -> Path:
    """Return the path to the compiled ``syscap`` binary, building if needed.

    Raises ``RuntimeError`` with an actionable message if the source or the
    Swift toolchain is missing, or if compilation fails.
    """
    if not _SRC.exists():
        raise RuntimeError(f"system-audio helper source missing: {_SRC}")
    if not (force or _needs_build()):
        return _BINARY

    swiftc = swiftc_path()
    if swiftc is None:
        raise RuntimeError(
            "Swift compiler not found. Install the Xcode Command Line Tools:\n"
            "  xcode-select --install"
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [swiftc, "-O", str(_SRC), "-o", str(_BINARY)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not _BINARY.exists():
        raise RuntimeError(
            "Failed to compile the system-audio helper:\n" + (proc.stderr or proc.stdout)
        )
    return _BINARY
