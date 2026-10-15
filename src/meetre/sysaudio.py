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
# Echo-cancelled microphone (macOS Voice Processing), same build/sign path.
_MIC_SRC = Path(__file__).parent / "helpers" / "miccap.swift"
_MIC_BINARY = _CACHE_DIR / "miccap"


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


def _needs_build(src: Path = _SRC, binary: Path = _BINARY) -> bool:
    if not binary.exists():
        return True
    # Rebuild if the Swift source has changed since the last compile.
    return src.stat().st_mtime > binary.stat().st_mtime


def helper_path(force: bool = False) -> Path:
    """Return the path to the compiled ``syscap`` binary, building if needed.

    Raises ``RuntimeError`` with an actionable message if the source or the
    Swift toolchain is missing, or if compilation fails.
    """
    return _build(_SRC, _BINARY, force)


def mic_helper_path(force: bool = False) -> Path:
    """The compiled ``miccap`` (echo-cancelled mic) binary, building if needed."""
    return _build(_MIC_SRC, _MIC_BINARY, force)


def _build(_SRC: Path, _BINARY: Path, force: bool) -> Path:
    if not _SRC.exists():
        raise RuntimeError(f"audio helper source missing: {_SRC}")
    if not (force or _needs_build(_SRC, _BINARY)):
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
            f"Failed to compile {_SRC.name}:\n" + (proc.stderr or proc.stdout)
        )
    _sign(_BINARY, f"net.cubedpixels.meetre.{_BINARY.name}")
    return _BINARY


_IDENTITY = "meetre Local Signing"


def _sign(binary: Path, identifier: str = "net.cubedpixels.meetre.syscap") -> None:
    """Sign with the stable local identity (mac/Scripts/signing-identity.sh)
    when it exists — an ad-hoc signature changes with every compile, which
    makes macOS forget the capture permission. Best-effort."""
    try:
        has = subprocess.run(["/usr/bin/security", "find-certificate", "-c", _IDENTITY],
                             capture_output=True, timeout=10).returncode == 0
        if has:
            subprocess.run(["/usr/bin/codesign", "--force", "--sign", _IDENTITY,
                            "--identifier", identifier, str(binary)],
                           capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass
