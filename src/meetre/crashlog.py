"""Crash + error logging for the menu-bar app.

The risky failures in a pyobjc/rumps app are *native* — a segfault from an
over-release or a bad AppKit call kills the process before a normal Python
traceback can print. ``faulthandler`` still dumps the Python stack on a fatal
signal, so we point it at a file; uncaught Python exceptions (main thread and
worker threads) are logged too. Everything lands in ``<repo>/crashlogs/`` so a
crash leaves a diagnosable trail instead of vanishing.
"""

from __future__ import annotations

import datetime as _dt
import faulthandler
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

_DIR: Optional[Path] = None
_fault_fp = None  # keep the faulthandler file open for the whole session
_installed = False


def crashlog_dir() -> Path:
    """The directory crash logs are written to (created on first use).

    Prefers ``<repo>/crashlogs`` (this package is installed editable from the
    repo); falls back to the config dir if the repo isn't writable.
    """
    global _DIR
    if _DIR is not None:
        return _DIR
    # src/meetre/crashlog.py -> parents[2] == repo root
    candidate = Path(__file__).resolve().parents[2] / "crashlogs"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        _DIR = candidate
    except Exception:  # noqa: BLE001
        from .config import CONFIG_DIR

        _DIR = CONFIG_DIR / "crashlogs"
        _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_exception(kind, exc_type, exc, tb) -> None:
    try:
        path = crashlog_dir() / f"crash-{_ts()}.log"
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(f"=== {kind} at {_dt.datetime.now().isoformat()} ===\n")
            traceback.print_exception(exc_type, exc, tb, file=fp)
            fp.write("\n")
    except Exception:  # noqa: BLE001
        pass


def install() -> None:
    """Idempotently install fault + exception handlers."""
    global _fault_fp, _installed
    if _installed:
        return
    _installed = True

    # Native fatal signals (SIGSEGV/SIGABRT/SIGBUS/…): dump the Python stack.
    try:
        _fault_fp = open(crashlog_dir() / "faulthandler.log", "a", encoding="utf-8")
        _fault_fp.write(f"\n=== session start {_dt.datetime.now().isoformat()} ===\n")
        _fault_fp.flush()
        faulthandler.enable(file=_fault_fp, all_threads=True)
    except Exception:  # noqa: BLE001
        pass

    # Uncaught exceptions on the main thread.
    _prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        _write_exception("uncaught exception", exc_type, exc, tb)
        _prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook

    # Uncaught exceptions on worker threads (Python 3.8+).
    if hasattr(threading, "excepthook"):
        _prev_thread = threading.excepthook

        def _thook(args):
            _write_exception(
                f"thread '{getattr(args.thread, 'name', '?')}' exception",
                args.exc_type, args.exc_value, args.exc_traceback,
            )
            _prev_thread(args)

        threading.excepthook = _thook
