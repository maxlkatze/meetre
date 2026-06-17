"""Model download helpers with progress reporting.

HuggingFace progress bars are disabled globally (see ``__init__``) so the CLI
and menu-bar UIs can render their own. ``ensure_model`` downloads a repo if it
is missing and reports fractional progress by polling the cache while
``snapshot_download`` runs on a worker thread.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

ProgressCB = Callable[[float, int, int], None]  # (fraction, bytes_done, bytes_total)


def _model_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def is_cached(repo: str) -> bool:
    snaps = _model_dir(repo) / "snapshots"
    return snaps.exists() and any(snaps.iterdir())


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def remote_size(repo: str) -> int:
    """Total download size in bytes (0 if it can't be determined)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        return sum((s.size or 0) for s in (info.siblings or []))
    except Exception:  # noqa: BLE001
        return 0


def ensure_model(repo: str, on_progress: Optional[ProgressCB] = None) -> None:
    """Download ``repo`` if not already cached, reporting progress.

    ``on_progress(fraction, bytes_done, bytes_total)`` is called periodically
    while downloading. Raises if the download fails.
    """
    if is_cached(repo):
        return
    from huggingface_hub import snapshot_download

    total = remote_size(repo)
    blobs = _model_dir(repo) / "blobs"
    state = {"done": False, "err": None}

    def _dl():
        try:
            snapshot_download(repo)
        except Exception as e:  # noqa: BLE001
            state["err"] = e
        finally:
            state["done"] = True

    worker = threading.Thread(target=_dl, daemon=True)
    worker.start()

    while not state["done"]:
        if on_progress:
            cur = _dir_size(blobs)
            frac = (cur / total) if total else 0.0
            on_progress(min(frac, 0.999), cur, total)
        time.sleep(0.4)
    worker.join()
    if state["err"]:
        raise state["err"]
    if on_progress:
        on_progress(1.0, total, total)


def bar(fraction: float, width: int = 10) -> str:
    """A unicode block progress bar, e.g. ``████████░░``."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1_048_576:.0f} MB"
