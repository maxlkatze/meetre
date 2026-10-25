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


def _pinned_revision(repo: str) -> Optional[str]:
    """Commit an HF-cache model is pinned to (none now: the summary models
    are GGUF files, pinned in gguf.CATALOG)."""
    return None


def is_cached(repo: str) -> bool:
    if repo.startswith("gguf:"):  # a local GGUF file: nothing to download
        return Path(repo[5:]).is_file()
    snaps = _model_dir(repo) / "snapshots"
    rev = _pinned_revision(repo)
    if rev:
        return (snaps / rev).is_dir()
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
    if repo.startswith("gguf:"):
        _ensure_gguf(repo[5:], on_progress)
        return
    if is_cached(repo):
        return
    from huggingface_hub import snapshot_download

    total = remote_size(repo)
    blobs = _model_dir(repo) / "blobs"
    state = {"done": False, "err": None}

    def _dl():
        try:
            snapshot_download(repo, revision=_pinned_revision(repo))
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


def local_first(load):
    """Run a model load against the files on disk only (Hugging Face offline
    mode); only if something is missing go online. Each load otherwise asks
    the Hub first — slow on a bad network, and pointless once downloaded.
    (From LocalCompanion: Parakeet load 0.36 s → 0.04 s.)"""
    from huggingface_hub import constants

    was = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        return load()
    except Exception:  # noqa: BLE001 — not (fully) on disk: go online
        constants.HF_HUB_OFFLINE = was
        return load()
    finally:
        constants.HF_HUB_OFFLINE = was


def _ensure_gguf(path: str, on_progress: Optional[ProgressCB]) -> None:
    """A GGUF summary model + llama-server, downloading what's missing (a
    catalog file by name; any other file must already be there)."""
    from . import gguf

    gguf.ensure_server(on_progress)
    if Path(path).is_file():
        return
    alias = next((a for a, spec in gguf.CATALOG.items() if spec.file == Path(path).name), None)
    if alias is None:
        raise RuntimeError(f"GGUF model not found: {path}")
    gguf.ensure(alias, on_progress)
