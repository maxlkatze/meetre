"""Live captions while recording.

A worker thread tails the recorder's WAVs (:class:`meetre.recorder.WavTail`),
keeps the not-yet-committed audio in memory and re-transcribes that window
every couple of seconds. Text near the end of the window is *tentative* (it
may change as more audio arrives); once the window grows long enough, every
segment but the last is *committed* and the window start moves past it, so
each pass only decodes the last ~10–25 s.

Each committed segment is labelled "me" or "remote" by comparing the energy of
the mic and system-audio stems over its span (the system stem only ever holds
the other participants; the mic may carry their echo, hence the bias).

This is a preview only — the saved transcript still comes from the full
post-recording pass with diarization.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

SR = 16_000
STEP = 1.5             # seconds between passes (after the previous one finishes)
MIN_WINDOW = 1.0       # don't decode less than this
COMMIT_AFTER = 10.0    # once the window is this long, commit all but the last segment
MAX_WINDOW = 25.0      # hard cap: commit everything
SILENCE_RMS = 0.004    # window quieter than this is skipped, not decoded
PROMPT_CHARS = 200     # committed-text tail used as Whisper's prompt


def _rms(a) -> float:
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


class LiveTranscriber:
    """``on_update(lines, tentative, status)`` is called from the worker thread
    with ``lines`` = [(speaker | None, text), …] (committed, consecutive
    same-speaker segments merged), ``tentative`` = the in-flux tail text and
    ``status`` = a short state label ("listening", "loading model", …)."""

    def __init__(self, paths: dict, model: str, language: Optional[str],
                 on_update: Callable, labels: dict):
        from .recorder import WavTail

        self._tails = {role: WavTail(p) for role, p in paths.items()}
        self._bufs = {role: np.zeros(0, dtype="float32") for role in paths}
        self._model, self._language = model, language
        self._on_update = on_update
        self._labels = labels  # {"mic": "Me", "system": "Speaker"}
        self._lines: List[list] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="meetre-live")
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop; doesn't wait for an in-flight pass."""
        self._stop.set()

    # -- worker ---------------------------------------------------------------

    def _emit(self, tentative: str = "", status: str = "listening") -> None:
        try:
            self._on_update([tuple(l) for l in self._lines], tentative, status)
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        try:
            self._prepare()
        except Exception as e:  # noqa: BLE001
            self._emit(status=f"unavailable: {e}")
            return
        self._emit()
        while not self._stop.wait(STEP):
            try:
                self._pass()
            except Exception:  # noqa: BLE001 — a bad pass must not end captions
                import sys

                from . import crashlog

                crashlog._write_exception("Live transcription pass failed", *sys.exc_info())

    def _prepare(self) -> None:
        from . import downloads, transcriber

        repo = transcriber.mlx_repo(self._model)
        if not downloads.is_cached(repo):
            downloads.ensure_model(
                repo, lambda frac, *_: self._emit(status=f"downloading model {int(frac * 100)}%"))
        self._emit(status="loading model")
        # Warm-up decode so the first real pass isn't paying the model load.
        transcriber.transcribe_live(np.zeros(SR, dtype="float32"), self._model, self._language)

    def _pull(self) -> None:
        for role, tail in self._tails.items():
            new = tail.read()
            if len(new):
                self._bufs[role] = np.concatenate([self._bufs[role], new])

    def _window(self) -> np.ndarray:
        n = max(len(b) for b in self._bufs.values())
        mix = np.zeros(n, dtype="float32")
        for b in self._bufs.values():
            mix[: len(b)] += b
        return mix

    def _advance(self, seconds: float) -> None:
        """Drop ``seconds`` of committed audio from the front of every stem."""
        k = int(seconds * SR)
        for role in self._bufs:
            self._bufs[role] = self._bufs[role][k:]

    def _speaker(self, start: float, end: float) -> Optional[str]:
        if len(self._bufs) < 2:
            return None
        a, b = int(start * SR), max(int(end * SR), int(start * SR) + 1)
        mic = _rms(self._bufs.get("mic", np.zeros(0))[a:b])
        sys_ = _rms(self._bufs.get("system", np.zeros(0))[a:b])
        return self._labels["system"] if sys_ > 0.3 * mic else self._labels["mic"]

    def _commit(self, segs) -> None:
        for s in segs:
            who = self._speaker(s.start, s.end)
            if self._lines and self._lines[-1][0] == who:
                self._lines[-1][1] += " " + s.text
            else:
                self._lines.append([who, s.text])

    def _prompt(self) -> Optional[str]:
        text = " ".join(t for _, t in self._lines[-4:])[-PROMPT_CHARS:]
        return text or None

    def _pass(self) -> None:
        from . import transcriber

        self._pull()
        window = self._window()
        dur = len(window) / SR
        if dur < MIN_WINDOW:
            return
        if _rms(window[-int(MIN_WINDOW * SR):]) < SILENCE_RMS and _rms(window) < SILENCE_RMS:
            # Nothing but silence pending: drop it (keep a short lead-in).
            if dur > 3.0:
                self._advance(dur - 1.0)
            self._emit()
            return

        t0 = time.monotonic()
        segs = transcriber.transcribe_live(window, self._model, self._language, self._prompt())
        if self._stop.is_set():
            return
        if not segs:
            if dur > MAX_WINDOW:
                self._advance(dur - 1.0)
            self._emit()
            return

        # Commit settled text: everything but the last segment once the window
        # is long, everything once the speaker has paused or the cap is hit.
        paused = dur - segs[-1].end > 1.5
        if dur >= MAX_WINDOW or (paused and dur >= 4.0):
            self._commit(segs)
            self._advance(min(segs[-1].end, dur))
            segs = []
        elif dur >= COMMIT_AFTER and len(segs) >= 2:
            self._commit(segs[:-1])
            self._advance(segs[-1].start)
            segs = segs[-1:]
        self._emit(" ".join(s.text for s in segs))
        # Slow machine / big model: back off so passes don't queue up.
        self._stop.wait(max(0.0, (time.monotonic() - t0) - STEP))
