"""Live captions while recording.

A worker thread tails the recorder's WAVs (:class:`meetre.recorder.WavTail`)
and transcribes each source as its OWN stream — the system audio (the other
participants) and the microphone (you) — then merges the results by time into
one caption feed. Decoding the sources separately means people talking over
each other never garble one mixed signal, and every line's speaker is simply
its source.

Per stream, the not-yet-committed audio is kept in memory and re-decoded as
often as the model allows. Text near the end of the window is *tentative*
(it may still change); once the window is long enough (or the speaker
pauses) the settled segments are *committed* and the window moves on, so each
pass only decodes the last ~10–25 s.

The mic is normally echo-free (macOS Voice Processing, see
:class:`meetre.recorder._VoiceMicSource`). With the plain fallback mic and
speakers on, the call leaks into the mic; mic segments that repeat what the
system stream said at about the same time are dropped as echo.

This is a preview only — the saved transcript still comes from the full
post-recording pass with diarization.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, List, Optional

import numpy as np

SR = 16_000
STEP = 1.5             # max seconds between passes (after the previous one finishes)
MIN_STEP = 0.5         # fast models (Parakeet) re-decode this often
MIN_WINDOW = 1.0       # don't decode less than this
COMMIT_AFTER = 10.0    # once the window is this long, commit all but the last segment
MAX_WINDOW = 25.0      # hard cap: commit everything
SILENCE_RMS = 0.004    # window quieter than this is skipped, not decoded
PROMPT_CHARS = 200     # committed-text tail used as Whisper's prompt
ECHO_WINDOW = 12.0     # seconds: mic text repeating system text this close is echo
ECHO_OVERLAP = 0.6     # share of a mic segment's words found in the system text


def _rms(a) -> float:
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class _Stream:
    """One source (system audio or mic): its tail, pending audio and output."""

    def __init__(self, role: str, label: Optional[str], path):
        from .recorder import WavTail

        self.role, self.label = role, label
        self.tail = WavTail(path)
        self.buf = np.zeros(0, dtype="float32")
        self.base = 0.0            # stream time (s) of buf[0]
        self.tentative: list = []  # [(abs_start, text)] still in flux

    def pull(self) -> None:
        new = self.tail.read()
        if len(new):
            self.buf = np.concatenate([self.buf, new])

    def advance(self, seconds: float) -> None:
        k = int(seconds * SR)
        self.buf = self.buf[k:]
        self.base += k / SR


class LiveTranscriber:
    """``on_update(lines, tentative, status)`` is called from the worker thread
    with ``lines`` = [(speaker | None, text), …] (committed, time-ordered,
    consecutive same-speaker segments merged), ``tentative`` = the in-flux
    tail text and ``status`` = a short state label ("listening", …)."""

    def __init__(self, paths: dict, model: str, language: Optional[str],
                 on_update: Callable, labels: dict):
        multi = len(paths) > 1
        # System first: when both sides speak in the same pass, the call's
        # words are known before the mic's echo check runs.
        order = sorted(paths, key=lambda r: r != "system")
        self._streams = [_Stream(r, labels.get(r) if multi else None, paths[r]) for r in order]
        self._model, self._language = model, language
        self._on_update = on_update
        self._committed: List[tuple] = []   # (abs_start, speaker, text)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step = STEP

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="meetre-live")
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop; doesn't wait for an in-flight pass."""
        self._stop.set()

    # -- output -----------------------------------------------------------------

    def _lines(self) -> List[tuple]:
        lines: List[list] = []
        for _, who, text in sorted(self._committed, key=lambda c: c[0]):
            if lines and lines[-1][0] == who:
                lines[-1][1] += " " + text
            else:
                lines.append([who, text])
        return [tuple(l) for l in lines]

    def _tentative(self, lines) -> str:
        pending = sorted(((t, s.label, text) for s in self._streams for t, text in s.tentative),
                         key=lambda p: p[0])
        last = lines[-1][0] if lines else None
        parts = []
        for _, who, text in pending:
            # Name the speaker when the in-flux text isn't theirs to continue.
            parts.append(f"{who}: {text}" if who and who != last else text)
            last = who
        return " ".join(parts)

    def _emit(self, status: str = "listening") -> None:
        try:
            lines = self._lines()
            self._on_update(lines, self._tentative(lines), status)
        except Exception:  # noqa: BLE001
            pass

    # -- worker ---------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._prepare()
        except Exception as e:  # noqa: BLE001
            self._emit(status=f"unavailable: {e}")
            return
        self._emit()
        while not self._stop.wait(self._step):
            t0 = time.monotonic()
            try:
                for s in self._streams:
                    if self._stop.is_set():
                        return
                    self._pass(s)
                self._emit()
            except Exception:  # noqa: BLE001 — a bad pass must not end captions
                import sys

                from . import crashlog

                crashlog._write_exception("Live transcription pass failed", *sys.exc_info())
            # Re-decode as often as the model allows: about twice the time a
            # pass takes (fast models update near-instantly), never more than STEP.
            took = time.monotonic() - t0
            self._step = min(STEP, max(MIN_STEP, 2 * took))

    def _prepare(self) -> None:
        from . import downloads, transcriber

        repo = transcriber.mlx_repo(self._model)
        if not downloads.is_cached(repo):
            downloads.ensure_model(
                repo, lambda frac, *_: self._emit(status=f"downloading model {int(frac * 100)}%"))
        self._emit(status="loading model")
        # Warm-up decode so the first real pass isn't paying the model load.
        transcriber.transcribe_live(np.zeros(SR, dtype="float32"), self._model, self._language)

    def _prompt(self) -> Optional[str]:
        text = " ".join(t for _, _, t in sorted(self._committed)[-6:])[-PROMPT_CHARS:]
        return text or None

    def _is_echo(self, s: _Stream, start: float, text: str) -> bool:
        """A mic segment that repeats the call (speakers → mic) — drop it."""
        if s.role != "mic":
            return False
        words = _words(text)
        if len(words) < 2:
            return False
        nearby = [t for st in self._streams if st.role == "system"
                  for t0, t in [(c[0], c[2]) for c in self._committed] + st.tentative
                  if abs(t0 - start) < ECHO_WINDOW]
        if not nearby:
            return False
        heard = set(_words(" ".join(nearby)))
        return sum(w in heard for w in words) / len(words) >= ECHO_OVERLAP

    def _commit(self, s: _Stream, segs) -> None:
        for seg in segs:
            start = s.base + seg.start
            if not self._is_echo(s, start, seg.text):
                self._committed.append((start, s.label, seg.text))

    def _pass(self, s: _Stream) -> None:
        from . import transcriber

        s.pull()
        dur = len(s.buf) / SR
        if dur < MIN_WINDOW:
            return
        if _rms(s.buf[-int(MIN_WINDOW * SR):]) < SILENCE_RMS and _rms(s.buf) < SILENCE_RMS:
            # Nothing but silence pending: drop it (keep a short lead-in).
            if dur > 3.0:
                s.advance(dur - 1.0)
            s.tentative = []
            return

        regions = None
        if s.role == "mic":
            # The (gain-controlled) mic has faint residual sound that ASR
            # happily "hears" words in. Only decode where VAD finds speech.
            from .align import speech_regions_array

            regions = speech_regions_array(s.buf)
            if regions == []:
                if dur > 3.0:
                    s.advance(dur - 1.0)
                s.tentative = []
                return

        segs = transcriber.transcribe_live(s.buf, self._model, self._language, self._prompt())
        if self._stop.is_set():
            return
        if regions:
            segs = [g for g in segs
                    if any(min(g.end, e) - max(g.start, b) > 0.1 for b, e in regions)]
        if not segs:
            if dur > MAX_WINDOW:
                s.advance(dur - 1.0)
            s.tentative = []
            return

        # Commit settled text: everything once the speaker paused or the cap
        # is hit, else all but the last segment once the window is long.
        paused = dur - segs[-1].end > 1.5
        if dur >= MAX_WINDOW or (paused and dur >= 4.0):
            self._commit(s, segs)
            s.advance(min(segs[-1].end, dur))
            segs = []
        elif dur >= COMMIT_AFTER and len(segs) >= 2:
            self._commit(s, segs[:-1])
            s.advance(segs[-1].start)
            segs = segs[-1:]
        s.tentative = [(s.base + g.start, g.text) for g in segs
                       if not self._is_echo(s, s.base + g.start, g.text)]
