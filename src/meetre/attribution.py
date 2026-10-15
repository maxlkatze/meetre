"""Who said it — you (microphone) or the call (system audio)? Word by word.

The old approach compared raw RMS of the two stems over a whole segment, which
went wrong in three common cases:

* a segment (Parakeet "sentence") spanning both people got ONE label;
* the stems have very different gains (a quiet mic vs. loud call audio), so
  "louder" is meaningless without normalisation;
* without headphones the mic also hears the call through the speakers (echo),
  so the remote person often "won" on the mic stem — or swamped your words.

Here each stem is measured in short frames, relative to its own noise floor;
the speaker echo the mic picks up is estimated from frames where only the
call is talking and subtracted; then every WORD is assigned by voting over its
frames, smoothed, and segments are split wherever the side changes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

SR = 16_000
FRAME = 0.02            # seconds per analysis frame
MARGIN_DB = 3.0         # how much more active one side must be to win a frame
ACTIVE_DB = 6.0         # level above the noise floor that counts as speech
MIN_RUN = 0.35          # seconds: shorter side switches are smoothed away


def _frame_power(x: Optional[np.ndarray], n: int) -> np.ndarray:
    hop = int(SR * FRAME)
    if x is None:
        return np.zeros(n, dtype="float64")
    x = x[: n * hop]
    if len(x) < n * hop:
        x = np.pad(x, (0, n * hop - len(x)))
    return (x.reshape(n, hop).astype("float64") ** 2).mean(axis=1) + 1e-12


def _floor(p: np.ndarray) -> float:
    return float(np.percentile(p, 10)) + 1e-12


def side_activity(mic: Optional[np.ndarray], system: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame activity in dB above each stem's own noise floor
    (mic with the estimated speaker echo removed)."""
    length = max(len(mic) if mic is not None else 0, len(system) if system is not None else 0)
    n = max(1, length // int(SR * FRAME))
    m = _frame_power(mic, n)
    s = _frame_power(system, n)
    fm, fs = _floor(m), _floor(s)

    # Echo: on frames where the call is clearly talking, the mic level that
    # tracks it is speaker bleed. A low percentile of mic/system there is the
    # bleed gain (frames where you talk over them sit higher and are ignored).
    s_active = 10 * np.log10(s / fs) > ACTIVE_DB + 6
    if system is not None and mic is not None and s_active.sum() > 25:
        gain = float(np.percentile(m[s_active] / s[s_active], 30))
        m = np.maximum(m - 1.5 * gain * s, fm)  # 1.5×: margin for echo jitter

    a_mic = 10 * np.log10(m / fm)
    a_sys = 10 * np.log10(s / fs)
    return a_mic, a_sys


def _word_side(a_mic, a_sys, start: float, end: float) -> Optional[str]:
    i = max(0, int(start / FRAME))
    j = max(i + 1, int(np.ceil(end / FRAME)))
    am, as_ = a_mic[i:j], a_sys[i:j]
    if not len(am):
        return None
    local = ((am > as_ + MARGIN_DB) & (am > ACTIVE_DB)).sum()
    remote = ((as_ > am + MARGIN_DB) & (as_ > ACTIVE_DB)).sum()
    if local == remote:
        # No clear frame majority: fall back to the louder side overall.
        dm, ds = float(am.mean()), float(as_.mean())
        if abs(dm - ds) < 1.0:
            return None
        return "local" if dm > ds else "remote"
    return "local" if local > remote else "remote"


def _smooth(words: List[dict]) -> None:
    """Fill undecided words and absorb very short switches into their context."""
    last = None
    for w in words:
        if w["side"] is None:
            w["side"] = last
        last = w["side"] or last
    nxt = None
    for w in reversed(words):  # leading undecided words take the next side
        if w["side"] is None:
            w["side"] = nxt
        nxt = w["side"] or nxt
    # Runs shorter than MIN_RUN between two runs of the same other side flip.
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and words[j + 1]["side"] == words[i]["side"]:
            j += 1
        dur = words[j]["end"] - words[i]["start"]
        before = words[i - 1]["side"] if i > 0 else None
        after = words[j + 1]["side"] if j + 1 < len(words) else None
        if dur < MIN_RUN and before is not None and before == after:
            for k in range(i, j + 1):
                words[k]["side"] = before
        i = j + 1


def _words_of(seg) -> List[dict]:
    """Word timings for a segment; evenly spread over its span if none known."""
    if seg.words:
        return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                for w in seg.words]
    toks = seg.text.split()
    if not toks:
        return []
    span = max(seg.end - seg.start, 0.01)
    step = span / len(toks)
    return [{"word": t, "start": seg.start + i * step, "end": seg.start + (i + 1) * step}
            for i, t in enumerate(toks)]


def split_by_side(segments, mic, system, local_label: str, remote_label: str):
    """Return new segments, each spoken by ONE side, labelled local/remote.

    A segment whose words change side is split at the change.
    """
    from .transcriber import Segment

    if mic is None or system is None:
        label = remote_label if mic is None else local_label
        for s in segments:
            s.speaker = label
        return list(segments)

    a_mic, a_sys = side_activity(mic, system)
    out = []
    for seg in segments:
        words = _words_of(seg)
        if not words:
            continue
        for w in words:
            w["side"] = _word_side(a_mic, a_sys, w["start"], w["end"])
        _smooth(words)
        if all(w["side"] is None for w in words):
            for w in words:  # nothing measurable: whole segment, louder side
                w["side"] = _word_side(a_mic, a_sys, seg.start, seg.end) or "remote"
        run: List[dict] = []
        for w in words + [None]:
            if run and (w is None or w["side"] != run[0]["side"]):
                text = _join([r["word"] for r in run]) if seg.words else " ".join(r["word"] for r in run)
                if text.strip():
                    out.append(Segment(
                        start=run[0]["start"], end=run[-1]["end"], text=text.strip(),
                        speaker=local_label if run[0]["side"] == "local" else remote_label,
                        words=[{k: r[k] for k in ("word", "start", "end")} for r in run]))
                run = []
            if w is not None:
                run.append(w)
    return out


def _join(words: List[str]) -> str:
    """Join word strings; Parakeet words keep their own leading spaces."""
    if any(w.startswith(" ") for w in words[1:]):
        return "".join(words)
    return " ".join(w.strip() for w in words)


def split_by_turns(segments, turns):
    """Split segments where diarization turns change speaker, word by word.

    ``turns`` are pyannote (start, end, speaker). Words take the speaker of
    the turn they overlap most; short switches are smoothed away (MIN_RUN).
    Segments without word timings are left whole.
    """
    from .transcriber import Segment, _best_turn

    if not turns:
        return list(segments)
    out = []
    for seg in segments:
        if not seg.words or len(seg.words) < 2:
            out.append(seg)
            continue
        words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                  "side": _best_turn(turns, float(w["start"]), float(w["end"]))}
                 for w in seg.words]
        _smooth(words)
        if len({w["side"] for w in words}) <= 1:
            out.append(seg)
            continue
        run = []
        for w in words + [None]:
            if run and (w is None or w["side"] != run[0]["side"]):
                text = _join([r["word"] for r in run]).strip()
                if text:
                    out.append(Segment(start=run[0]["start"], end=run[-1]["end"], text=text,
                                       words=[{k: r[k] for k in ("word", "start", "end")}
                                              for r in run]))
                run = []
            if w is not None:
                run.append(w)
    return out
