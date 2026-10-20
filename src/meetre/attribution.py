"""Who said it — you (microphone) or the call (system audio)?

Each source is transcribed as its own stream (live captions and the final
pass alike), so a segment's side is simply its source. Comparing stem energies
on one transcription of the mix went wrong: the call stem is digital silence
between words, so measured against its own ~0 noise floor it out-shouts the
mic on nearly every frame and almost everything became "remote".

What's left is echo: without headphones (or macOS Voice Processing) the mic
also hears the call through the speakers, so the mic stream transcribes the
remote side a second time. Mic segments whose words mostly repeat what the
call said at about the same time are dropped (:func:`is_echo`).

Diarization turns are mapped onto words by :func:`split_by_turns`.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

MIN_RUN = 0.35          # seconds: shorter speaker switches are smoothed away
ECHO_WINDOW = 12.0      # seconds: mic text repeating system text this close is echo
ECHO_OVERLAP = 0.6      # share of a mic segment's words found in the system text


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def is_echo(text: str, start: float, system: Iterable[Tuple[float, str]]) -> bool:
    """True if mic ``text`` (starting at ``start``) repeats the call: most of
    its words occur in ``system`` items ``(start, text)`` within ECHO_WINDOW."""
    words = _words(text)
    if len(words) < 2:
        return False
    nearby = [t for t0, t in system if abs(t0 - start) < ECHO_WINDOW]
    if not nearby:
        return False
    heard = set(_words(" ".join(nearby)))
    return sum(w in heard for w in words) / len(words) >= ECHO_OVERLAP


def drop_echo(mic_segs, sys_segs) -> list:
    """Mic segments minus those that are the call heard through the speakers."""
    system = [(g.start, g.text) for g in sys_segs]
    return [g for g in mic_segs if not is_echo(g.text, g.start, system)]


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
