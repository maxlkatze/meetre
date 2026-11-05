"""Known voices — recognise people across meetings by their voiceprint.

Every meeting keeps each speaker's averaged TitaNet print in a hidden sidecar
next to the transcript (``.{stem}.voices.json``). Renaming a speaker
(``rename_speaker``: "Vor Ort 1" → "Anna") rewrites the transcript and saves
that print under the name in the voice book (``~/.config/meetre/voices.json``).
Later meetings match each speaker against the book (``recognize``) before the
LLM guesses names from the conversation, and a confident match refines the
saved print, so a voice gets better known with every meeting.

Prints never leave the machine; ``forget`` removes a person.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CONFIG_DIR

BOOK_PATH = CONFIG_DIR / "voices.json"
# Same person across calls/mics scores 0.4-0.7 between segments, different
# people 0.0-0.35 (voices.SAME_VOICE); averaged prints are steadier than
# single segments, so 0.5 names a voice and only ≥ 0.6 also refines the saved
# print (a wrong match must not drift someone's voice towards another's).
MATCH = 0.5
REFINE = 0.6
# The saved print is a speech-time weighted mean; capping the old weight keeps
# it adapting to new mics/rooms instead of freezing after a few meetings.
MAX_WEIGHT = 1800.0  # seconds


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype="float64")
    return v / (np.linalg.norm(v) + 1e-9)


# -- the book -------------------------------------------------------------------------

def load() -> Dict[str, dict]:
    """``{name: {"print": [...], "seconds": float, "meetings": int, "updated": iso}}``."""
    try:
        data = json.loads(BOOK_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(book: Dict[str, dict]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BOOK_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(book, ensure_ascii=False))
    tmp.replace(BOOK_PATH)


def _merge(entry: Optional[dict], vec, seconds: float) -> dict:
    old_w = min(float(entry.get("seconds", 0.0)), MAX_WEIGHT) if entry else 0.0
    new = _unit(vec) * max(seconds, 1.0)
    if entry:
        new = new + _unit(entry["print"]) * old_w
    return {"print": [round(float(x), 5) for x in _unit(new)],
            "seconds": round((entry or {}).get("seconds", 0.0) + seconds, 1),
            "meetings": (entry or {}).get("meetings", 0) + 1,
            "updated": datetime.now().isoformat(timespec="seconds")}


def enroll(name: str, vec, seconds: float) -> None:
    """Add (or blend into) ``name``'s saved voice."""
    book = load()
    book[name] = _merge(book.get(name), vec, seconds)
    _save(book)


def forget(name: str) -> bool:
    book = load()
    if name not in book:
        return False
    del book[name]
    _save(book)
    return True


def people() -> List[dict]:
    """The saved voices, most speech first (for the UI/CLI)."""
    return sorted(({"name": k, "seconds": v.get("seconds", 0.0),
                    "meetings": v.get("meetings", 0), "updated": v.get("updated")}
                   for k, v in load().items()), key=lambda p: -p["seconds"])


# -- one meeting ----------------------------------------------------------------------

def centroids(segments) -> Dict[str, Tuple[np.ndarray, float]]:
    """``{speaker label: (unit print, seconds of speech)}`` from the prints
    voices.py attached to the segments."""
    acc: Dict[str, list] = {}
    for s in segments:
        v = getattr(s, "voice", None)
        if v is None or not s.speaker:
            continue
        d = max(0.0, s.end - s.start)
        a = acc.setdefault(s.speaker, [np.zeros(len(v)), 0.0])
        a[0] = a[0] + _unit(v) * max(d, 0.1)
        a[1] += d
    return {k: (_unit(v), t) for k, (v, t) in acc.items()}


def recognize(segments, *, refine: bool = True) -> Dict[str, str]:
    """``{label: name}`` for speakers whose voice is in the book (≥ MATCH;
    each name and each label at most once, best pairs first). Confident
    matches (≥ REFINE) blend this meeting's print into the saved one."""
    book = load()
    cents = centroids(segments)
    if not book or not cents:
        return {}
    pairs = []
    for label, (v, _t) in cents.items():
        for name, e in book.items():
            try:
                pairs.append((float(v @ _unit(e["print"])), label, name))
            except (KeyError, ValueError):
                continue
    found: Dict[str, str] = {}
    used = set()
    for sim, label, name in sorted(pairs, reverse=True):
        if sim < MATCH:
            break
        if label in found or name in used:
            continue
        found[label] = name
        used.add(name)
        if refine and sim >= REFINE:
            book[name] = _merge(book[name], cents[label][0], cents[label][1])
    if refine and any(sim >= REFINE for sim, label, name in pairs
                      if found.get(label) == name):
        _save(book)
    return found


def sidecar_path(md: Path) -> Path:
    return md.with_name(f".{md.stem}.voices.json")


def write_sidecar(md: Path, segments) -> None:
    """Each speaker's print for this transcript (removed when there is none,
    so a re-transcription never keeps stale prints under new labels)."""
    cents = centroids(segments)
    path = sidecar_path(md)
    try:
        if not cents:
            path.unlink(missing_ok=True)
            return
        path.write_text(json.dumps(
            {k: {"print": [round(float(x), 5) for x in v], "seconds": round(t, 1)}
             for k, (v, t) in cents.items()}, ensure_ascii=False))
    except OSError:
        pass


def _read_sidecar(md: Path) -> Dict[str, dict]:
    try:
        return json.loads(sidecar_path(md).read_text())
    except (OSError, ValueError):
        return {}


def rename_speaker(md: Path, old: str, new: str, *, remember: bool = True) -> bool:
    """Rename speaker ``old`` → ``new`` in the transcript (speaker headers, the
    Speakers line and bold mentions in the summary) and, with ``remember``,
    save the voice under ``new``. Renaming to a name already in the meeting
    merges the two. Returns whether a voice was saved."""
    old, new = old.strip(), new.strip()
    if not old or not new or "*" in new or "\n" in new:
        raise ValueError("Invalid speaker name.")
    text = md.read_text()
    if f"**{old}**" not in text:
        raise KeyError(old)
    if old == new:
        lines = text.split("\n")
    else:
        lines = text.split("\n")
        word = re.compile(rf"(?<![\w]){re.escape(old)}(?![\w])")
        in_transcript = False
        for i, line in enumerate(lines):
            if line.startswith("- **Speakers:**"):
                names = [s.strip() for s in line.split(":**", 1)[1].split(",") if s.strip()]
                names = sorted(dict.fromkeys(new if n == old else n for n in names))
                lines[i] = f"- **Speakers:** {', '.join(names)}"
            elif line == f"**{old}**":
                lines[i] = f"**{new}**"
            elif line.strip() in ("## Transcript", "## Transkript"):
                in_transcript = True
            elif not in_transcript and not line.startswith("`["):
                # Summary: "**Vor Ort 1:** …", "**Vor Ort 1**" — and numbered
                # labels anywhere (never prose like "Ich").
                lines[i] = re.sub(rf"\*\*{re.escape(old)}(:?)\*\*", rf"**{new}\1**", line)
                if any(c.isdigit() for c in old):
                    lines[i] = word.sub(new, lines[i])
        lines = _merge_turns(lines)
        tmp = md.with_suffix(".md.tmp")
        tmp.write_text("\n".join(lines))
        tmp.replace(md)

    side = _read_sidecar(md)
    entry = side.pop(old, None)
    if entry and old != new:
        if new in side:  # two labels were one person: blend
            a, b = side[new], entry
            v = _unit(np.asarray(a["print"]) * max(a["seconds"], 1.0)
                      + np.asarray(b["print"]) * max(b["seconds"], 1.0))
            side[new] = {"print": [round(float(x), 5) for x in v],
                         "seconds": round(a["seconds"] + b["seconds"], 1)}
        else:
            side[new] = entry
        try:
            sidecar_path(md).write_text(json.dumps(side, ensure_ascii=False))
        except OSError:
            pass
    saved = side.get(new) if old != new else entry
    if remember and saved:
        enroll(new, saved["print"], float(saved.get("seconds", 0.0)))
        return True
    return False


def _merge_turns(lines: List[str]) -> List[str]:
    """Drop a speaker header that repeats the previous turn's speaker (after a
    rename two adjacent turns can be the same person). Speaker headers are the
    bold-only lines followed by a timestamped line."""
    out: List[str] = []
    last = None
    for i, line in enumerate(lines):
        head = re.fullmatch(r"\*\*(.+)\*\*", line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if head and nxt.startswith("`["):
            if head.group(1) == last:
                while out and out[-1] == "":
                    out.pop()
                continue
            last = head.group(1)
        out.append(line)
    return out
