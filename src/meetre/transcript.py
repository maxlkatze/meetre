"""Render segments into a readable Markdown transcript file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from .transcriber import Segment


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_transcript(
    segments: List[Segment],
    out_dir: Path,
    *,
    title: str,
    started_at: datetime,
    duration: float,
    model: str,
    backend: str,
    person_detection: bool,
    summary: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"{slug}_{_slugify(title)}.md"

    speakers = sorted({s.speaker for s in segments if s.speaker})

    lines = [
        f"# {title}",
        "",
        f"- **Date:** {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Duration:** {_ts(duration)}",
        f"- **Model:** {model} ({backend})",
        f"- **Person detection:** {'on' if person_detection else 'off'}",
    ]
    if speakers:
        lines.append(f"- **Speakers:** {', '.join(speakers)}")
    lines += ["", "---", ""]

    if summary.strip():
        lines += [summary.strip(), "", "---", "", "## Transcript", ""]

    last_speaker = object()  # sentinel so the first line always prints a header
    for seg in segments:
        if person_detection and seg.speaker != last_speaker:
            lines.append("")
            lines.append(f"**{seg.speaker or 'Unknown'}**")
            last_speaker = seg.speaker
        prefix = f"`[{_ts(seg.start)}]` "
        lines.append(f"{prefix}{seg.text}")

    lines.append("")
    path.write_text("\n".join(lines))
    write_words(path, segments)
    return path


def words_path(md: Path) -> Path:
    """Hidden sidecar with each transcript line's timing and word timings,
    for playback with the current word highlighted."""
    return md.with_name(f".{md.stem}.words.json")


def write_words(md: Path, segments: List[Segment]) -> None:
    """One entry per transcript line, in order: ``{"s", "e", "w": [[word,
    start, end], …]}`` (``w`` only when the backend gave word timings)."""
    import json

    out = []
    for seg in segments:
        item = {"s": round(seg.start, 2), "e": round(seg.end, 2)}
        if seg.words:
            item["w"] = [[w["word"].strip(), round(float(w["start"]), 2),
                          round(float(w["end"]), 2)] for w in seg.words if w["word"].strip()]
        out.append(item)
    try:
        words_path(md).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    except OSError:
        pass  # playback falls back to estimated word times


def move_words(old: Path, new: Path) -> None:
    """Keep the sidecar with its transcript (``new=None`` deletes it)."""
    src = words_path(old)
    if not src.exists():
        return
    if new is None:
        src.unlink(missing_ok=True)
    else:
        src.replace(words_path(new))


def _slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "meeting"
