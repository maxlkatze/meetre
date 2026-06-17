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
    return path


def _slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "meeting"
