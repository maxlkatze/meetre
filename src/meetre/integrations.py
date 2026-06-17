"""Hand-offs to other macOS apps: Claude Desktop and Apple Notes.

* **Claude Desktop** has no public API or URL scheme to inject a prompt and
  read back a reply, so we do the reliable thing: put a ready-to-send
  summarisation prompt (plus the transcript) on the clipboard and open the
  app. You paste (⌘V) and press Enter to get the summary.

* **Apple Notes** *is* scriptable, so the full transcript is written there
  automatically via AppleScript.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

CLAUDE_APP = "/Applications/Claude.app"


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def to_clipboard(text: str) -> None:
    """Copy ``text`` to the macOS clipboard via pbcopy."""
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


# ---------------------------------------------------------------------------
# Claude Desktop
# ---------------------------------------------------------------------------

def claude_desktop_installed() -> bool:
    return Path(CLAUDE_APP).exists()


def summary_prompt(title: str, transcript_md: str, language: Optional[str] = "de") -> str:
    """Build the summarisation prompt that gets handed to Claude Desktop."""
    lang_line = {
        "de": "Antworte auf Deutsch.",
        "en": "Respond in English.",
    }.get(language or "", "Respond in the meeting's language.")
    return (
        f"Hier ist das Transkript eines Meetings („{title}“). "
        "Bitte erstelle eine strukturierte Zusammenfassung mit: \n"
        "1. TL;DR (2–3 Sätze)\n"
        "2. Wichtigste Themen / Entscheidungen\n"
        "3. Action Items (mit Verantwortlichen, falls erkennbar)\n"
        "4. Offene Fragen\n\n"
        f"{lang_line}\n\n"
        "---- TRANSKRIPT ----\n\n"
        f"{transcript_md}"
    )


def forward_to_claude(title: str, transcript_md: str, language: Optional[str] = "de") -> str:
    """Copy the summarisation prompt to the clipboard and open Claude Desktop.

    Returns the prompt text that was placed on the clipboard. Raises
    ``RuntimeError`` if Claude Desktop is not installed.
    """
    if not claude_desktop_installed():
        raise RuntimeError(
            f"Claude Desktop not found at {CLAUDE_APP}. Install it from "
            "https://claude.ai/download"
        )
    prompt = summary_prompt(title, transcript_md, language)
    to_clipboard(prompt)
    # Bring Claude Desktop to the front; the prompt is already on the clipboard.
    subprocess.run(["open", "-a", CLAUDE_APP], check=False)
    return prompt


# ---------------------------------------------------------------------------
# Apple Notes
# ---------------------------------------------------------------------------

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _markdown_to_note_html(md: str) -> str:
    """A minimal Markdown→HTML conversion good enough for Apple Notes.

    Handles headings (``#``…), bold (``**``), horizontal rules and line breaks.
    Notes renders the ``body`` property as HTML.
    """
    import re

    html_lines = []
    for raw in md.splitlines():
        line = _html_escape(raw)
        # **bold** → <b>bold</b>
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        stripped = raw.strip()
        if stripped.startswith("### "):
            html_lines.append(f"<h3>{_html_escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{_html_escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{_html_escape(stripped[2:])}</h1>")
        elif stripped in ("---", "***", "___"):
            html_lines.append("<hr>")
        elif stripped == "":
            html_lines.append("<br>")
        else:
            html_lines.append(line + "<br>")
    return "\n".join(html_lines)


def add_to_apple_notes(
    title: str,
    transcript_md: str,
    summary_md: Optional[str] = None,
    folder: Optional[str] = None,
) -> None:
    """Create a note containing an optional summary plus the full transcript.

    Uses AppleScript (the first run prompts for Automation permission). When
    ``summary_md`` is None, a placeholder "Summary" section is added for you to
    paste Claude Desktop's reply into.
    """
    parts = [f"# {title}", ""]
    if summary_md and summary_md.strip():
        # The summary already carries its own section headings — don't wrap it
        # in another "Zusammenfassung" heading (that caused a double heading).
        parts += [summary_md.strip(), ""]
    else:
        parts += ["## Zusammenfassung", "_(keine Zusammenfassung)_", ""]
    parts += ["---", "", "## Volltext-Transkript", "", transcript_md]
    body_html = _markdown_to_note_html("\n".join(parts))

    if folder:
        make = (
            f'tell folder "{folder}" to make new note '
            "with properties {name:noteTitle, body:noteBody}"
        )
    else:
        make = "make new note with properties {name:noteTitle, body:noteBody}"

    script = (
        "on run argv\n"
        "  set noteTitle to item 1 of argv\n"
        "  set noteBody to item 2 of argv\n"
        "  tell application \"Notes\"\n"
        f"    {make}\n"
        "  end tell\n"
        "end run\n"
    )
    proc = subprocess.run(
        ["osascript", "-e", script, title, body_html],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not create the Apple Note: " + (proc.stderr.strip() or "unknown error")
        )
