"""Local LLM summarization of meeting transcripts via MLX-LM.

Default model is Qwen3-8B at 4-bit (~4.7 GB) — the strongest German/multilingual
quality in the sub-8B class that still fits a 5 GB budget, running on Apple's
MLX runtime (same stack as transcription, no extra server needed).
"""

from __future__ import annotations

import re
from typing import List, Optional

# Recommended local models, all ≤5 GB at 4-bit. First that loads wins as default.
SUMMARY_MODELS = {
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",        # ~4.7 GB, best quality
    "qwen3-4b": "mlx-community/Qwen3-4B-4bit",        # ~2.5 GB, faster
    "gemma3-4b": "mlx-community/gemma-3-4b-it-4bit",  # ~2.6 GB, 140+ languages
}

# Roughly how many characters we feed per chunk before map-reduce kicks in.
_CHUNK_CHARS = 24_000

_SYSTEM = {
    "de": "Du bist ein professioneller Protokollant. Du fasst Meetings sachlich "
          "und strukturiert auf Deutsch zusammen, überwiegend als Stichpunkte. "
          "Erfinde nichts — verwende nur, was im Transkript steht.",
    "en": "You are a professional meeting note-taker. You summarize meetings "
          "factually and in a structured way, mostly as bullet points. Do not "
          "invent anything — use only what is in the transcript.",
}

# The default, user-editable instruction (no transcript — that's appended).
_DEFAULT_INSTRUCTION = {
    "de": "Erstelle eine professionelle, strukturierte Zusammenfassung des "
          "Meetings. Verwende überwiegend Stichpunkte.\n\n"
          "Regeln:\n"
          "- Erkenne ALLE Aufgaben, Verpflichtungen und Fristen — auch in "
          "Ich-Form (z. B. „ich muss/sollte/werde …“, „Abgabe“, „fertig werden "
          "bis …“) sowie konkrete Uhrzeiten, Termine und Personennamen. Trage "
          "jede davon unter „Aufgaben / To-dos“ ein, mit verantwortlicher Person "
          "und Frist, falls genannt.\n"
          "- Eine Aufgabe gehört unter „Aufgaben / To-dos“, NICHT in die "
          "Zusammenfassung. In der Zusammenfassung nur die übergeordneten Themen.\n"
          "- Erfinde nichts; nutze nur, was im Transkript steht. Wenn ein "
          "Abschnitt wirklich leer ist, schreibe „Keine“.\n\n"
          "Antworte in genau dieser Struktur:\n"
          "## Zusammenfassung\n"
          "- 2–4 übergeordnete Stichpunkte zu Themen und Ergebnissen\n\n"
          "## Entscheidungen\n"
          "- getroffene Entscheidungen (oder „Keine“)\n\n"
          "## Aufgaben / To-dos\n"
          "- Aufgabe — verantwortliche Person, Frist (falls genannt)\n\n"
          "## Offene Fragen\n"
          "- ungeklärte Punkte (oder „Keine“)",
    "en": "Write a professional, structured summary of the meeting. Use mostly "
          "bullet points.\n\n"
          "Rules:\n"
          "- Detect ALL tasks, commitments and deadlines — including "
          "first-person ones (e.g. \"I need to / should / will …\", \"due\", "
          "\"finish by …\") as well as specific times, dates and people's names. "
          "List each under \"Action items\" with owner and deadline if mentioned.\n"
          "- A task belongs under \"Action items\", NOT in the summary. The "
          "summary holds only the high-level topics.\n"
          "- Do not invent anything; use only what is in the transcript. If a "
          "section is truly empty, write \"None\".\n\n"
          "Respond in exactly this structure:\n"
          "## Summary\n"
          "- 2–4 high-level bullets on topics and outcomes\n\n"
          "## Decisions\n"
          "- decisions made (or \"None\")\n\n"
          "## Action items\n"
          "- task — owner, deadline (if mentioned)\n\n"
          "## Open questions\n"
          "- unresolved points (or \"None\")",
}

_REDUCE = {
    "de": "Fasse die folgenden Teil-Zusammenfassungen zu einer einzigen, "
          "kohärenten Zusammenfassung auf Deutsch zusammen — überwiegend "
          "Stichpunkte, mit den Abschnitten Zusammenfassung, Entscheidungen, "
          "Aufgaben / To-dos und Offene Fragen.",
    "en": "Combine the following partial summaries into one coherent summary — "
          "mostly bullets, with the sections Summary, Decisions, Action items "
          "and Open questions.",
}

_TRANSCRIPT_LABEL = {"de": "Transkript", "en": "Transcript"}


def default_prompt(language: Optional[str] = None) -> str:
    """The default, user-editable summarization instruction for a language."""
    return _DEFAULT_INSTRUCTION.get(language or "", _DEFAULT_INSTRUCTION["en"])


def available() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_model(name: Optional[str]) -> str:
    """Accept a friendly alias or a full HF repo id."""
    if not name:
        return SUMMARY_MODELS["qwen3-8b"]
    return SUMMARY_MODELS.get(name, name)


def _strip_thinking(text: str) -> str:
    # Qwen3 may emit <think>…</think> reasoning; drop it.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _chunks(text: str, size: int = _CHUNK_CHARS) -> List[str]:
    if len(text) <= size:
        return [text]
    out, cur = [], []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > size and cur:
            out.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        out.append("".join(cur))
    return out


def transcript_body(md: str) -> str:
    """Extract just the spoken content from a transcript .md (drop metadata
    and any existing summary section)."""
    if "## Transcript" in md:
        return md.split("## Transcript", 1)[1].strip()
    if "\n---\n" in md:
        return md.split("\n---\n", 1)[1].strip()
    return md.strip()


def extract_summary(md: str) -> str:
    """Return the summary section already embedded in a transcript .md, if any.

    write_transcript lays out: metadata ``---`` summary ``---`` ``## Transcript``.
    """
    if "## Transcript" not in md:
        return ""
    head = md.split("## Transcript", 1)[0]
    parts = head.split("\n---\n")
    if len(parts) >= 3:
        return parts[1].strip()
    return ""


def summarize_markdown(md: str, model: Optional[str] = None, language: Optional[str] = None) -> str:
    """Summarize a transcript .md by first stripping it to spoken content."""
    return summarize(transcript_body(md), model=model, language=language)


def summarize(
    text: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Summarize transcript ``text`` with a local MLX model. Returns markdown.

    ``prompt`` overrides the default instruction (the transcript is appended
    automatically); when None, :func:`default_prompt` is used.
    """
    if not available():
        raise RuntimeError("Summarization needs mlx-lm: pip install mlx-lm")
    if not text.strip():
        return ""

    from mlx_lm import generate, load

    repo = resolve_model(model)
    lm, tokenizer = load(repo)
    lang = language or ""
    system = _SYSTEM.get(lang, _SYSTEM["en"])
    instruction = (prompt or default_prompt(language)).strip()
    label = _TRANSCRIPT_LABEL.get(lang, "Transcript")

    def _run(user: str, max_tokens: int = 900) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            chat = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        out = generate(lm, tokenizer, prompt=chat, max_tokens=max_tokens, verbose=False)
        return _strip_thinking(out)

    parts = _chunks(text)
    if len(parts) == 1:
        return _run(f"{instruction}\n\n{label}:\n{parts[0]}")

    # Map-reduce for long meetings: summarize each chunk, then combine.
    partials = [_run(f"{instruction}\n\n{label}:\n{p}", max_tokens=500) for p in parts]
    reduce_instr = _REDUCE.get(lang, _REDUCE["en"])
    return _run(f"{reduce_instr}\n\n{chr(10).join(partials)}")
