"""Local LLM summarization of meeting transcripts via MLX-LM.

Default model is Qwen3-8B at 4-bit (~4.7 GB) — the strongest German/multilingual
quality in the sub-8B class that still fits a 5 GB budget, running on Apple's
MLX runtime (same stack as transcription, no extra server needed).
"""

from __future__ import annotations

import os
import re
import shutil
import stat as _stat
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A summary model: HF repo, approximate 4-bit weight size in GB (measured from
# the mlx-community repos), and a one-line note. The size drives the per-machine
# "does this fit?" check below — bigger models give better German summaries but
# need proportionally more unified memory.
ModelSpec = namedtuple("ModelSpec", "repo size_gb note")

SUMMARY_MODELS = {
    "qwen3-32b":     ModelSpec("mlx-community/Qwen3-32B-4bit", 18.4, "dense — top reasoning quality"),
    "gemma3-27b":    ModelSpec("mlx-community/gemma-3-27b-it-4bit", 16.9, "dense — excellent German / 140+ languages"),
    "qwen3-30b-a3b": ModelSpec("mlx-community/Qwen3-30B-A3B-4bit", 17.2, "MoE, ~3B active — fast + strong"),
    "mistral-24b":   ModelSpec("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", 13.3, "solid all-rounder, lighter"),
    "qwen3-14b":     ModelSpec("mlx-community/Qwen3-14B-4bit", 8.3, "balanced quality/speed"),
    "qwen3-8b":      ModelSpec("mlx-community/Qwen3-8B-4bit", 4.6, "lightweight"),
    "gemma3-4b":     ModelSpec("mlx-community/gemma-3-4b-it-4bit", 3.4, "minimal, 140+ languages"),
    "qwen3-4b":      ModelSpec("mlx-community/Qwen3-4B-4bit", 2.3, "minimal, fastest"),
    "qwen3-235b":    ModelSpec("mlx-community/Qwen3-235B-A22B-4bit", 132.3, "flagship MoE — needs a Mac Studio/Ultra"),
}

# Best → smallest. Used both for menu order and for picking the best model that
# fits a given machine when summary_model is "auto".
_BEST_FIRST = [
    "qwen3-235b", "qwen3-32b", "gemma3-27b", "qwen3-30b-a3b",
    "mistral-24b", "qwen3-14b", "qwen3-8b", "gemma3-4b", "qwen3-4b",
]

# Fraction of total unified memory we treat as usable for model weights + the
# KV cache, leaving the rest for the OS and other apps. Apple's GPU can wire
# ~75% of RAM by default; 0.70 keeps a safety margin for long transcripts.
_USABLE_FRACTION = 0.70


def system_memory_gb() -> float:
    """Total physical/unified RAM in GB for this machine (0.0 if unknown)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
        )
        return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


def model_budget_gb(total_gb: Optional[float] = None) -> float:
    """How much memory we'll allow a summary model to occupy on this machine."""
    total = system_memory_gb() if total_gb is None else total_gb
    return max(0.0, total * _USABLE_FRACTION)


def model_fits(name_or_spec, total_gb: Optional[float] = None) -> bool:
    """Whether a model (alias or ModelSpec) is small enough to run here.

    Unknown names (e.g. a custom HF repo) are assumed to fit — we can't size
    them, so we don't block the user.
    """
    spec = name_or_spec if isinstance(name_or_spec, ModelSpec) else SUMMARY_MODELS.get(name_or_spec)
    if spec is None:
        return True
    budget = model_budget_gb(total_gb)
    return budget <= 0 or spec.size_gb <= budget


def default_model(total_gb: Optional[float] = None) -> str:
    """Best-quality model alias that fits this machine (machine-variable)."""
    for alias in _BEST_FIRST:
        if model_fits(alias, total_gb):
            return alias
    return "qwen3-4b"


def model_catalog(total_gb: Optional[float] = None) -> List[Tuple[str, ModelSpec, bool]]:
    """(alias, spec, fits) for every model, best→smallest, for pickers/menus."""
    total = system_memory_gb() if total_gb is None else total_gb
    return [(a, SUMMARY_MODELS[a], model_fits(a, total)) for a in _BEST_FIRST]


# ---------------------------------------------------------------------------
# Downloaded-model management (HuggingFace cache)
# ---------------------------------------------------------------------------

def hf_cache_dir() -> Path:
    """The HuggingFace hub cache directory, honouring the usual env overrides."""
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            return Path(v).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_to_dirname(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _dirname_to_repo(name: str) -> str:
    return name[len("models--"):].replace("--", "/") if name.startswith("models--") else name


def _dir_size(path: Path) -> int:
    """On-disk bytes under ``path``. Counts real files (blobs) once; HF stores
    snapshots as symlinks into ``blobs/``, so we skip symlinks to avoid double
    counting."""
    total = 0
    for p in path.rglob("*"):
        try:
            st = p.lstat()
            if _stat.S_ISREG(st.st_mode):
                total += st.st_size
        except OSError:
            pass
    return total


def _repo_for(name_or_repo: str) -> str:
    """Resolve an alias/auto to its HF repo id; pass repo ids through."""
    spec = SUMMARY_MODELS.get(name_or_repo)
    if spec is not None:
        return spec.repo
    if not name_or_repo or name_or_repo == "auto":
        return resolve_model("auto")
    return name_or_repo


def installed_size(name_or_repo: str) -> int:
    """Bytes a model occupies in the cache, or 0 if not downloaded."""
    p = hf_cache_dir() / _repo_to_dirname(_repo_for(name_or_repo))
    return _dir_size(p) if p.is_dir() else 0


def is_installed(name_or_repo: str) -> bool:
    return (hf_cache_dir() / _repo_to_dirname(_repo_for(name_or_repo))).is_dir()


def installed_models() -> List[Dict[str, object]]:
    """Every model in the HF cache: ``{repo, alias, size, path}``, largest first.

    Includes non-summary models (e.g. Whisper) so the user can reclaim space.
    """
    cache = hf_cache_dir()
    if not cache.is_dir():
        return []
    repo_to_alias = {spec.repo: alias for alias, spec in SUMMARY_MODELS.items()}
    out = []
    for d in cache.glob("models--*"):
        if not d.is_dir():
            continue
        repo = _dirname_to_repo(d.name)
        out.append({
            "repo": repo,
            "alias": repo_to_alias.get(repo),
            "size": _dir_size(d),
            "path": d,
        })
    out.sort(key=lambda m: m["size"], reverse=True)
    return out


def uninstall_model(name_or_repo: str) -> int:
    """Delete a model from the HF cache. Returns freed bytes (0 if absent)."""
    repo = _repo_for(name_or_repo)
    p = hf_cache_dir() / _repo_to_dirname(repo)
    if not p.is_dir():
        return 0
    freed = _dir_size(p)
    shutil.rmtree(p, ignore_errors=True)
    return freed


def human_size(num_bytes: float) -> str:
    """Compact human-readable size, e.g. ``18.4 GB``."""
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

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
    "de": "Erstelle eine konkrete, inhaltsreiche Zusammenfassung des Meetings. "
          "Schreibe so, dass jemand, der nicht dabei war, den tatsächlichen "
          "Inhalt versteht — nicht nur, DASS etwas besprochen wurde, sondern WAS "
          "genau. Verwende überwiegend Stichpunkte.\n\n"
          "Regeln:\n"
          "- Sei konkret. Nenne die tatsächlichen Inhalte: konkrete Probleme, "
          "Argumente, Zahlen, Namen von Personen/Seiten/Objekten, Uhrzeiten und "
          "Fristen aus dem Transkript. Lieber ein paar Stichpunkte mehr, dafür "
          "mit Substanz.\n"
          "- VERBOTEN sind leere Meta-Formulierungen wie „Diskussion über …“, "
          "„Überlegungen zu …“, „Austausch über …“, „Erwähnung von …“. Schreibe "
          "stattdessen, was konkret gesagt, kritisiert oder beschlossen wurde.\n"
          "  Schlecht: „Diskussion über die Umsetzung von Designs und die "
          "Herausforderungen bei der Konsistenz.“\n"
          "  Gut: „Zu viele parallele Designvarianten und fehlende verbindliche "
          "Figma-Vorlagen machen es schwer, Inhalte konsistent einzubauen.“\n"
          "- Wenn das Gespräch ein zentrales Problem oder Spannungsfeld hat, "
          "benenne es zuerst klar in 1–2 Sätzen, bevor du die Einzelpunkte "
          "auflistest.\n"
          "- Das Transkript wurde automatisch aus Sprache erzeugt und enthält "
          "deshalb Hörfehler — verstümmelte Namen, falsch erkannte Fachbegriffe, "
          "Tool- oder Produktnamen, Zahlen und zusammengezogene Wörter. "
          "Korrigiere solche offensichtlichen Fehler stillschweigend aus dem "
          "Kontext (z. B. wenn ein Name mal richtig und mal falsch geschrieben "
          "ist, nimm durchgängig die richtige Form) und schreibe die "
          "wahrscheinlich gemeinte Fassung. Erfinde dabei nichts hinzu; bist du "
          "dir bei einem Begriff wirklich unsicher, gib ihn so wieder, wie er "
          "dasteht. Weise NICHT auf die Korrekturen hin.\n"
          "- Erkenne ALLE Aufgaben, Verpflichtungen und Fristen — auch in "
          "Ich-Form (z. B. „ich muss/sollte/werde …“, „Abgabe“, „fertig werden "
          "bis …“) sowie konkrete Uhrzeiten und Termine. Trage jede davon unter "
          "„Aufgaben / To-dos“ ein, mit verantwortlicher Person und Frist, falls "
          "genannt.\n"
          "- Sprecher sind im Transkript nur mit anonymen Labels gekennzeichnet "
          "(z. B. „Sprecher 1“, „Sprecher 2“, „Vor Ort 1“, „Ich“). Verwende "
          "GENAU diese Labels, wenn du eine Aufgabe zuordnest. Übernimm sie "
          "wörtlich.\n"
          "- Namen, die im Gespräch fallen (z. B. „die Seite von Julia“, „ich "
          "rede mit Daniel“), bezeichnen meist BESPROCHENE Personen oder Objekte, "
          "NICHT den Sprecher. Ordne eine Aufgabe einer benannten Person nur dann "
          "zu, wenn das Transkript ausdrücklich sagt, dass genau diese Person die "
          "Aufgabe übernimmt. Setze sonst das Sprecher-Label oder lass die "
          "Verantwortlichkeit weg.\n"
          "- Erfinde keine Verantwortlichen, keine Fristen und keine Aufgaben. "
          "Im Zweifel lieber keine Person nennen als eine falsche.\n"
          "- Eine Aufgabe gehört unter „Aufgaben / To-dos“, NICHT in die "
          "Zusammenfassung. In der Zusammenfassung nur die übergeordneten Themen.\n"
          "- Erfinde nichts; nutze nur, was im Transkript steht. Wenn ein "
          "Abschnitt wirklich leer ist, schreibe „Keine“.\n\n"
          "Antworte in genau dieser Struktur:\n"
          "## Zusammenfassung\n"
          "- Beginne mit 1–2 Sätzen zum Kernthema bzw. Kernproblem des "
          "Gesprächs.\n"
          "- Dann konkrete Stichpunkte zu den wichtigsten besprochenen Punkten, "
          "Problemen und Ergebnissen — jeweils mit Details, Namen und konkretem "
          "Inhalt, nicht nur dem Thema.\n\n"
          "## Entscheidungen\n"
          "- konkret getroffene Entscheidungen, jeweils mit dem Was und Warum "
          "(oder „Keine“)\n\n"
          "## Aufgaben / To-dos\n"
          "- Aufgabe — verantwortliche Person (Sprecher-Label aus dem "
          "Transkript), Frist (falls genannt). Beschreibe die Aufgabe konkret, "
          "nicht als „Fortsetzung der Arbeit an …“.\n\n"
          "## Offene Fragen\n"
          "- konkrete ungeklärte Punkte (oder „Keine“)",
    "en": "Write a concrete, substantive summary of the meeting. Write so that "
          "someone who was not there understands the actual content — not just "
          "THAT something was discussed, but WHAT exactly. Use mostly bullet "
          "points.\n\n"
          "Rules:\n"
          "- Be concrete. Name the actual content: specific problems, arguments, "
          "numbers, names of people/pages/objects, times and deadlines from the "
          "transcript. A few extra bullets with substance beat fewer empty ones.\n"
          "- FORBIDDEN are empty meta-phrasings like \"discussion about …\", "
          "\"considerations regarding …\", \"exchange about …\", \"mention of …\". "
          "Write instead what was concretely said, criticized or decided.\n"
          "  Bad: \"Discussion about implementing designs and the challenges of "
          "consistency.\"\n"
          "  Good: \"Too many parallel design variants and missing binding Figma "
          "templates make it hard to embed content consistently.\"\n"
          "- If the conversation has a central problem or tension, state it "
          "clearly in 1–2 sentences first, before listing the individual points.\n"
          "- The transcript was produced automatically from speech and therefore "
          "contains recognition errors — garbled names, misheard technical terms, "
          "tool or product names, numbers, and run-together words. Silently fix "
          "such obvious errors from context (e.g. if a name appears both right "
          "and wrong, use the correct form throughout) and write the most likely "
          "intended version. Do not invent anything; if you are genuinely unsure "
          "about a term, keep it as written. Do NOT point out the corrections.\n"
          "- Detect ALL tasks, commitments and deadlines — including "
          "first-person ones (e.g. \"I need to / should / will …\", \"due\", "
          "\"finish by …\") as well as specific times and dates. List each under "
          "\"Action items\" with owner and deadline if mentioned.\n"
          "- Speakers are marked in the transcript only with anonymous labels "
          "(e.g. \"Speaker 1\", \"Speaker 2\", \"On-site 1\", \"Me\"). Use "
          "EXACTLY those labels, verbatim, when you assign an owner.\n"
          "- Names dropped in conversation (e.g. \"Julia's page\", \"I'll talk to "
          "Daniel\") usually refer to people or objects being DISCUSSED, NOT to "
          "the speaker. Assign a task to a named person only if the transcript "
          "explicitly says that this person takes on the task. Otherwise use the "
          "speaker label or leave the owner out.\n"
          "- Do not invent owners, deadlines or tasks. When in doubt, name no "
          "owner rather than the wrong one.\n"
          "- A task belongs under \"Action items\", NOT in the summary. The "
          "summary holds only the high-level topics.\n"
          "- Do not invent anything; use only what is in the transcript. If a "
          "section is truly empty, write \"None\".\n\n"
          "Respond in exactly this structure:\n"
          "## Summary\n"
          "- Start with 1–2 sentences on the core topic or core problem of the "
          "conversation.\n"
          "- Then concrete bullets on the most important points, problems and "
          "outcomes discussed — each with detail, names and actual content, not "
          "just the topic.\n\n"
          "## Decisions\n"
          "- decisions actually made, each with the what and why (or \"None\")\n\n"
          "## Action items\n"
          "- task — owner (speaker label from the transcript), deadline (if "
          "mentioned). Describe the task concretely, not as \"continue work on …\".\n\n"
          "## Open questions\n"
          "- concrete unresolved points (or \"None\")",
}

_REDUCE = {
    "de": "Fasse die folgenden Teil-Zusammenfassungen zu einer einzigen, "
          "kohärenten Zusammenfassung auf Deutsch zusammen — überwiegend "
          "Stichpunkte, mit den Abschnitten Zusammenfassung, Entscheidungen, "
          "Aufgaben / To-dos und Offene Fragen. Behalte konkrete Details, Namen, "
          "Zahlen und Fristen bei; verallgemeinere nicht zu leeren "
          "Meta-Formulierungen („Diskussion über …“).",
    "en": "Combine the following partial summaries into one coherent summary — "
          "mostly bullets, with the sections Summary, Decisions, Action items "
          "and Open questions. Keep concrete details, names, numbers and "
          "deadlines; do not generalize into empty meta-phrasings "
          "(\"discussion about …\").",
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
    """Accept a friendly alias, ``"auto"``, or a full HF repo id.

    ``"auto"`` (or an empty value) resolves to the best model that fits this
    machine, so the same config behaves sensibly on a 16 GB laptop and a 64 GB
    Studio alike.
    """
    if not name or name == "auto":
        name = default_model()
    spec = SUMMARY_MODELS.get(name)
    return spec.repo if spec is not None else name


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
