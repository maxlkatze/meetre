"""Local LLM summarization of meeting transcripts via MLX-LM.

Models are chosen per machine from SUMMARY_MODELS (K2 Horizon 7B preferred,
plus Qwen3.5 + Gemma 4); "auto" picks the preferred model that fits the
available unified memory. Everything runs on Apple's MLX runtime — same stack as
transcription, no extra server needed.
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

from .mlxthread import on_mlx_thread

# A summary model: HF repo, approximate 4-bit weight size in GB (measured from
# the mlx-community repos), a one-line note, and whether it is a reasoning
# ("thinking") model. The size drives the per-machine "does this fit?" check
# below — bigger models give better German summaries but need proportionally
# more unified memory. ``thinks`` makes summarize() turn on the model's internal
# reasoning pass (the <think> block is generated, then stripped from the output).
# ``revision`` pins a repo to a reviewed commit — required for repos that ship
# their own model code (mlx-lm executes it on load), so a later upload can't
# swap in different code. ``template_kwargs`` are extra chat-template options.
ModelSpec = namedtuple("ModelSpec", "repo size_gb note thinks revision template_kwargs")
ModelSpec.__new__.__defaults__ = (False, None, None)

# Current generation (June 2026): Qwen3.5 + Gemma 4. Qwen3.5 are hybrid models
# but we run them in direct (non-reasoning) mode — for summarization the hidden
# reasoning pass adds latency and can truncate/leak with no quality win, so all
# models below answer directly. Gemma 4 has the strongest multilingual coverage.
SUMMARY_MODELS = {
    # --- Qwen3.5 (run in direct mode) ----------------------------------
    "qwen3.5-397b":  ModelSpec("mlx-community/Qwen3.5-397B-A17B-4bit", 210.0, "flagship MoE — Mac Studio Ultra only"),
    "qwen3.5-122b":  ModelSpec("mlx-community/Qwen3.5-122B-A10B-MLX-4bit", 66.0, "large MoE — high-RAM Studio"),
    "qwen3.5-35b":   ModelSpec("mlx-community/Qwen3.5-35B-A3B-4bit", 19.6, "MoE, ~3B active — fast, best all-round"),
    "qwen3.5-27b":   ModelSpec("mlx-community/Qwen3.5-27B-4bit", 15.0, "dense — top quality"),
    "qwen3.5-9b":    ModelSpec("mlx-community/Qwen3.5-9B-4bit", 5.0, "balanced — fits 16 GB"),
    "qwen3.5-4b":    ModelSpec("mlx-community/Qwen3.5-4B-MLX-4bit", 2.4, "small — fast"),
    "qwen3.5-2b":    ModelSpec("mlx-community/Qwen3.5-2B-MLX-4bit", 1.3, "minimal — fastest"),
    # --- Gemma 4 (instruction, strongest multilingual) -----------------
    "gemma4-26b":    ModelSpec("mlx-community/gemma-4-26b-a4b-it-4bit", 14.5, "MoE — excellent German / 140+ languages"),
    "gemma4-12b":    ModelSpec("mlx-community/gemma-4-12B-4bit", 6.8, "dense — strong multilingual, lighter"),
    "gemma4-e4b":    ModelSpec("mlx-community/gemma-4-e4b-it-4bit", 3.4, "minimal, 140+ languages"),
    # --- K2 Horizon (IFM, Sept 2026; always reasons before answering) ---
    # Not yet a native mlx-lm architecture: the repo ships k2_horizon.py, which
    # mlx-lm runs on load. Pinned to the commit whose code was reviewed.
    "k2-horizon-7b": ModelSpec("mlx-community/K2-Horizon-7B-4bit", 7.2,
                               "reasoning — excellent summary writing", True,
                               "c8698376a102bc5e7ad49089b21b62c096a2506c",
                               # "low": ~8 s for a 3-min meeting; "medium"
                               # overran the think budget (~160 s) for the
                               # same text, so it's not worth it here.
                               {"reasoning_effort": "low"}),
    # --- Mistral (no newer 2026 release; solid all-rounder) ------------
    "mistral-24b":   ModelSpec("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", 13.3, "solid all-rounder, no reasoning"),
    # --- Legacy aliases: not shown in the menu, kept so an existing
    #     config.summary_model still resolves to a real repo. -----------
    "qwen3-32b":     ModelSpec("mlx-community/Qwen3-32B-4bit", 18.4, "legacy — Qwen3 dense"),
    "gemma3-27b":    ModelSpec("mlx-community/gemma-3-27b-it-4bit", 16.9, "legacy — Gemma 3"),
    "qwen3-30b-a3b": ModelSpec("mlx-community/Qwen3-30B-A3B-4bit", 17.2, "legacy — Qwen3 MoE"),
    "qwen3-14b":     ModelSpec("mlx-community/Qwen3-14B-4bit", 8.3, "legacy — Qwen3"),
    "qwen3-8b":      ModelSpec("mlx-community/Qwen3-8B-4bit", 4.6, "legacy — Qwen3"),
    "gemma3-4b":     ModelSpec("mlx-community/gemma-3-4b-it-4bit", 3.4, "legacy — Gemma 3"),
    "qwen3-4b":      ModelSpec("mlx-community/Qwen3-4B-4bit", 2.3, "legacy — Qwen3"),
    "qwen3-235b":    ModelSpec("mlx-community/Qwen3-235B-A22B-4bit", 132.3, "legacy — Qwen3 flagship MoE"),
}

# Preferred → fallback. Used both for menu order and for picking the model that
# "auto" runs on a given machine. K2 Horizon 7B leads: it writes the best
# summaries we've seen and, at ~7 GB, fits almost every Mac; the rest are ranked
# best → smallest for machines where it doesn't. Only current-generation models
# appear here; legacy aliases above stay resolvable but off the menu.
_BEST_FIRST = [
    "k2-horizon-7b",
    "qwen3.5-397b", "qwen3.5-122b", "qwen3.5-35b", "qwen3.5-27b",
    "gemma4-26b", "mistral-24b", "gemma4-12b", "qwen3.5-9b",
    "gemma4-e4b", "qwen3.5-4b", "qwen3.5-2b",
]

# Fraction of total unified memory we treat as usable for model weights + the
# KV cache, leaving the rest for the OS and other apps. Apple's GPU can wire
# ~75% of RAM by default; 0.70 keeps a safety margin for long transcripts.
_USABLE_FRACTION = 0.70


def system_memory_gb() -> float:
    """Total physical/unified RAM in GB for this machine (0.0 if unknown).

    Detection must work when the app is launched by launchd / "start at login"
    or after a "Restart meetre", where ``PATH`` is empty — a bare
    ``subprocess.run(["sysctl", ...])`` then raises ``FileNotFoundError`` and we
    used to fall back to 0.0, which made *every* model look like it fits (the
    budget check treats 0 as "unknown → allow"). So prefer the in-process
    ``os.sysconf`` route, which needs no subprocess at all, and only fall back
    to an absolute-path ``sysctl`` if that is unavailable.
    """
    # 1) In-process: page size × physical pages. No PATH, no subprocess.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass

    # 2) Fallback: sysctl by absolute path so an empty PATH can't hide it.
    for sysctl in ("/usr/sbin/sysctl", "/sbin/sysctl", "sysctl"):
        try:
            out = subprocess.run(
                [sysctl, "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
            )
            value = out.stdout.strip()
            if value:
                return int(value) / (1024 ** 3)
        except (OSError, ValueError):
            continue
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
    "de": "Fasse das Meeting für jemanden zusammen, der nicht dabei war. Er soll "
          "danach wissen, WAS konkret gesagt, entschieden und vereinbart wurde — "
          "nicht nur, DASS über etwas gesprochen wurde.\n\n"
          "INHALT\n"
          "- Verdichte, statt nachzuerzählen: Die Zusammenfassung ist deutlich "
          "kürzer als das Transkript. Ein kurzes Gespräch ergibt 2–4 Stichpunkte.\n"
          "- Sei konkret: Nenne Probleme, Argumente, Zahlen, Beträge, Namen von "
          "Personen/Projekten/Tools, Termine und Fristen so, wie sie fallen.\n"
          "- Jeder Stichpunkt bringt eine neue Information. Nichts doppelt — "
          "weder innerhalb eines Abschnitts noch über Abschnitte hinweg.\n"
          "- Keine leeren Meta-Sätze („Diskussion über …“, „Austausch zu …“, "
          "„Erwähnung von …“). Schreibe den Inhalt selbst.\n"
          "  Schlecht: „Diskussion über Designs und Konsistenz.“\n"
          "  Gut: „Zu viele parallele Designvarianten und keine verbindlichen "
          "Figma-Vorlagen — Inhalte lassen sich nicht konsistent einbauen.“\n\n"
          "TRANSKRIPTIONSFEHLER\n"
          "- Das Transkript stammt aus automatischer Spracherkennung und enthält "
          "Hörfehler (Namen, Fachbegriffe, Produktnamen, Zahlen). Korrigiere "
          "offensichtliche Fehler stillschweigend aus dem Kontext und verwende "
          "eine Schreibweise durchgängig. Bei echter Unsicherheit übernimm den "
          "Begriff unverändert. Erwähne die Korrekturen nicht.\n\n"
          "AUFGABEN UND VERANTWORTLICHE\n"
          "- Erfasse JEDE Aufgabe, Zusage und Frist — auch beiläufige in "
          "Ich-Form („ich schicke …“, „ich muss bis Freitag …“) und Bitten "
          "(„Kannst du …“ → zuständig ist der/die Angesprochene).\n"
          "- Sprecher heißen im Transkript nur „Sprecher 1“, „Vor Ort 1“, „Ich“ "
          "usw. Übernimm diese Labels wörtlich als Verantwortliche.\n"
          "- Ein im Gespräch genannter Name („die Seite von Julia“) ist meist das "
          "THEMA, nicht der Sprecher. Ordne einer benannten Person nur zu, was "
          "sie laut Transkript ausdrücklich übernimmt.\n"
          "- Unklar, wer zuständig ist? Dann lass die Person weg. Nie raten, nie "
          "Platzhalter wie „unbekannt“. Erfinde keine Aufgaben oder Fristen.\n\n"
          "FORMAT — antworte NUR mit diesen vier Abschnitten, ohne Vorrede:\n"
          "## Zusammenfassung\n"
          "Ein bis zwei Sätze zum Kernthema bzw. Kernproblem. Danach nur bei "
          "Bedarf Stichpunkte mit weiteren wesentlichen Punkten (ohne den "
          "Einleitungssatz zu wiederholen). Aufgaben gehören nicht hierher.\n\n"
          "## Entscheidungen\n"
          "- Was wurde beschlossen und warum (oder „Keine“)\n\n"
          "## Aufgaben / To-dos\n"
          "- Konkrete Aufgabe — Sprecher-Label, Frist (falls genannt) "
          "(oder „Keine“)\n\n"
          "## Offene Fragen\n"
          "- Konkret ungeklärte Punkte (oder „Keine“)",
    "en": "Summarize the meeting for someone who was not there. Afterwards they "
          "should know WHAT was concretely said, decided and agreed — not just "
          "THAT something was discussed.\n\n"
          "CONTENT\n"
          "- Condense, don't retell: the summary is much shorter than the "
          "transcript. A short conversation yields 2–4 bullets.\n"
          "- Be concrete: name problems, arguments, numbers, amounts, names of "
          "people/projects/tools, dates and deadlines as they come up.\n"
          "- Every bullet adds new information. Nothing twice — not within a "
          "section and not across sections.\n"
          "- No empty meta-phrases (\"discussion about …\", \"exchange on …\", "
          "\"mention of …\"). Write the content itself.\n"
          "  Bad: \"Discussion about designs and consistency.\"\n"
          "  Good: \"Too many parallel design variants and no binding Figma "
          "templates — content can't be embedded consistently.\"\n\n"
          "TRANSCRIPTION ERRORS\n"
          "- The transcript comes from automatic speech recognition and contains "
          "mishearings (names, technical terms, product names, numbers). Silently "
          "fix obvious errors from context and use one spelling throughout. If "
          "genuinely unsure, keep the term as written. Don't mention the fixes.\n\n"
          "TASKS AND OWNERS\n"
          "- Capture EVERY task, commitment and deadline — including casual "
          "first-person ones (\"I'll send …\", \"I need to … by Friday\") and "
          "requests (\"Can you …\" → the owner is the person addressed).\n"
          "- Speakers appear in the transcript only as \"Speaker 1\", "
          "\"On-site 1\", \"Me\", etc. Use those labels verbatim as owners.\n"
          "- A name dropped in conversation (\"Julia's page\") is usually the "
          "TOPIC, not the speaker. Assign something to a named person only if the "
          "transcript explicitly says they take it on.\n"
          "- Owner unclear? Leave it out. Never guess, never use placeholders "
          "like \"unknown\". Don't invent tasks or deadlines.\n\n"
          "FORMAT — respond with ONLY these four sections, no preamble:\n"
          "## Summary\n"
          "One or two sentences on the core topic or core problem. Then, only if "
          "needed, bullets with further essential points (without restating the "
          "opening). Tasks don't go here.\n\n"
          "## Decisions\n"
          "- What was decided and why (or \"None\")\n\n"
          "## Action items\n"
          "- Concrete task — speaker label, deadline (if mentioned) "
          "(or \"None\")\n\n"
          "## Open questions\n"
          "- Concrete unresolved points (or \"None\")",
}

# Repeated AFTER the transcript: small local models drift from instructions
# that sit thousands of tokens back, so restate the essentials last.
_REMINDER = {
    "de": "Schreibe jetzt die Zusammenfassung des obigen Transkripts gemäß der "
          "Anleitung: konkret, verdichtet, ohne Wiederholungen, genau die vier "
          "Abschnitte (Zusammenfassung, Entscheidungen, Aufgaben / To-dos, "
          "Offene Fragen), auf Deutsch.",
    "en": "Now write the summary of the transcript above following the "
          "instructions: concrete, condensed, no repetition, exactly the four "
          "sections (Summary, Decisions, Action items, Open questions).",
}

_TRANSCRIPT_LABEL = {"de": "Transkript", "en": "Transcript"}

_TITLE_INSTRUCTION = {
    "de": "Gib einen kurzen, konkreten Titel für dieses Meeting an — 3 bis 8 "
          "Wörter, der das Kernthema benennt. Keine Anführungszeichen, kein "
          "Punkt am Ende, keine Vorrede. Antworte mit NUR dem Titel.",
    "en": "Give a short, concrete title for this meeting — 3 to 8 words naming "
          "the core topic. No quotes, no trailing period, no preamble. Respond "
          "with ONLY the title.",
}

_REDUCE = {
    "de": "Unten stehen Teil-Zusammenfassungen aufeinanderfolgender Abschnitte "
          "EINES Meetings. Führe sie zu einer einzigen Zusammenfassung auf "
          "Deutsch zusammen, mit genau den Abschnitten ## Zusammenfassung, "
          "## Entscheidungen, ## Aufgaben / To-dos und ## Offene Fragen.\n"
          "- Fasse Doppeltes zusammen; eine später geklärte Frage ist keine "
          "offene Frage mehr.\n"
          "- Behalte konkrete Details, Namen, Zahlen, Sprecher-Labels und "
          "Fristen bei; verallgemeinere nicht zu leeren Meta-Formulierungen "
          "(„Diskussion über …“).\n"
          "- Übernimm jede Aufgabe genau einmal.",
    "en": "Below are partial summaries of consecutive parts of ONE meeting. "
          "Merge them into a single summary with exactly the sections "
          "## Summary, ## Decisions, ## Action items and ## Open questions.\n"
          "- Merge duplicates; a question answered in a later part is no longer "
          "open.\n"
          "- Keep concrete details, names, numbers, speaker labels and "
          "deadlines; don't generalize into empty meta-phrases (\"discussion "
          "about …\").\n"
          "- Keep every action item exactly once.",
}


def default_prompt(language: Optional[str] = None) -> str:
    """The default, user-editable summarization instruction for a language."""
    return _DEFAULT_INSTRUCTION.get(language or "", _DEFAULT_INSTRUCTION["en"])


def available() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def model_thinks(name: Optional[str]) -> bool:
    """Whether a model (alias or 'auto') reasons before answering.

    Unknown/custom repos are detected heuristically from the repo name so a
    pasted "…-Thinking…" repo still gets a reasoning pass and token headroom.
    """
    alias = default_model() if (not name or name == "auto") else name
    spec = SUMMARY_MODELS.get(alias)
    if spec is not None:
        return bool(spec.thinks)
    return "think" in alias.lower()


def _spec(name_or_repo: Optional[str]) -> Optional[ModelSpec]:
    """The catalog entry for an alias, ``"auto"``, or a repo id (None if unknown)."""
    if not name_or_repo or name_or_repo == "auto":
        name_or_repo = default_model()
    spec = SUMMARY_MODELS.get(name_or_repo)
    if spec is not None:
        return spec
    return next((s for s in SUMMARY_MODELS.values() if s.repo == name_or_repo), None)


def model_revision(name_or_repo: Optional[str]) -> Optional[str]:
    """Pinned commit for a model (None = latest)."""
    spec = _spec(name_or_repo)
    return spec.revision if spec is not None else None


def _load_model(model: Optional[str]):
    """mlx_lm.load with the model's pinned revision.

    Repos that ship their own architecture code (K2 Horizon's k2_horizon.py)
    need ``trust_remote_code``; it is granted ONLY to specs pinned to a
    reviewed commit, so unreviewed code can never run. Older mlx-lm releases
    have no such flag (and executed the code unconditionally) — fall back.
    """
    from mlx_lm import load

    repo = resolve_model(model)
    rev = model_revision(model)
    if rev:
        try:
            return load(repo, revision=rev, trust_remote_code=True)
        except TypeError:
            pass
    return load(repo, revision=rev)


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
    """Remove a reasoning model's chain-of-thought from its output.

    Reasoning models (Qwen3.5) emit a <think>…</think> block. Their chat
    template pre-fills the OPENING <think> as part of the prompt, so the
    generated text usually contains only the reasoning followed by a trailing
    </think> and then the answer — with no opening tag. Drop everything up to
    and including the last closing tag, then also clear any fully-tagged block.
    K2 Horizon uses ``<ifm|think>`` / ``<ifm|think_fast>`` / ``<ifm|think_faster>``.
    """
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    text = re.sub(r"<(?:ifm\|)?think(?:_fast|_faster)?>.*?</(?:ifm\|)?think(?:_fast|_faster)?>",
                  "", text, flags=re.DOTALL)
    return text.replace("<|ifm|im_end|>", "").strip()


_THINK_CLOSE = re.compile(r"</(?:ifm\|)?think(?:_fast|_faster)?>")


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


def meeting_meta(md: str) -> Tuple[Optional[str], Optional[str]]:
    """(meeting title, date 'YYYY-MM-DD') from a transcript .md header.

    write_transcript writes ``# {title}`` then ``- **Date:** YYYY-MM-DD HH:MM:SS``.
    Either value is None if not found.
    """
    title = date = None
    for line in md.splitlines():
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        elif date is None and "**Date:**" in line:
            date = line.split("**Date:**", 1)[1].strip()[:10]
        if title and date:
            break
    return title, date


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


@on_mlx_thread
def summarize(
    text: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    on_progress=None,
) -> str:
    """Summarize transcript ``text`` with a local MLX model. Returns markdown.

    ``prompt`` overrides the default instruction (the transcript is appended
    automatically); when None, :func:`default_prompt` is used.

    ``on_progress(phase, text, stage)`` streams the output as it is generated:
    ``phase`` is "thinking" (``text`` = the reasoning so far) or "writing"
    (``text`` = the answer so far); ``stage`` is "" for a single pass, or
    "part 2/5" / "merge" while long meetings are summarized in chunks.
    Called from the generating thread.
    """
    if not available():
        raise RuntimeError("Summarization needs mlx-lm: pip install mlx-lm")
    if not text.strip():
        return ""

    from mlx_lm import stream_generate

    repo = resolve_model(model)
    thinks = model_thinks(model)
    spec = _spec(model)
    tmpl = dict((spec.template_kwargs if spec else None) or {})
    stage = {"label": ""}
    # Models with a reasoning_effort knob can't switch thinking off; their
    # fallback is a shorter reasoning pass instead of a direct answer.
    always_thinks = "reasoning_effort" in tmpl
    lm, tokenizer = _load_model(model)
    lang = language or ""
    system = _SYSTEM.get(lang, _SYSTEM["en"])
    instruction = (prompt or default_prompt(language)).strip()
    label = _TRANSCRIPT_LABEL.get(lang, "Transcript")
    # Custom prompts may ask for a different structure — no reminder for them.
    reminder = "" if prompt else _REMINDER.get(lang, _REMINDER["en"])

    # Reasoning models spend tokens on a hidden <think> pass before the answer,
    # so they need far more headroom. Non-thinking models answer directly.
    answer_budget = 900
    reduce_budget = 500
    think_budget = 4000  # extra tokens reserved for the reasoning pass

    def _generate(user: str, max_tokens: int, use_thinking: bool, **extra) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            chat = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=use_thinking,
                **{**tmpl, **extra},
            )
        except TypeError:
            chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        raw = ""
        for resp in stream_generate(lm, tokenizer, prompt=chat, max_tokens=max_tokens):
            raw += resp.text
            if on_progress is not None and resp.text:
                if use_thinking and not _THINK_CLOSE.search(raw):
                    on_progress("thinking", raw, stage["label"])
                else:
                    on_progress("writing", _strip_thinking(raw), stage["label"])
        return raw

    def _run(user: str, max_tokens: int) -> str:
        if thinks:
            raw = _generate(user, max_tokens + think_budget, True)
            # A reasoning model's answer follows the closing </think>. If the
            # tag is missing the model ran out of budget mid-thought (no answer
            # was produced) — never dump the raw reasoning; fall back instead.
            answer = _strip_thinking(raw) if _THINK_CLOSE.search(raw) else ""
            if answer:
                return answer
            if always_thinks:
                # Can't turn reasoning off: retry with the shortest pass, and
                # give up (empty) rather than return half a thought.
                raw = _generate(user, max_tokens + think_budget, True,
                                reasoning_effort="low")
                return _strip_thinking(raw) if _THINK_CLOSE.search(raw) else ""
            # Reasoning truncated or produced nothing usable: redo the call with
            # the thinking pass off so the model answers directly.
            raw = _generate(user, max_tokens, False)
        else:
            raw = _generate(user, max_tokens, False)
        return _strip_thinking(raw)

    parts = _chunks(text)
    if len(parts) == 1:
        return _run(f"{instruction}\n\n{label}:\n{parts[0]}\n\n{reminder}".rstrip(), answer_budget)

    # Map-reduce for long meetings: summarize each chunk, then combine.
    partials = []
    for i, p in enumerate(parts, 1):
        stage["label"] = f"part {i}/{len(parts)}"
        partials.append(_run(f"{instruction}\n\n{label}:\n{p}\n\n{reminder}".rstrip(), reduce_budget))
    stage["label"] = "merge"
    reduce_instr = _REDUCE.get(lang, _REDUCE["en"])
    return _run(f"{reduce_instr}\n\n{chr(10).join(partials)}", answer_budget)


@on_mlx_thread
def generate_title(
    text: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Generate a short meeting title from ``text`` (a summary or transcript).

    Uses the same local MLX model as :func:`summarize`. Returns a single
    cleaned-up line; ``""`` if unavailable or the input is empty.
    """
    if not available():
        raise RuntimeError("Title generation needs mlx-lm: pip install mlx-lm")
    if not text.strip():
        return ""

    from mlx_lm import generate

    repo = resolve_model(model)
    thinks = model_thinks(model)
    spec = _spec(model)
    tmpl = dict((spec.template_kwargs if spec else None) or {})
    if "reasoning_effort" in tmpl:
        tmpl["reasoning_effort"] = "low"  # a title doesn't need deep reasoning
    lm, tokenizer = _load_model(model)
    lang = language or ""
    system = _SYSTEM.get(lang, _SYSTEM["en"])
    instruction = _TITLE_INSTRUCTION.get(lang, _TITLE_INSTRUCTION["en"])
    label = _TRANSCRIPT_LABEL.get(lang, "Transcript")

    # A title only needs a handful of words; keep the model focused on the start
    # of the content so a long meeting stays cheap.
    snippet = text.strip()[:6000]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"{instruction}\n\n{label}:\n{snippet}"}]
    try:
        chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=thinks, **tmpl)
    except TypeError:
        chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    max_tokens = 40 + (4000 if thinks else 0)
    raw = generate(lm, tokenizer, prompt=chat, max_tokens=max_tokens, verbose=False)
    return _clean_title(_strip_thinking(raw))


def _clean_title(text: str) -> str:
    """Reduce model output to a single, filename-friendly title line."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    # Drop common lead-ins, surrounding quotes/markdown and a trailing period.
    line = re.sub(r'^(title|titel)\s*[:\-]\s*', "", line, flags=re.IGNORECASE)
    # Strip surrounding quotes/markdown and trailing punctuation together so a
    # model that emits e.g. "Q3 Budget Review". loses both the period and quote.
    return line.strip().strip('"\'`*#.').strip()


# ---------------------------------------------------------------------------
# Interactive chat (menubar "Chat with K2…" window)
# ---------------------------------------------------------------------------

_chat_model = {"key": None, "lm": None, "tok": None}


def _load_for_chat(model: Optional[str]):
    """Load (and keep) the chat model so follow-up questions start instantly.

    Summaries load and drop their model per meeting; the chat window keeps one
    resident while it is open and calls :func:`release_chat_model` on close.
    """
    repo = resolve_model(model)
    rev = model_revision(model)
    if _chat_model["key"] != (repo, rev):
        release_chat_model()
        lm, tok = _load_model(model)
        _chat_model.update(key=(repo, rev), lm=lm, tok=tok)
    return _chat_model["lm"], _chat_model["tok"]


@on_mlx_thread
def release_chat_model() -> None:
    """Free the resident chat model (and the Metal buffers it holds)."""
    _chat_model.update(key=None, lm=None, tok=None)
    try:
        import gc

        import mlx.core as mx

        gc.collect()
        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


_CHAT_SYSTEM = {
    "de": "Du bist K2, ein hilfreicher Assistent in meetre, einer lokalen "
          "Meeting-App. Antworte präzise und auf Deutsch, außer der Nutzer "
          "schreibt in einer anderen Sprache. Wenn ein Meeting-Transkript "
          "mitgegeben wird, stütze dich darauf und erfinde nichts; zitiere bei "
          "Bedarf Sprecher und Zeitstempel. Wenn Websuche-Ergebnisse mitgegeben "
          "werden, nenne die verwendeten Quellen als [n].",
    "en": "You are K2, a helpful assistant inside meetre, a local meeting app. "
          "Answer concisely. When a meeting transcript is provided, ground your "
          "answers in it and don't invent anything; cite speakers and "
          "timestamps when useful. When web search results are provided, cite "
          "the sources you used as [n].",
}

# Transcript budget for the chat context (~15k tokens): long meetings keep
# their beginning and end, which is where agendas and wrap-ups live.
_CHAT_TRANSCRIPT_CHARS = 60_000


def _fit_transcript(text: str, limit: int = _CHAT_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n[… middle of the meeting omitted for length …]\n\n{text[-half:]}"


@on_mlx_thread
def chat(
    history: List[Dict[str, str]],
    model: Optional[str] = None,
    language: Optional[str] = None,
    transcript: str = "",
    web_results: str = "",
    on_progress=None,
    should_stop=None,
    max_tokens: int = 1500,
    tools=None,
    on_tool=None,
    max_rounds: int = 6,
) -> str:
    """Answer the last user turn of ``history`` ([{role, content}, …]).

    ``transcript`` (a meeting .md body) and ``web_results`` (formatted search
    hits) are injected as context. ``on_progress(phase, text)`` streams like
    :func:`summarize` ("thinking" / "writing"); ``should_stop()`` → True aborts
    generation and returns what was written so far.

    ``tools`` (a :class:`meetre.tools.ToolBox`) enables native tool calling:
    the model may emit tool calls, which are executed (``on_tool(label,
    result_or_None)`` reports start/finish) and fed back, up to ``max_rounds``.
    """
    if not available():
        raise RuntimeError("Chat needs mlx-lm: pip install mlx-lm")
    from mlx_lm import stream_generate

    from . import tools as _tools

    lm, tokenizer = _load_for_chat(model)
    spec = _spec(model)
    tmpl = dict((spec.template_kwargs if spec else None) or {})
    thinks = model_thinks(model)
    system = _CHAT_SYSTEM.get(language or "", _CHAT_SYSTEM["en"])
    schemas = tools.schemas() if tools is not None else []
    if schemas:
        system += ("\n\nYou can call tools (web search, reading the user's files, "
                   "MCP servers). Use them when the question needs fresh or "
                   "external information; don't call tools for things you already know.")
    if transcript.strip():
        label = _TRANSCRIPT_LABEL.get(language or "", "Transcript")
        system += f"\n\n{label}:\n{_fit_transcript(transcript.strip())}"

    messages = [{"role": "system", "content": system}] + [dict(m) for m in history]
    if web_results.strip() and messages[-1]["role"] == "user":
        messages[-1]["content"] = (f"Web search results:\n{web_results.strip()}\n\n"
                                   f"Question: {messages[-1]['content']}")

    def render() -> str:
        kw = dict(tmpl)
        if schemas:
            kw.update(tools=schemas)  # K2 native (xml) tool-call format
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=thinks, **kw)
        except TypeError:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    budget = max_tokens + (4000 if thinks else 0)
    for _round in range(max_rounds + 1):
        raw = ""
        for resp in stream_generate(lm, tokenizer, prompt=render(), max_tokens=budget):
            raw += resp.text
            if on_progress is not None and resp.text:
                if thinks and not _THINK_CLOSE.search(raw):
                    on_progress("thinking", raw)
                else:
                    on_progress("writing", _tools.strip_tool_calls(_strip_thinking(raw)))
            if should_stop is not None and should_stop():
                break
        if thinks and not _THINK_CLOSE.search(raw):
            return ""  # stopped (or ran out) mid-thought: no answer to show
        answer = _strip_thinking(raw)
        calls = _tools.parse_tool_calls(answer) if schemas else []
        if not calls or (should_stop is not None and should_stop()) or _round == max_rounds:
            return _tools.strip_tool_calls(answer) if schemas else answer
        messages.append({"role": "assistant", "content": _tools.strip_tool_calls(answer),
                         "tool_calls": [{"type": "function", "function": c} for c in calls]})
        for c in calls:
            label = tools.label(c["name"], c["arguments"])
            if on_tool is not None:
                on_tool(label, None)
            result = tools.call(c["name"], c["arguments"])
            if on_tool is not None:
                on_tool(label, result)
            messages.append({"role": "tool", "name": c["name"], "content": result})
    return ""


# ---------------------------------------------------------------------------
# Speaker naming: replace "Speaker 2" with the person's real name
# ---------------------------------------------------------------------------
#
# Diarization only yields anonymous clusters. Current local approaches to name
# them (LLM speaker-attribution / "speaker name identification" work, e.g.
# DiarizationLM-style LLM post-processing) read the transcript for explicit
# evidence — self-introductions ("Hi, I'm Anna"), being addressed by name
# ("Thanks, Ben — what do you think?"), hand-offs and sign-offs. We ask the
# local model for a JSON mapping plus its evidence, then *verify* every answer
# against the transcript so a hallucinated name can never be applied.

_NAME_INSTRUCTION = (
    "Below is a meeting transcript with anonymous speaker labels. Identify the "
    "real first name (or full name) of each speaker ONLY where the transcript "
    "gives explicit evidence: the speaker introduces themselves, is addressed "
    "by name right before they speak, or is named when handed the floor. "
    "A name a speaker says to someone else (\"Thanks Ben\", \"Ben, can you …\") "
    "belongs to a DIFFERENT speaker — usually the one who answers next. "
    "Do not guess. Leave a speaker out if unsure.\n\n"
    "Answer with JSON only, e.g. "
    '{"Speaker 1": {"name": "Anna", "evidence": "Thanks Anna, go ahead."}}\n\n'
    "Speaker labels: {labels}"
)


def _parse_json_object(text: str) -> dict:
    import json

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


@on_mlx_thread
def infer_speaker_names(
    text: str,
    labels: List[str],
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, str]:
    """Map anonymous speaker labels in ``text`` to real names (only verified).

    ``text`` is the speaker-labelled transcript ("Speaker 1: …" lines). Returns
    {label: name} for the speakers the model could name *and* whose name
    actually occurs in the transcript; never two labels with the same name.
    """
    if not available() or not labels or not text.strip():
        return {}
    from mlx_lm import generate

    repo = resolve_model(model)
    spec = _spec(model)
    tmpl = dict((spec.template_kwargs if spec else None) or {})
    if "reasoning_effort" in tmpl:
        tmpl["reasoning_effort"] = "low"
    thinks = model_thinks(model)
    lm, tokenizer = _load_model(model)
    snippet = _fit_transcript(text.strip(), 40_000)
    instruction = _NAME_INSTRUCTION.replace("{labels}", ", ".join(labels))
    messages = [{"role": "user", "content": f"{instruction}\n\nTranscript:\n{snippet}"}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=thinks, **tmpl)
    except TypeError:
        chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(lm, tokenizer, prompt=chat_prompt,
                   max_tokens=400 + (4000 if thinks else 0), verbose=False)
    return _verified_names(_parse_json_object(_strip_thinking(raw)), labels, text)


_SELF_INTRO = (r"i'?m", r"i am", r"my name is", r"this is", r"ich bin", r"ich heiße",
               r"mein name ist", r"hier ist")


def _says_name_to_others(text: str, label: str, name: str) -> bool:
    """True if ``label`` utters ``name`` but never to introduce themselves."""
    own = " ".join(line.split(":", 1)[1] for line in text.splitlines()
                   if line.startswith(f"{label}:")).lower()
    n = re.escape(name.lower())
    if not re.search(rf"\b{n}\b", own):
        return False
    intro = "|".join(_SELF_INTRO)
    return not re.search(rf"\b(?:{intro})\s+{n}\b|\b{n}\s+(?:here|hier)\b", own)


def _verified_names(data: dict, labels: List[str], text: str) -> Dict[str, str]:
    """Keep only plausible names that literally occur in the transcript."""
    body = text.lower()
    out: Dict[str, str] = {}
    for label, val in data.items():
        if label not in labels:
            continue
        name = (val.get("name") if isinstance(val, dict) else val) or ""
        name = str(name).strip().strip(".,:;\"'")
        if not name or len(name) > 40 or not re.match(r"^[^\W\d_][\w .'-]*$", name):
            continue
        if name.lower() in {l.lower() for l in labels} or name.lower() not in body:
            continue
        if name in out.values():
            continue  # two clusters can't be the same person by name
        if _says_name_to_others(text, label, name):
            continue  # they address someone else by it ("Thanks Ben")
        out[label] = name
    return out
