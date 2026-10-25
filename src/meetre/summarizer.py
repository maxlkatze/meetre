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


# A summary model (all GGUF, run by llama-server — see gguf.py): the
# downloadable file in gguf.CATALOG under the same alias, its size in GB (for
# the per-machine "does this fit?" check), a one-line note, and whether it
# reasons before answering. ``template_kwargs`` are chat-template options.
ModelSpec = namedtuple("ModelSpec", "size_gb note thinks template_kwargs")
ModelSpec.__new__.__defaults__ = (False, None)

SUMMARY_MODELS = {
    # K2 Horizon (IFM, Sept 2026; always reasons). "low" = no reasoning: K2
    # closes the think block at once (see CLAUDE.md for the measurements).
    "k2-horizon-7b": ModelSpec(9.6, "excellent summary writing", True,
                               {"reasoning_effort": "low"}),
    # Gemma 4 QAT: best German in LocalCompanion's tests (36/38).
    "gemma4-12b":    ModelSpec(6.7, "strong German, 140+ languages"),
    "qwen3.5-9b":    ModelSpec(6.7, "balanced", False, {"enable_thinking": False}),
    "gemma4-e4b":    ModelSpec(4.2, "small, multilingual"),
    "qwen3.5-4b":    ModelSpec(2.9, "smallest, fastest", False, {"enable_thinking": False}),
}

# Retired aliases (the MLX lineup) → the closest model now, so an existing
# config.summary_model keeps working.
_RETIRED = {
    "qwen3.5-397b": "k2-horizon-7b", "qwen3.5-122b": "k2-horizon-7b",
    "qwen3.5-35b": "gemma4-12b", "qwen3.5-27b": "gemma4-12b", "gemma4-26b": "gemma4-12b",
    "mistral-24b": "gemma4-12b", "qwen3.5-2b": "qwen3.5-4b", "qwen3-32b": "gemma4-12b",
    "gemma3-27b": "gemma4-12b", "qwen3-30b-a3b": "gemma4-12b", "qwen3-14b": "qwen3.5-9b",
    "qwen3-8b": "qwen3.5-9b", "gemma3-4b": "gemma4-e4b", "qwen3-4b": "qwen3.5-4b",
    "qwen3-235b": "k2-horizon-7b",
}

# Preferred → fallback: menu order and what "auto" picks on a given machine.
# Gemma 4 12B QAT leads (measured 2026-10-25 on 4 real German meetings): as
# fast as K2 Q8 (37 tok/s), 8.9 GB instead of 11.4 GB, the only model that
# kept unsettled points out of "Entscheidungen", and the best name reader.
_BEST_FIRST = ["gemma4-12b", "k2-horizon-7b", "qwen3.5-9b", "gemma4-e4b", "qwen3.5-4b"]

# The MLX repos meetre downloaded before the switch to GGUF: offered for
# deletion in Settings (other apps' models share the HF cache — never touched).
LEGACY_MLX_REPOS = {
    "mlx-community/K2-Horizon-7B-4bit", "mlx-community/Qwen3.5-397B-A17B-4bit",
    "mlx-community/Qwen3.5-122B-A10B-MLX-4bit", "mlx-community/Qwen3.5-35B-A3B-4bit",
    "mlx-community/Qwen3.5-27B-4bit", "mlx-community/Qwen3.5-9B-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit", "mlx-community/Qwen3.5-2B-MLX-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit", "mlx-community/gemma-4-12B-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit", "mlx-community/Qwen3-32B-4bit",
    "mlx-community/gemma-3-27b-it-4bit", "mlx-community/Qwen3-30B-A3B-4bit",
    "mlx-community/Qwen3-14B-4bit", "mlx-community/Qwen3-8B-4bit",
    "mlx-community/gemma-3-4b-it-4bit", "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-235B-A22B-4bit",
}

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
    return "qwen3.5-4b"


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


def _blob_files(path: Path) -> set:
    """The real files holding a model's data. HF snapshots link into the
    model's ``blobs/``, and some caches (deduplicated stores) link those on
    into a shared ``hub/blobs/`` — so links are followed, each file once."""
    files = set()
    for p in path.rglob("*"):
        try:
            real = p.resolve()
            if _stat.S_ISREG(real.stat().st_mode):
                files.add(real)
        except OSError:
            pass
    return files


def _dir_size(path: Path) -> int:
    """On-disk bytes of a model, following links, each real file once."""
    total = 0
    for f in _blob_files(path):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _alias(name: Optional[str]) -> Optional[str]:
    """Catalog alias for an alias / ``"auto"`` / retired alias (None for a
    ``gguf:`` path or anything unknown)."""
    if not name or name == "auto":
        return default_model()
    name = _RETIRED.get(name, name)
    return name if name in SUMMARY_MODELS else None


def installed_size(name: str) -> int:
    """Bytes a catalog model occupies on disk, or 0 if not there."""
    from . import gguf

    a = _alias(name)
    p = gguf.path_for(a) if a else (Path(name[len(gguf.PREFIX):]) if gguf.is_gguf(name) else None)
    return p.stat().st_size if p is not None and p.is_file() else 0


def is_installed(name: str) -> bool:
    return installed_size(name) > 0


def installed_models() -> List[Dict[str, object]]:
    """Everything meetre downloaded that can be deleted: GGUF files it fetched
    into the models folder, and MLX models left from before the switch.
    ``{id, alias, size, kind}``, largest first. The user's own GGUFs and
    other apps' models in the HF cache are never listed."""
    from . import gguf

    out = [{"id": str(d["path"]), "alias": d["alias"], "size": d["size"], "kind": "gguf"}
           for d in gguf.downloaded()]
    cache = hf_cache_dir()
    for repo in sorted(LEGACY_MLX_REPOS):
        d = cache / _repo_to_dirname(repo)
        if d.is_dir():
            out.append({"id": repo, "alias": repo.split("/")[-1], "size": _dir_size(d),
                        "kind": "legacy"})
    out.sort(key=lambda m: m["size"], reverse=True)
    return out


def uninstall_model(id: str) -> int:
    """Delete a model meetre downloaded: a GGUF file it fetched, or a legacy
    MLX repo (or any HF repo id — speech models). Returns freed bytes."""
    from . import gguf

    if id.startswith("/"):
        return gguf.delete(id)
    p = hf_cache_dir() / _repo_to_dirname(id)
    if not p.is_dir():
        return 0
    files = _blob_files(p)
    outside = {f for f in files if p.resolve() not in f.parents}
    freed = sum(f.stat().st_size for f in files - outside if f.exists())
    shutil.rmtree(p, ignore_errors=True)
    if outside:  # shared store: delete only what no other model still links to
        still = set()
        for d in hf_cache_dir().glob("models--*"):
            still |= _blob_files(d) & outside
        for f in outside - still:
            try:
                freed += f.stat().st_size
                f.unlink()
            except OSError:
                pass
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
    "de": "Du bist ein erfahrener Protokollant. Du fasst Meetings sachlich, "
          "konkret und knapp auf Deutsch zusammen. Du verwendest nur, was im "
          "Transkript steht.",
    "en": "You are an experienced meeting note-taker. You summarize meetings "
          "factually, concretely and concisely. You use only what is in the "
          "transcript.",
}

# The default, user-editable instruction. Tuned on K2 Horizon 7B (reasoning
# "low") against real meetings: it comes AFTER the transcript (see
# summarize()) — the old instruction-first layout plus a trailing reminder let
# K2 fall into repetition loops on 30-min meetings (141 s, 18k chars of one
# repeated bullet vs. 12 s now). Each section has one job (the summary gives
# context, not the decisions again), "- Keine" only for empty sections, and
# nonsense from speech recognition is dropped rather than copied.
_DEFAULT_INSTRUCTION = {
    "de": "Fasse das Meeting oben für jemanden zusammen, der nicht dabei war. "
          "Danach soll er wissen, WAS konkret gesagt, entschieden und vereinbart "
          "wurde — nicht nur, DASS über etwas gesprochen wurde.\n\n"
          "REGELN\n"
          "- Verdichten statt nacherzählen: ein kurzes Gespräch ergibt nur wenige "
          "Stichpunkte, ein langes Meeting entsprechend mehr.\n"
          "- Konkret: Probleme, Argumente, Zahlen, Werte, Namen von "
          "Projekten/Tools, Termine und Fristen so, wie sie fallen. Keine leeren "
          "Meta-Sätze („Das Team diskutierte …“, „Austausch zu …“).\n"
          "  Schlecht: „Diskussion über Designs und Konsistenz.“\n"
          "  Gut: „Zu viele parallele Designvarianten ohne verbindliche "
          "Figma-Vorlagen — Inhalte lassen sich nicht konsistent einbauen.“\n"
          "- Jede Information genau einmal: Was unter Entscheidungen, Aufgaben "
          "oder Offene Fragen steht, steht nicht noch einmal woanders.\n"
          "- Keine Widersprüche: Ein Punkt ist entweder entschieden oder offen, "
          "nie beides.\n"
          "- Entscheidung heißt: Die Runde hat sich ausdrücklich festgelegt "
          "(„machen wir so“, „nehmen wir“, „lassen wir“). Nur vorgeschlagen, "
          "erwogen oder bevorzugt → Offene Frage. Fakten, Preise und Zahlen "
          "sind keine Entscheidungen — sie gehören als Begründung zur "
          "Entscheidung oder in die Zusammenfassung.\n"
          "- Das Transkript stammt aus Spracherkennung und enthält Hörfehler. "
          "Korrigiere offensichtliche Fehler still aus dem Kontext (z. B. "
          "„Jason“ → „JSON“). Ergibt ein Wort oder Satz auch so keinen Sinn, lass "
          "ihn weg — übernimm nie Unsinn.\n"
          "- Aufgaben sind konkrete Arbeitsschritte, die jemand zusagt oder um "
          "die jemand gebeten wird — auch beiläufige („ich schick dir …“, „kannst "
          "du …“). Keine Aufgaben aus Floskeln, Scherzen oder bloßen Ideen.\n"
          "- Zuständig ist, wer zusagt bzw. wer gebeten wird, mit dem "
          "Sprecher-Label aus dem Transkript („Ich“, „Sprecher 1“ …). Ein im "
          "Gespräch genannter Name („die Seite von Julia“) ist meist das Thema, "
          "nicht der Sprecher. Ist unklar, wer zuständig ist, schreib die Aufgabe "
          "ohne Person. Nie raten, nichts erfinden.\n\n"
          "FORMAT — antworte nur mit diesen vier Abschnitten, ohne Vorrede, auf "
          "Deutsch:\n"
          "## Zusammenfassung\n"
          "2–3 Sätze: Worum ging es, was war das Problem bzw. die Ausgangslage, "
          "wo steht das Thema jetzt. Beschlüsse und Aufgaben gehören nicht "
          "hierher, die stehen in ihren Abschnitten.\n\n"
          "## Entscheidungen\n"
          "- Ausdrücklich Beschlossenes — Begründung, falls genannt\n\n"
          "## Aufgaben / To-dos\n"
          "- **Sprecher-Label:** Aufgabe (bis Frist, falls genannt)\n\n"
          "## Offene Fragen\n"
          "- Konkret ungeklärter Punkt\n\n"
          "Hat ein Abschnitt keinen Inhalt, schreibe darunter nur die Zeile "
          "„- Keine“.",
    "en": "Summarize the meeting above for someone who was not there. Afterwards "
          "they should know WHAT was concretely said, decided and agreed — not "
          "just THAT something was discussed.\n\n"
          "RULES\n"
          "- Condense, don't retell: a short conversation yields only a few "
          "bullets, a long meeting correspondingly more.\n"
          "- Be concrete: problems, arguments, numbers, values, names of "
          "projects/tools, dates and deadlines as they come up. No empty "
          "meta-phrases (\"The team discussed …\", \"exchange on …\").\n"
          "  Bad: \"Discussion about designs and consistency.\"\n"
          "  Good: \"Too many parallel design variants and no binding Figma "
          "templates — content can't be embedded consistently.\"\n"
          "- Every piece of information exactly once: what is under Decisions, "
          "Action items or Open questions does not appear anywhere else.\n"
          "- No contradictions: a point is either decided or open, never both.\n"
          "- A decision means the group explicitly settled it (\"let's do that\", "
          "\"we'll take …\", \"we drop …\"). Only proposed, considered or "
          "preferred → Open question. Facts, prices and numbers are not "
          "decisions — they go with a decision as its reason, or into the "
          "summary.\n"
          "- The transcript comes from speech recognition and contains "
          "mishearings. Silently fix obvious errors from context (e.g. "
          "\"Jason\" → \"JSON\"). If a word or sentence still makes no sense, "
          "leave it out — never copy nonsense.\n"
          "- Action items are concrete work steps someone commits to or is asked "
          "to do — including casual ones (\"I'll send you …\", \"can you …\"). No "
          "action items from pleasantries, jokes or mere ideas.\n"
          "- The owner is whoever commits or is asked, using the speaker label "
          "from the transcript (\"Me\", \"Speaker 1\" …). A name dropped in "
          "conversation (\"Julia's page\") is usually the topic, not the speaker. "
          "If the owner is unclear, write the task without a person. Never "
          "guess, never invent.\n\n"
          "FORMAT — respond with only these four sections, no preamble:\n"
          "## Summary\n"
          "2–3 sentences: what it was about, what the problem or starting point "
          "was, where the topic stands now. Decisions and tasks don't go here — "
          "they have their own sections.\n\n"
          "## Decisions\n"
          "- What was explicitly decided — reason, if given\n\n"
          "## Action items\n"
          "- **Speaker label:** task (by deadline, if mentioned)\n\n"
          "## Open questions\n"
          "- Concrete unresolved point\n\n"
          "If a section has no content, write only the line \"- None\" under it.",
}

_TRANSCRIPT_LABEL = {"de": "Transkript", "en": "Transcript"}

_STAMP = re.compile(r"^`\[\d+:\d+(?::\d+)?\]`\s*")
_SPEAKER_HEAD = re.compile(r"^\*\*(.+?)\*\*\s*$")
_SPEAKER_PREFIX = re.compile(r"^([^:.!?\n]{1,40}):\s+(.*)$")


def compact_transcript(text: str) -> str:
    """One ``Speaker: text`` line per turn, without timestamps or markdown.

    Accepts the saved transcript body (``**Speaker**`` headers + timestamped
    lines) or ``Speaker: text`` lines per segment. ~25–30% fewer tokens: faster
    prefill, more meeting per chunk, and less noise for the model.
    """
    turns: List[list] = []
    who = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = _SPEAKER_HEAD.match(line)
        if head:
            who = head.group(1)
            continue
        line = _STAMP.sub("", line)
        m = _SPEAKER_PREFIX.match(line)
        spk, said = (m.group(1).strip(), m.group(2)) if m else (who, line)
        if turns and turns[-1][0] == spk:
            turns[-1][1].append(said)
        else:
            turns.append([spk, [said]])
    return "\n".join((f"{spk}: " if spk else "") + " ".join(said) for spk, said in turns)


def _dedupe_lines(text: str) -> str:
    """Drop exact repeats of substantive lines (greedy-decoding loops)."""
    seen, out = set(), []
    for line in text.splitlines():
        key = line.strip()
        if len(key) > 12 and not key.startswith("#"):
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)

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
    """Summaries run on llama.cpp, which meetre fetches itself when missing."""
    return True


def model_thinks(name: Optional[str]) -> bool:
    """Whether a model (alias, 'auto' or a GGUF file) reasons before answering."""
    spec = _spec(name)
    if spec is not None:
        return bool(spec.thinks)
    return "think" in (name or "").lower()


def _spec(name: Optional[str]) -> Optional[ModelSpec]:
    """The catalog entry for an alias / ``"auto"`` / retired alias. A GGUF
    file of a catalog family (``K2-Horizon-7B-Q6_K.gguf``) gets that family's
    settings (reasoning, template options); other files None."""
    a = _alias(name)
    if a is not None:
        return SUMMARY_MODELS[a]
    from . import gguf

    if gguf.is_gguf(name):
        file = Path(name).name.lower().replace("_", "-")
        for alias, pattern in (("k2-horizon-7b", "k2-horizon"), ("gemma4-12b", "gemma-4"),
                               ("qwen3.5-9b", "qwen3.5")):
            if pattern in file:
                return SUMMARY_MODELS[alias]._replace(size_gb=0.0)
    return None


def _load_model(model: Optional[str]):
    """``(server, tokenizer)``: the llama-server running this model."""
    from . import gguf

    srv = gguf.load(resolve_model(model))
    return srv, gguf.Tokenizer(srv)


def _chat(lm, messages: List[dict], max_tokens: int, tmpl: Optional[dict] = None,
          penalty: bool = False, json_schema: Optional[dict] = None):
    """Text pieces of a reply, reasoning wrapped in ``<think>…</think>`` so
    the callers' think handling (_THINK_CLOSE, _strip_thinking) is unchanged.

    ``penalty``: mild repetition penalty over the recent output (see
    :func:`summarize`)."""
    from . import gguf

    state = "start"  # → "think" → "text"
    for kind, piece in gguf.chat(lm, messages, max_tokens, template_kwargs=tmpl,
                                 json_schema=json_schema,
                                 repetition_penalty=1.05 if penalty else None):
        if kind == "think" and state == "start":
            state, piece = "think", "<think>" + piece
        elif kind == "text" and state != "text":
            # The answer has begun: always close the (maybe empty) reasoning
            # block, or callers take a direct answer for a cut-off thought
            # (K2 on "low" reasons not at all — every summary came back empty).
            piece = ("</think>" if state == "think" else "<think></think>") + piece
            state = "text"
        yield piece


def resolve_model(name: Optional[str]) -> str:
    """``gguf:<path>`` for an alias, ``"auto"`` (the best model that fits this
    machine), a retired alias, or a ``gguf:`` path (passed through)."""
    from . import gguf

    if gguf.is_gguf(name):
        return name
    return gguf.PREFIX + str(gguf.path_for(_alias(name) or default_model()))


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


def _looping(raw: str, repeats: int = 3) -> bool:
    """True once the newest complete line already occurred ``repeats`` times."""
    lines = raw.rstrip("\n").split("\n")
    last = lines[-1].strip() if lines else ""
    # Headers legitimately recur: reasoning models draft the answer (with its
    # "## …" sections) inside the think block before writing it for real.
    if len(last) <= 12 or last.startswith("#"):
        return False
    return sum(1 for l in lines if l.strip() == last) >= repeats


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
    if not text.strip():
        return ""

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
    text = compact_transcript(text) or text

    def _user(chunk: str) -> str:
        # Default prompt: transcript first, instructions last (freshest when
        # the model starts reasoning). Custom prompts keep the classic
        # instruction-then-transcript order their wording may rely on.
        if prompt:
            return f"{instruction}\n\n{label}:\n{chunk}"
        return f"<{label.lower()}>\n{chunk}\n</{label.lower()}>\n\n{instruction}"

    # Reasoning models spend tokens on a hidden <think> pass before the answer,
    # so they need far more headroom. Non-thinking models answer directly.
    answer_budget = 900
    reduce_budget = 500
    think_budget = 4000  # extra tokens reserved for the reasoning pass

    # Mild repetition penalty over the recent output. Greedy decoding on long
    # meetings otherwise drifts into restating every decision as a task, or
    # loops one bullet until the token cap. Measured on K2 Horizon 7B with real
    # 30-min meetings: all four sections every time, ~40% faster; temperature
    # sampling instead dropped whole sections.
    def _generate(user: str, max_tokens: int, use_thinking: bool, **extra) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        raw = ""
        for piece in _chat(lm, messages, max_tokens,
                           {"enable_thinking": use_thinking, **tmpl, **extra}, penalty=True):
            raw += piece
            if "\n" in piece and _looping(raw):
                break  # stuck repeating one line; the rest is dedupe'd away
            if on_progress is not None and piece:
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
        return _dedupe_lines(_run(_user(parts[0]), answer_budget))

    # Map-reduce for long meetings: summarize each chunk, then combine.
    partials = []
    for i, p in enumerate(parts, 1):
        stage["label"] = f"part {i}/{len(parts)}"
        partials.append(_dedupe_lines(_run(_user(p), reduce_budget)))
    stage["label"] = "merge"
    reduce_instr = _REDUCE.get(lang, _REDUCE["en"])
    return _dedupe_lines(_run(f"{reduce_instr}\n\n{chr(10).join(partials)}", answer_budget))


def generate_title(
    text: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Generate a short meeting title from ``text`` (a summary or transcript).

    Uses the same local MLX model as :func:`summarize`. Returns a single
    cleaned-up line; ``""`` if unavailable or the input is empty.
    """
    if not text.strip():
        return ""

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
    # "low" reasoning closes the think block at once: no reasoning budget.
    max_tokens = 40 + (4000 if thinks and tmpl.get("reasoning_effort") != "low" else 0)
    raw = "".join(_chat(lm, messages, max_tokens, {"enable_thinking": thinks, **tmpl}))
    return _clean_title(_strip_thinking(raw))


def _clean_title(text: str) -> str:
    """Reduce model output to a single, filename-friendly title line."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    # Drop common lead-ins ("**Titel:** …"), surrounding quotes/markdown and a
    # trailing period — and any leftover leading ":"/dash, which otherwise
    # ends up in file names and Note titles (": Tanzen von …").
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r'^\s*(title|titel)\s*[:\-–—]\s*', "", line, flags=re.IGNORECASE)
    line = line.strip().strip('"\'`„“”').lstrip(":-–— ")
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
    if _chat_model["key"] != repo:
        lm, tok = _load_model(model)
        _chat_model.update(key=repo, lm=lm, tok=tok)
    return _chat_model["lm"], _chat_model["tok"]


def release_chat_model() -> None:
    """Free the chat model: the llama-server stops (it also stops by itself
    after gguf.IDLE_SECONDS without use)."""
    from . import gguf

    _chat_model.update(key=None, lm=None, tok=None)
    gguf.release()


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
_CHAT_TRANSCRIPT_CHARS = 40_000  # fits gguf.CONTEXT (16k tokens) with the answer


def _fit_transcript(text: str, limit: int = _CHAT_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n[… middle of the meeting omitted for length …]\n\n{text[-half:]}"


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
        from . import gguf

        for piece in gguf.complete(lm, render(), budget):
            raw += piece
            if on_progress is not None and piece:
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
    "Below is a meeting transcript; each line starts with an anonymous speaker "
    "label ({labels}). List the places where a person's first name (or full "
    "name) is SPOKEN in the transcript. For each one give: the exact words "
    "(quote), the label of the speaker who says it, the name, and how it is "
    "used:\n"
    "- \"addresses_listener\": said to the person being spoken to "
    "(\"Thanks, Anna\", \"Ben, can you …\", \"Hey Lara\")\n"
    "- \"introduces_self\": the speaker's own name (\"I'm Anna\", \"Ich bin Ben\", "
    "\"Anna here\")\n"
    "- \"talks_about_someone\": anyone else (\"Julia's page\", a story, a quote)\n"
    "Only names that are really spoken; at most 12, the clearest first. "
    "Answer with JSON only."
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
    repo = resolve_model(model)
    spec = _spec(model)
    tmpl = dict((spec.template_kwargs if spec else None) or {})
    if "reasoning_effort" in tmpl:
        tmpl["reasoning_effort"] = "low"
    thinks = model_thinks(model)
    lm, tokenizer = _load_model(model)
    snippet = _fit_transcript(text.strip(), 40_000)
    # The example uses the real labels: with "Speaker 1" in it, K2 answered
    # for "Speaker 1 … 134" (labels here are "Ich", "Sprecher") — all dropped.
    instruction = (_NAME_INSTRUCTION.replace("{labels}", ", ".join(labels))
                   .replace("{first}", labels[-1]))
    messages = [{"role": "user", "content": f"{instruction}\n\nTranscript:\n{snippet}"}]
    # The model only reads lines (who says which name, how); the rules of who
    # that makes whom are applied in code (_names_from_mentions). Asked to
    # map labels to names itself, every model gave the name to the speaker
    # who SAYS it ("…, Jenny" → "Ich = Jenny"); free-form JSON also came in
    # the wrong shape or looped to the token cap. The schema (llama-server
    # grammar) fixes the shape; the quote comes first so the rest follows it.
    schema = {"type": "object", "additionalProperties": False, "required": ["mentions"],
              "properties": {"mentions": {"type": "array", "maxItems": 12, "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["quote", "said_by", "name", "use"],
                  "properties": {
                      "quote": {"type": "string", "maxLength": 160},
                      "said_by": {"type": "string", "enum": list(labels)},
                      "name": {"type": "string", "maxLength": 40},
                      "use": {"type": "string", "enum": ["addresses_listener", "introduces_self",
                                                         "talks_about_someone"]}}}}}}
    kw = {"enable_thinking": thinks, **tmpl}
    budget = 900 + (4000 if thinks and tmpl.get("reasoning_effort") != "low" else 0)
    raw = "".join(_chat(lm, messages, budget, kw, json_schema=schema))
    data = _first_json_object(_strip_thinking(raw)) or {}
    return _names_from_mentions(data.get("mentions") or [], labels, text)


def _names_from_mentions(mentions: list, labels: List[str], text: str) -> Dict[str, str]:
    """Who is called what, from the name mentions the model read off the lines.

    Only mentions whose name really occurs in a line of that speaker count.
    A name said to the listener belongs to the other speaker (with two
    labels), else to whoever speaks next after that line; a self-introduction
    to the speaker; names merely talked about to nobody. Each label takes its
    most frequent name; a name goes to one label only."""
    lines = [(l.split(":", 1)[0], l.split(":", 1)[1].lower())
             for l in text.splitlines() if ":" in l and l.split(":", 1)[0] in labels]
    votes: Dict[Tuple[str, str], int] = {}
    for m in mentions:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip().strip(".,:;\"'")
        who, use = m.get("said_by"), m.get("use")
        if (who not in labels or not name or len(name) > 40
                or not re.match(r"^[^\W\d_][\w .'-]*$", name)
                or name.lower() in {l.lower() for l in labels}):
            continue
        pat = re.compile(rf"\b{re.escape(name.lower())}\b")
        at = next((i for i, (spk, t) in enumerate(lines) if spk == who and pat.search(t)), None)
        if at is None:
            continue  # not actually said by that speaker
        if use == "introduces_self":
            target = who
        elif use == "addresses_listener":
            others = [l for l in labels if l != who]
            target = others[0] if len(others) == 1 else next(
                (spk for spk, _ in lines[at + 1:] if spk != who), None)
        else:
            continue
        if target is not None:
            votes[(target, name)] = votes.get((target, name), 0) + 1
    out: Dict[str, str] = {}
    for (label, name), _n in sorted(votes.items(), key=lambda kv: -kv[1]):
        if label not in out and name not in out.values():
            out[label] = name
    return out


def _first_json_object(text: str) -> Optional[dict]:
    """The first complete, parseable ``{…}`` in ``text`` (None if none yet)."""
    import json

    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        return data if isinstance(data, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


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
