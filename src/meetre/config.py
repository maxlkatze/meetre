"""Persistent configuration for meetre.

Settings live in ``~/.config/meetre/config.json``. Transcripts default to a
``transcripts`` folder inside the current working directory so the tool is
useful out of the box, but the location is configurable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_DIR = Path(os.path.expanduser("~/.config/meetre"))
CONFIG_PATH = CONFIG_DIR / "config.json"
# Transcripts + MP3 backups. A fixed home location: the old default was relative
# to the working directory, which is "/" when launched by launchd or the .app,
# so every transcript write failed silently.
DEFAULT_DATA_DIR = Path.home() / "Documents" / "meetre"

# Transcription models. Whisper sizes (smallest/fastest first; large-v3-turbo is
# the best Whisper quality/speed balance on Apple Silicon), plus NVIDIA Parakeet
# TDT v3 on MLX (multilingual, very fast, supports live streaming; the default).
MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo",
          "parakeet-tdt-v3"]


@dataclass
class Config:
    # Transcription model (Whisper size or Parakeet).
    model: str = "parakeet-tdt-v3"
    # Language code (e.g. "en", "de") or None to auto-detect. German by default.
    language: Optional[str] = "de"
    # Whether to run speaker diarization ("person detection").
    person_detection: bool = False
    # Analyze the merged mix in a single diarization pass instead of running
    # pyannote separately on each source (mic / system). One analysis is faster;
    # sources that never exceed a low decibel level are skipped entirely.
    merged_analysis: bool = True
    # Exact number of participants (None = estimate). Takes priority over the
    # min/max range below when set.
    num_speakers: Optional[int] = None
    # Estimation range for the number of participants, e.g. 3–6. Used only when
    # num_speakers is None. Either bound may be None (open-ended).
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Where transcripts are written.
    transcripts_dir: str = field(default_factory=lambda: str(DEFAULT_DATA_DIR / "transcripts"))
    # Where the compressed MP3 backup of each recording is kept.
    audio_backup_dir: str = field(default_factory=lambda: str(DEFAULT_DATA_DIR / "audio_backup"))
    # Microphone (your voice). None = system default input.
    mic_device: Optional[int] = None
    # Loopback device capturing system audio (other participants). None = auto.
    system_device: Optional[int] = None
    # Record system audio (other participants) in addition to the mic.
    capture_system: bool = True
    # Capture system audio natively via ScreenCaptureKit (no BlackHole). When
    # False, fall back to the `system_device` loopback (BlackHole/Loopback).
    native_system: bool = True
    # HuggingFace token required by pyannote for diarization.
    hf_token: Optional[str] = None
    # Compute type for faster-whisper ("int8" is fast & light on CPU).
    compute_type: str = "int8"
    # Voice-activity detection: drop silence-only (hallucinated) Whisper
    # segments before saving. Needs the 'align' extra (Silero VAD).
    vad: bool = True
    # Phoneme forced alignment: attach exact per-word timestamps to each
    # segment. Needs the 'align' extra (torchaudio MMS_FA).
    word_timestamps: bool = True
    # Local LLM summarization. "auto" picks the best model that fits this
    # machine's RAM; or pin an alias (e.g. qwen3-32b/gemma3-27b/qwen3-8b) or a
    # full HF repo id. See meetre.summarizer.SUMMARY_MODELS.
    summary_model: str = "auto"
    # Generate a summary section automatically after each transcription.
    auto_summarize: bool = True
    # Automatically save the summary + transcript to Apple Notes after recording.
    auto_notes: bool = True
    # Custom summarization prompt (instruction). Empty = use the built-in
    # professional default for the configured language.
    summary_prompt: str = ""
    # Check for updates (git pull) automatically when the menu-bar app launches.
    auto_update: bool = True
    # Menu-bar app: when a Microsoft Teams call starts, pop up from the status
    # item and ask whether to record it.
    detect_calls: bool = True
    # Menu-bar app: stream the summary into a translucent, click-through
    # overlay in the top-right corner while it is being written.
    summary_overlay: bool = True
    # After speaker detection, let the local LLM read the transcript for
    # explicit name evidence ("Thanks, Anna …") and replace "Speaker 2" with
    # the real name. Names are verified against the transcript before use.
    name_speakers: bool = True
    # Your own name: replaces "Me"/"Ich" (the microphone side) in transcripts.
    my_name: str = ""
    # Chat tools (Tools & MCP window). The model decides when to call them.
    tool_web_search: bool = True
    tool_files: bool = True
    # Folders the chat model may always read ("Always allow"); others prompt.
    tool_folders: List[str] = field(default_factory=list)
    # MCP servers, Claude Desktop format: {name: {command, args, env, enabled}}.
    mcp_servers: Dict[str, dict] = field(default_factory=dict)
    # MCP tools allowed without asking ("server/tool").
    mcp_allowed_tools: List[str] = field(default_factory=list)

    @property
    def transcripts_path(self) -> Path:
        return Path(self.transcripts_dir).expanduser()

    @property
    def audio_backup_path(self) -> Path:
        return Path(self.audio_backup_dir).expanduser()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            cfg = cls()
            cfg.save()
            return cfg
        data = json.loads(CONFIG_PATH.read_text())
        # Only keep keys we know about so stale configs don't crash.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        if cfg._repair_dirs():
            cfg.save()
        return cfg

    def _repair_dirs(self) -> bool:
        """Point unwritable output dirs (e.g. "/transcripts") at ~/Documents/meetre."""
        changed = False
        for key, sub in (("transcripts_dir", "transcripts"), ("audio_backup_dir", "audio_backup")):
            if not _writable_dir(Path(getattr(self, key) or "").expanduser()):
                setattr(self, key, str(DEFAULT_DATA_DIR / sub))
                changed = True
        return changed


def _writable_dir(path: Path) -> bool:
    """True if ``path`` exists (or can be created) and accepts new files."""
    if not str(path) or not path.is_absolute():
        return False
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False
