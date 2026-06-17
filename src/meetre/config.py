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
from typing import Optional

CONFIG_DIR = Path(os.path.expanduser("~/.config/meetre"))
CONFIG_PATH = CONFIG_DIR / "config.json"

# Whisper model sizes, smallest/fastest first. large-v3-turbo is the best
# quality/speed balance on Apple Silicon and the default.
MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]


@dataclass
class Config:
    # Whisper model size used for transcription.
    model: str = "large-v3-turbo"
    # Language code (e.g. "en", "de") or None to auto-detect. German by default.
    language: Optional[str] = "de"
    # Whether to run speaker diarization ("person detection").
    person_detection: bool = False
    # Exact number of participants (None = estimate). Takes priority over the
    # min/max range below when set.
    num_speakers: Optional[int] = None
    # Estimation range for the number of participants, e.g. 3–6. Used only when
    # num_speakers is None. Either bound may be None (open-ended).
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Where transcripts are written.
    transcripts_dir: str = field(default_factory=lambda: str(Path.cwd() / "transcripts"))
    # Where the compressed MP3 backup of each recording is kept.
    audio_backup_dir: str = field(default_factory=lambda: str(Path.cwd() / "audio_backup"))
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
    # Local LLM summarization. Alias (qwen3-8b/qwen3-4b/gemma3-4b) or HF repo id.
    summary_model: str = "qwen3-8b"
    # Generate a summary section automatically after each transcription.
    auto_summarize: bool = True
    # Automatically save the summary + transcript to Apple Notes after recording.
    auto_notes: bool = True
    # Custom summarization prompt (instruction). Empty = use the built-in
    # professional default for the configured language.
    summary_prompt: str = ""
    # Check for updates (git pull) automatically when the menu-bar app launches.
    auto_update: bool = True

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
        return cls(**{k: v for k, v in data.items() if k in known})
