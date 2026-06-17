# CLAUDE.md

Guidance for working in this repository.

## What meetre is

macOS-only meeting recorder + transcriber. Captures microphone and system audio
(Zoom/Meet/Teams), transcribes locally with Whisper or Parakeet (Metal/MLX),
optionally labels speakers (pyannote diarization), and summarizes with a local
LLM (Qwen3.5 / Gemma 4 / Mistral via MLX-LM). Everything runs on-device — no
audio or text leaves the machine.

- Python 3.11+. Source under `src/meetre/`. Package entry point: `meetre`.
- Config: `~/.config/meetre/config.json` (`Config` dataclass in `config.py`).
- Transcripts: Markdown in the configured transcripts dir; MP3 audio backups too.
- No project test suite; `.venv/` holds dependencies. Use `.venv/bin/python`.

## Running

- `meetre` / `meetre menubar` — macOS status-bar app (default; rumps + AppKit).
- `meetre cli` — interactive text menu. `meetre record|transcribe|summarize|list|devices`.
- `meetre model SIZE`, `meetre summary-model CHOICE`, `meetre persons on|off`,
  `meetre speakers SPEC`, `meetre config [KEY VALUE]`.

## Module map (`src/meetre/`)

- `menubar.py` — macOS status-bar app (rumps + AppKit). Click-to-record icon,
  settings popup (model/language/summary/speakers pickers), live progress.
  `MeetreApp`, `_build_menu`, `_begin_recording`, `_finish`, `_generate_summary`,
  `_generate_title`.
- `cli.py` — CLI subcommands (`main`, `do_record`, `do_transcribe`, `interactive`).
- `config.py` — JSON config dataclass `Config` (`load`/`save`).
- `recorder.py` — `Recorder`: mic (sounddevice) + system audio mixed to 16 kHz WAV;
  `save_mp3`, `list_devices`.
- `transcriber.py` — Whisper (MLX / faster-whisper) + pyannote diarization.
  `transcribe`, `transcribe_attributed`, `diarize`, `available_backend`, `Segment`.
- `summarizer.py` — local MLX-LM summaries + title generation. `summarize`,
  `generate_title`, model catalog/fit logic (`SUMMARY_MODELS`, `system_memory_gb`,
  `model_fits`, `model_catalog`, `default_model`), model install/uninstall.
- `transcript.py` — render speaker-labelled Markdown transcript (`write_transcript`).
- `integrations.py` — Apple Notes (`add_to_apple_notes`) + Claude Desktop hand-off.
- `align.py` — Silero VAD + phoneme forced alignment for Whisper segments.
- `sysaudio.py` — compiles bundled Swift ScreenCaptureKit helper for system audio.
- `downloads.py` — HF model download with progress. `icon.py` — status-bar icon.
- `autostart.py` — launchd "start at login". `bundle.py` — `.app` wrapper.
- `updater.py` — self-update via `git pull`. `crashlog.py` — crash logging.
- `ui.py` — Rich CLI tables/panels. `__init__.py` — silences dep log noise.

## Notes / gotchas

- **Memory detection must be PATH-independent.** When launched by launchd
  ("start at login") or after "Restart meetre", `PATH` is empty, so a bare
  `subprocess.run(["sysctl", ...])` raises `FileNotFoundError`. `system_memory_gb()`
  uses in-process `os.sysconf` first (absolute-path `sysctl` only as fallback);
  if RAM reads as 0 the model-fit check would treat *every* model as fitting and
  fail to gray out oversized ones in the menu. Don't reintroduce a PATH-relative
  subprocess for hardware detection.
- Summary/title generation needs `mlx-lm`; gated on `summarizer.available()`.
  "Record with previous" skips the settings dialog and derives the meeting title
  from the summary (`_auto_title` → `_generate_title`) to name the transcript/Note.
