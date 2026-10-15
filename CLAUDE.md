# CLAUDE.md

Guidance for working in this repository.

## Commits

- Do **not** add a `Co-Authored-By` trailer or any Claude/AI attribution to commits.

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

## Native app (`mac/`) — in progress

SwiftUI app (macOS 26) per `docs/DESIGN.md` (design & rule book — follow it)
and `docs/ROADMAP.md` (architecture, engine protocol, storage v2 / iCloud
sync, companion app plan). `mac/build.sh` → `mac/build/Meetre.app`. It spawns
`python -m meetre engine` (JSON-RPC over stdio) and never imports Python.
`MEETRE_FAKE_ENGINE=1` uses `mac/Scripts/fake_engine.py` fixtures; debug flags
(`--open-library`, `--preview-popover`, `--demo-record` …) in `MeetreApp.swift`.

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
- `callwatch.py` — detects a Teams call (Teams process capturing the mic, via
  CoreAudio process list) → menubar popover asking to record (`detect_calls`).
- `sysaudio.py` — compiles bundled Swift ScreenCaptureKit helper for system audio.
- `downloads.py` — HF model download with progress. `icon.py` — status-bar icon.
- `autostart.py` — launchd "start at login". `bundle.py` — `.app` wrapper.
- `updater.py` — self-update via `git pull`. `crashlog.py` — crash logging.
- `overlay.py` — click-through top-right HUDs: `SummaryOverlay` (fed by
  `summarize(on_progress=…)`, toggled by `summary_overlay`) and
  `TranscriptOverlay` (live captions while recording, `live_transcript`).
- `live.py` — `LiveTranscriber`: tails the recorder's WAVs (`recorder.WavTail`)
  and re-decodes the uncommitted window via `transcriber.transcribe_live`
  (MLX thread); mic-vs-system energy labels speakers. Preview only.
- `engine.py` — headless JSON-RPC server for the native app (`meetre engine`);
  stdout is protocol-only (all other output goes to stderr).
  `scripts/engine_smoke.py` exercises it.
- `pipeline.py` — UI-free post-recording pipeline (`Pipeline.process`, `Hooks`)
  shared by `menubar.py` and `engine.py`.
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
- **Diarization runs on MPS.** pyannote defaults to CPU; `_load_pipeline` moves
  it to Metal (~12x faster) and caches it per process, retrying on CPU if an op
  fails on MPS. The menubar `_finish` publishes progressively via `_publish`:
  transcript (.md + Apple Note) right after ASR, rewritten in place after
  speakers, then after summary/title (`update_apple_note`).
- **K2 Horizon 7B is the preferred summary model** (`_BEST_FIRST[0]`). mlx-lm has
  no native `k2_horizon` arch; the repo ships `k2_horizon.py` that mlx-lm executes
  on load, so the spec pins `revision` to a reviewed commit (honored by
  `downloads.ensure_model` and `load(..., revision=)`). Re-review that file before
  bumping the pin. K2 always reasons (`<ifm|think…>` tags); use
  `reasoning_effort="low"`. Note "low" = *no* reasoning (K2 closes the think
  block at once). Re-measured 2026-10-15 on 1–31 min meetings: medium is
  2.5–13× slower (165 s for 31 min) with worse owners/phrasing; high took
  60–214 s, truncated one answer and looped forever on a misheard word in
  another. Keep "low" unless a new model/prompt is re-benchmarked.
- **Summary prompt is tuned for K2** (benchmarked on real 1–31 min meetings).
  Default prompt goes *after* the transcript (compacted to one `Speaker: text`
  line per turn by `compact_transcript`); custom prompts keep instruction-first.
  Generation uses a mild repetition penalty (1.05 / 256 tokens) plus a
  line-loop guard (`_looping`, `_dedupe_lines`): greedy K2 otherwise looped one
  bullet to the token cap on long meetings (141 s → ~14 s). Temperature
  sampling dropped whole sections — don't switch to it without re-measuring.
