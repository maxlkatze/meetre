# meetre

Record a meeting on your Mac and turn it into a clean, speaker-labelled,
summarised transcript. Everything runs on-device.

![macOS](https://img.shields.io/badge/macOS-13%2B-000000?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1–M5-0071e3)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

meetre is **macOS and Apple Silicon only**. It is built on Apple's MLX runtime
and ScreenCaptureKit; it does not run on Windows or Linux, and Intel Macs are not
supported for the full pipeline.

## Quick install

```bash
git clone https://github.com/maxlkatze/meetre.git && cd meetre && bash install.sh
```

The installer needs no admin password. It uses your existing `python3`/`git` if
present, otherwise downloads a local, relocatable Python into `.runtime/`. It
creates a virtual environment, installs meetre and the menu-bar app, links
`meetre` into `~/.local/bin`, asks for an optional HuggingFace token, registers a
login startup item, and launches the menu bar.

After installing, look for the `✦` icon in the top-right menu bar and click
**Record…**.

> The first recording asks for the **Screen Recording** permission (system audio).
> Saving to Notes asks for **Automation** permission. Both are one-time.

## What it does

meetre captures your microphone and the system audio (the other participants in
Zoom, Meet, Teams, etc.), then:

1. Transcribes locally with Whisper `large-v3-turbo` via MLX (Metal-accelerated).
2. Optionally labels speakers using `pyannote` diarization.
3. Summarises with a local LLM (`Qwen3-8B`) into Zusammenfassung / Entscheidungen
   / Aufgaben / Offene Fragen.
4. Writes a Markdown transcript and an MP3 backup, and saves the summary to Apple
   Notes.

No audio or text leaves the machine. No accounts, no API keys.

## Features

- Microphone and system audio, mixed into one track. System audio is captured
  natively via ScreenCaptureKit, so no BlackHole or loopback driver is needed.
- MLX transcription, the fastest Whisper backend on Apple Silicon.
- German by default, with multilingual and auto-detect support.
- Speaker detection with an optional headcount hint (`auto`, `4`, or `3-6`).
- Local LLM summaries with a fully editable prompt.
- Menu-bar app with live status, model-download progress, and a settings popup.
- Auto-update (`git pull`) on every launch, and start-at-login.
- MP3 backups, and the ability to re-transcribe any audio file later.

## System requirements

| | Minimum | Recommended |
|---|---|---|
| Mac | Apple Silicon (M1) | M2 Pro / M3 / M4 / M5 |
| macOS | 13 Ventura (for ScreenCaptureKit) | 14 Sonoma or newer |
| RAM | 8 GB (use a smaller summary model) | 16 GB or more |
| Disk | ~6 GB for models | ~9 GB with speaker detection |
| Tools | Xcode Command Line Tools (`xcode-select --install`) | — |

8 GB Macs handle transcription well. For summaries on 8 GB, use `qwen3-4b`
(`meetre config summary_model qwen3-4b`) or turn summaries off.

Disk breakdown: Whisper `large-v3-turbo` is about 1.5 GB, `Qwen3-8B-4bit` about
4.7 GB, and Python dependencies 1–3 GB (the `persons` extra adds PyTorch).

## Performance by Mac (approximate)

Real-time factor is minutes of audio transcribed per minute of compute (higher is
faster). Summary time is for a typical 30-minute meeting with `Qwen3-8B-4bit`. The
first run also downloads the models once.

| Chip | RAM | Transcribe (large-v3-turbo) | Summary (Qwen3-8B) | Notes |
|------|-----|------------------------------|---------------------|-------|
| M1 / M1 Pro | 8–16 GB | ~6–9x real-time | ~10–20 s | Great for transcripts; use `qwen3-4b` on 8 GB |
| M2 / M2 Pro/Max | 8–32 GB | ~8–12x real-time | ~8–15 s | Full pipeline runs smoothly |
| M3 / M3 Pro/Max | 16–36 GB | ~10–14x real-time | ~6–12 s | Comfortable everywhere |
| M4 / M4 Pro/Max | 16–48 GB | ~12–16x real-time | ~5–9 s | Fast |
| M5 / M5 Pro/Max | 24–48 GB+ | ~14–18x real-time | ~4–8 s | Effortless |

These are estimates, not benchmarks. They vary with thermals, other running apps,
meeting length, and whether speaker detection is enabled. A 30-minute meeting
typically transcribes in about 2–4 minutes on most M-series Macs.

## Manual install

If you prefer not to use `install.sh`:

```bash
pip install -e .            # core, including MLX transcription + summaries
pip install -e '.[menubar]' # the macOS menu-bar app (rumps + pyobjc)
pip install -e '.[persons]' # optional speaker diarization
pip install -e '.[cpu]'     # faster-whisper fallback for non-Apple-Silicon
```

## Menu bar

Click the `✦` icon:

- **Record…** opens a settings popup: meeting name, model, language, summary
  model, system-audio and speaker toggles, a speaker slider, and an editable AI
  prompt with a reset button. Choices are remembered for next time.
- Live status while working, for example `02:14`, a download bar, or
  "Transcribing…".
- **Summarize last → Apple Notes**, **Check for updates**, **Start at login**.

The app lives only in the menu bar (no Dock icon) and keeps running after you
close the terminal.

## Command line

```bash
meetre                       # interactive menu
meetre menubar               # launch the menu-bar app (detached)

meetre record --name "Standup"
meetre transcribe call.mp3   # re-transcribe an audio file
meetre localsummary          # summarise a transcript in-place (offline)
meetre summarize             # latest transcript to Apple Notes (local)
meetre list / open / devices
meetre model large-v3-turbo  # tiny | base | small | medium | large-v3 | large-v3-turbo
meetre persons on            # speaker detection
meetre speakers 3-6          # auto | exact (4) | range (3-6)
meetre update                # git pull + reinstall
meetre config                # view or edit all settings
```

## Speaker detection

Uses `pyannote`. A headcount hint improves accuracy noticeably.

```bash
pip install -e '.[persons]'
meetre config hf_token <token>     # free, from huggingface.co/settings/tokens
meetre persons on
meetre speakers 3-6
```

Accept the model terms at `huggingface.co/pyannote/speaker-diarization-3.1` and
`.../segmentation-3.0` first.

## Local summaries

Every recording is summarised on-device. The summary is embedded at the top of
the transcript and saved to Apple Notes, generated once.

```bash
meetre config summary_model qwen3-8b   # ~4.7 GB, best quality (default)
meetre config summary_model qwen3-4b   # ~2.5 GB, faster and lighter
meetre config summary_model gemma3-4b  # ~2.6 GB, 140+ languages
meetre config auto_summarize off       # transcript only
```

The prompt is editable in the menu bar or via
`meetre config summary_prompt "..."`.

## Privacy

Recording, transcription, diarization and summarization all run locally via MLX
and PyTorch. No audio or text is uploaded. The only network access is the
one-time model downloads from HuggingFace and the optional `git pull` update.

## Configuration

Stored at `~/.config/meetre/config.json`. Keys: `model`, `language`,
`person_detection`, `num_speakers`, `min_speakers`, `max_speakers`,
`transcripts_dir`, `audio_backup_dir`, `mic_device`, `system_device`,
`capture_system`, `native_system`, `hf_token`, `compute_type`, `summary_model`,
`auto_summarize`, `auto_notes`, `summary_prompt`, `auto_update`.

## Updating

The menu bar runs `git pull` on every launch. You can also use **Check for
updates** or run `meetre update`. The clone from Quick install already has the
remote configured.

## How it works

```
recorder.py      mic (sounddevice) + system audio (ScreenCaptureKit Swift helper) -> mixed 16 kHz WAV
transcriber.py   MLX Whisper (large-v3-turbo); pyannote diarization
summarizer.py    MLX-LM (Qwen3 / Gemma) with an editable prompt
transcript.py    Markdown writer (summary + timestamped, speaker-labelled body)
integrations.py  Apple Notes (AppleScript)
menubar.py       rumps + AppKit status-bar app
updater.py       git-pull self-update
autostart.py     login LaunchAgent
```

## Contributing

Pull requests are welcome. It is a small, hackable Python codebase.

```bash
bash install.sh
.venv/bin/python -m py_compile src/meetre/*.py   # quick sanity check
```

## License

MIT, see [LICENSE](LICENSE).
