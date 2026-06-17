# meetre

Record a meeting on your Mac and turn it into a clean, speaker-labelled,
summarised transcript. Everything runs on-device.

![macOS](https://img.shields.io/badge/macOS-13%2B-000000?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1–M5-0071e3)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

meetre is **macOS and Apple Silicon only**. It is built on Apple's MLX runtime
and ScreenCaptureKit; it does not run on Windows or Linux, and Intel Macs are not
supported for the full pipeline.

## Quick install

No git or admin required — one line downloads and installs everything:

```bash
cd ~ && curl -fsSL https://github.com/maxlkatze/meetre/archive/refs/heads/main.tar.gz | tar -xz && cd meetre-main && bash install.sh
```

Already have git? You can clone instead:

```bash
cd ~ && git clone https://github.com/maxlkatze/meetre.git && cd meetre && bash install.sh
```

> ⚠️ **Don't install into `~/Downloads`, `~/Desktop` or `~/Documents`.** macOS
> privacy protection (TCC) blocks background apps from reading those folders, so
> the login item can't start and the `✦` menu-bar icon silently disappears after
> launch or at the next login. The commands above install into your home folder
> (`~/meetre-main` / `~/meetre`); a plain folder like `~/meetre` is ideal. The
> installer refuses these protected locations and tells you how to move out.

The installer needs no admin password. It uses your existing `python3`/`git` if
present, otherwise downloads a **local, relocatable Python** and a **local git**
(via micromamba) into `.runtime/`. It then creates a virtual environment,
installs meetre and the menu-bar app, links `meetre` into `~/.local/bin`, asks for
an optional HuggingFace token, links the install to the repo so auto-update works,
registers a login startup item, and launches the menu bar.

After installing, look for the `✦` icon in the top-right menu bar and click
**Record…**. Run `meetre` any time to open the menu bar, or `meetre cli` for the
text menu.

> The first recording asks for the **Screen Recording** permission (system audio).
> Saving to Notes asks for **Automation** permission. Both are one-time.

## What it does

meetre captures your microphone and the system audio (the other participants in
Zoom, Meet, Teams, etc.), then:

1. Transcribes locally with Whisper `large-v3-turbo` via MLX (Metal-accelerated).
2. Optionally labels speakers using `pyannote` diarization.
3. Summarises with a local LLM (Qwen3.5 / Gemma 4, auto-picked for your RAM)
   into Zusammenfassung / Entscheidungen / Aufgaben / Offene Fragen.
4. Writes a Markdown transcript and an MP3 backup, and saves the summary to Apple
   Notes.

No audio or text leaves the machine. No accounts, no API keys.

## Features

- Microphone and system audio, mixed into one track. System audio is captured
  natively via ScreenCaptureKit, so no BlackHole or loopback driver is needed.
- MLX transcription, the fastest Whisper backend on Apple Silicon — plus an
  optional **NVIDIA Parakeet TDT v3** model (multilingual, very fast) to compare.
- Automatic loudness levelling before transcription: soft-spoken or distant
  participants are boosted to match louder speakers (gated so silence and
  background noise aren't amplified), which improves accuracy on quiet voices.
  The saved recording stays untouched.
- Optional **VAD** (Silero) drops silence-only segments — fewer Whisper
  hallucinations during quiet stretches — and **phoneme forced alignment**
  (torchaudio MMS_FA) attaches exact per-word timestamps. Both on-device.
- German by default, with multilingual and auto-detect support.
- Speaker detection with an optional headcount hint (`auto`, `4`, or `3-6`).
- Local LLM summaries with a fully editable prompt. Newest models built in:
  Qwen3.5 and Gemma 4 (strongest multilingual), run in fast direct mode.
  `auto` picks the best that fits your machine.
- Menu-bar app with a real image icon, live status, progress bars (model
  download, transcription, speaker detection), native notifications when a
  meeting is ready, and a settings popup.
- An **About meetre** submenu (version, check for updates, restart, start at
  login, quit).
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

8 GB Macs handle transcription well. For summaries on 8 GB, use `qwen3.5-4b`
(`meetre summary-model qwen3.5-4b`) or turn summaries off. Leaving the summary
model on `auto` always picks the best model that fits your RAM.

Disk breakdown: Whisper `large-v3-turbo` is about 1.5 GB, a small summary model
(`qwen3.5-4b`) about 2.4 GB up to `qwen3.5-35b` at ~20 GB, and Python
dependencies 1–3 GB (the `persons` extra adds PyTorch).

## Performance by Mac (approximate)

Real-time factor is minutes of audio transcribed per minute of compute (higher is
faster). Summary time is for a typical 30-minute meeting with a small/mid summary
model. The first run also downloads the models once.

| Chip | RAM | Transcribe (large-v3-turbo) | Summary | Auto summary model | Notes |
|------|-----|------------------------------|---------|--------------------|-------|
| M1 / M1 Pro | 8–16 GB | ~6–9x real-time | ~10–25 s | `qwen3.5-4b` / `gemma4-12b` | Great for transcripts; full pipeline from 16 GB |
| M2 / M2 Pro/Max | 8–32 GB | ~8–12x real-time | ~8–20 s | `gemma4-12b` / `qwen3.5-9b` | Full pipeline runs smoothly |
| M3 / M3 Pro/Max | 16–36 GB | ~10–14x real-time | ~8–18 s | `qwen3.5-9b` / `qwen3.5-35b` | Comfortable everywhere |
| M4 / M4 Pro/Max | 16–48 GB | ~12–16x real-time | ~6–15 s | `qwen3.5-35b` | Fast |
| M5 / M5 Pro/Max | 24–48 GB+ | ~14–18x real-time | ~5–12 s | `qwen3.5-35b` / `qwen3.5-27b` | Effortless |

These are estimates, not benchmarks. They vary with thermals, other running apps,
meeting length, and whether speaker detection is enabled. A 30-minute meeting
typically transcribes in about 2–4 minutes on most M-series Macs.

## Manual install

If you prefer not to use `install.sh`:

```bash
pip install -e .            # core, including MLX transcription + summaries
pip install -e '.[menubar]' # the macOS menu-bar app (rumps + pyobjc)
pip install -e '.[persons]' # optional speaker diarization (pyannote 4.x)
pip install -e '.[parakeet]'# optional Parakeet TDT v3 transcription model
pip install -e '.[align]'   # optional VAD + phoneme word-alignment (Whisper)
pip install -e '.[cpu]'     # faster-whisper fallback for non-Apple-Silicon
```

## Menu bar

The app lives only in the menu bar (no Dock icon) and keeps running after you
close the terminal. It shows a real image icon (`✦`-style sparkle, tinted to
match light/dark menus) and stays live even while the menu is open.

Click the icon:

- **Record…** — opens a settings popup: meeting name, transcription model,
  language, summary model, system-audio and speaker toggles, a speaker slider,
  and an editable AI prompt with a reset button. Choices are remembered.
- **Stop** — finishes the recording and runs the pipeline in the background.
- **Model / Language / Summary model / Speakers** — quick pickers. The summary
  picker is sized for your machine: too-large models are greyed out, `✓` marks
  downloaded ones.
- **System audio** and **Person detection** toggles.
- **Settings…**, **Summarize last → Apple Notes (local)**, **Open transcripts
  folder**, and **Downloaded models** (click one to uninstall and free space).
- **About meetre** submenu — version, **Check for updates**, **Restart meetre**,
  **Start at login**, **Quit meetre**.

While working, the menu bar shows live status — recording time `⏺ 02:14`, a
download bar `⬇ 45%`, or a spinning `Transcribing… / Summarizing…` — and posts a
native notification (with the app icon) such as **✓ Standup — done · 31 min**
when the transcript and summary are ready.

## Command line

```bash
meetre                       # launch the menu-bar app (default)
meetre cli                   # interactive text menu
meetre menubar               # launch the menu-bar app (detached)

meetre record --name "Standup"   # record + transcribe (--persons to force diarization)
meetre transcribe call.mp3       # re-transcribe an audio file (MP3/WAV)
meetre localsummary              # summarise a transcript in-place (offline)
meetre summarize                 # latest transcript → Apple Notes (local)
meetre list                      # list saved transcripts
meetre open                      # open the transcripts folder in Finder
meetre devices                   # list audio input devices

meetre model large-v3-turbo      # tiny | base | small | medium | large-v3 | large-v3-turbo | parakeet-tdt-v3
meetre summary-model             # show summary models, sizes, and what fits
meetre summary-model qwen3.5-35b # set summary model (alias | auto | off | HF repo id)
meetre models                    # list downloaded models; pass a name/number to uninstall
meetre persons on                # speaker detection (on | off)
meetre speakers 3-6              # auto | exact (4) | range (3-6)

meetre update                    # git pull + reinstall
meetre config                    # view all settings
meetre config <key> <value>      # set one setting
```

## Speaker detection

meetre records your mic and the system audio as separate stems, so it can tell
**you (and anyone in your room) apart from the remote participants** even before
diarization. The transcript is produced from one mixed timeline (accurate
timestamps); each line is then attributed to a side by which stem was active, and
`pyannote` runs on each stem separately to split multiple speakers per side.

- One person on your side: labelled `Ich` (de) / `Me` (en).
- Several people in your room: `Vor Ort 1/2…` (de) / `Local 1/2…` (en).
- Remote participants: `Sprecher 1/2…` (de) / `Speaker 1/2…` (en).

Diarizing clean single-source stems is far more reliable than diarizing a mono
mix of mic + system audio.

```bash
pip install -e '.[persons]'
meetre config hf_token <token>     # free, from huggingface.co/settings/tokens
meetre persons on
meetre speakers 3-6                # headcount hint for the remote side
```

Speaker detection uses the `pyannote/speaker-diarization-community-1` pipeline
(pyannote.audio 4.x) — accept its model terms at
`huggingface.co/pyannote/speaker-diarization-community-1` first. It runs fully
on-device (not the cloud `precision-2` variant), audio is fed straight from
memory (no FFmpeg/torchcodec needed), and pyannote's usage telemetry is
disabled. Live progress (segmenting / analyzing voices / assigning) shows on the
menu-bar bar.

## Local summaries

Every recording is summarised on-device. The summary is embedded at the top of
the transcript and saved to Apple Notes, generated once. Leave the model on
`auto` and meetre runs the best one that fits your RAM.

Built-in models (current generation, June 2026):

| Alias | Repo | ~Size | Best for |
|-------|------|-------|----------|
| `qwen3.5-2b` | `Qwen3.5-2B-MLX-4bit` | 1.3 GB | 8 GB Macs, fastest |
| `qwen3.5-4b` | `Qwen3.5-4B-MLX-4bit` | 2.4 GB | small + fast |
| `gemma4-e4b` | `gemma-4-e4b-it-4bit` | 3.4 GB | minimal, 140+ languages |
| `qwen3.5-9b` | `Qwen3.5-9B-4bit` | 5.0 GB | balanced, fits 16 GB |
| `gemma4-12b` | `gemma-4-12B-4bit` | 6.8 GB | strong multilingual |
| `mistral-24b` | `Mistral-Small-3.2-24B-…-4bit` | 13.3 GB | concise all-rounder |
| `gemma4-26b` | `gemma-4-26b-a4b-it-4bit` | 14.5 GB | best German / multilingual |
| `qwen3.5-27b` | `Qwen3.5-27B-4bit` | 15 GB | top dense quality |
| `qwen3.5-35b` | `Qwen3.5-35B-A3B-4bit` | 20 GB | best all-round (MoE, fast) |
| `qwen3.5-122b` | `Qwen3.5-122B-A10B-MLX-4bit` | 66 GB | high-RAM Studio |
| `qwen3.5-397b` | `Qwen3.5-397B-A17B-4bit` | 210 GB | Mac Studio Ultra only |

All models answer directly (no slow hidden reasoning pass) for fast, reliable
summaries. Older `qwen3-*` / `gemma3-*` aliases still resolve for existing
configs.

```bash
meetre summary-model             # list models, sizes, and what fits your Mac
meetre summary-model auto        # best that fits (default)
meetre summary-model qwen3.5-35b # best all-round on 32 GB+
meetre summary-model qwen3.5-4b  # light + fast for 8 GB
meetre summary-model off         # transcript only
```

The prompt is editable in the menu bar or via
`meetre config summary_prompt "..."`.

## Privacy

Recording, transcription, diarization and summarization all run locally via MLX
and PyTorch. No audio or text is uploaded. pyannote.audio's usage telemetry is
disabled, and the cloud diarization variant is never used. The only network
access is the one-time model downloads from HuggingFace and the optional
`git pull` update.

## Configuration

Stored at `~/.config/meetre/config.json`. Keys: `model`, `language`,
`person_detection`, `num_speakers`, `min_speakers`, `max_speakers`,
`transcripts_dir`, `audio_backup_dir`, `mic_device`, `system_device`,
`capture_system`, `native_system`, `hf_token`, `compute_type`, `vad`,
`word_timestamps`, `summary_model`, `auto_summarize`, `auto_notes`,
`summary_prompt`, `auto_update`.

`vad` (default on) drops silence-only Whisper segments; `word_timestamps`
(default on) adds per-word times via phoneme alignment. Both need the `align`
extra and apply to the Whisper path. Turn off with `meetre config vad off` /
`meetre config word_timestamps off`.

## Updating

The menu bar runs `git pull` on every launch. You can also use **Check for
updates** or run `meetre update`. The clone from Quick install already has the
remote configured.

## How it works

```
recorder.py      mic (sounddevice) + system audio (ScreenCaptureKit Swift helper) -> mixed 16 kHz WAV
transcriber.py   MLX Whisper / Parakeet TDT v3; gated AGC level + pyannote diarization
align.py         Silero VAD (silence filtering) + MMS_FA phoneme word-alignment
summarizer.py    MLX-LM (Qwen3.5 / Gemma 4, direct mode) with an editable prompt
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
