<div align="center">

# ✦ meetre

**Record any meeting on your Mac and turn it into a clean, speaker-labelled,
summarised transcript — 100% on-device.**

![macOS](https://img.shields.io/badge/macOS-13%2B-000000?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1–M5-0071e3)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Local](https://img.shields.io/badge/100%25-local-2ea44f)
![License](https://img.shields.io/badge/license-MIT-blue)

*Mic **and** system audio · German-first · MLX-accelerated · menu-bar app · no cloud, no API keys*

</div>

---

> 🍏 **macOS + Apple Silicon only.** meetre is built around Apple's **MLX** runtime
> and **ScreenCaptureKit**. It does not run on Windows or Linux, and Intel Macs
> are not supported for the full experience (see [requirements](#-system-requirements)).

## What it does

You click record. meetre captures **your microphone and the system audio** (the
other participants — Zoom, Meet, Teams, a YouTube call, anything), then:

1. 🎙 **Transcribes** locally with Whisper `large-v3-turbo` via **MLX** (Metal-accelerated).
2. 🗣 **Labels speakers** ("Speaker 1 / 2 …") with `pyannote` diarization *(optional)*.
3. 🧠 **Summarises** with a local LLM (`Qwen3-8B`) into *Zusammenfassung / Entscheidungen / Aufgaben / Offene Fragen*.
4. 📝 **Saves** a Markdown transcript + an MP3 backup, and pushes the summary to **Apple Notes**.

Nothing ever leaves your Mac. No accounts, no API keys, no usage fees.

## ✨ Features

- 🎧 **Mic + system audio**, mixed into one track — native via **ScreenCaptureKit**, no BlackHole needed.
- ⚡️ **MLX transcription** — fastest Whisper on Apple Silicon.
- 🌍 **German-first**, multilingual (English + 90 more); auto-detect supported.
- 🧑‍🤝‍🧑 **Speaker detection** with an optional headcount hint (`auto`, `4`, or `3-6`).
- 🧠 **Local LLM summaries** with a fully **editable prompt** (and a reset button).
- 🍎 **Menu-bar app** with live status, model-download progress bars, and a settings popup.
- 🔁 **Auto-update** (`git pull`) on every launch + **start at login**.
- 🗂 **MP3 backups** and **re-transcribe** any audio file later.
- 🔐 **Private by design** — everything runs on-device.

## 🖥 System requirements

| | Minimum | Recommended |
|---|---|---|
| **Mac** | Apple Silicon (M1) | M2 Pro / M3 / M4 / M5 |
| **macOS** | 13 Ventura (for ScreenCaptureKit) | 14 Sonoma or newer |
| **RAM** | 8 GB (use a smaller summary model) | 16 GB+ |
| **Disk** | ~6 GB for models | ~9 GB with speaker detection |
| **Tools** | Xcode Command Line Tools (`xcode-select --install`) | — |

> 💡 8 GB Macs work great for **transcription**. For summaries on 8 GB, pick
> `qwen3-4b` (`meetre config summary_model qwen3-4b`) or turn summaries off.

**Disk breakdown:** Whisper `large-v3-turbo` ≈ 1.5 GB · `Qwen3-8B-4bit` ≈ 4.7 GB ·
Python deps ≈ 1–3 GB (the `persons` extra adds PyTorch).

## 🍎 Which Mac can run it? (approximate)

Real-time factor = *minutes of audio transcribed per minute of compute* (higher
is faster). Summary time is for a typical 30-minute meeting with `Qwen3-8B-4bit`.
First run also downloads the models once.

| Chip | RAM | Transcribe (large-v3-turbo) | Summary (Qwen3-8B) | Verdict |
|------|-----|------------------------------|---------------------|---------|
| **M1 / M1 Pro** | 8–16 GB | ~6–9× real-time | ~10–20 s | ✅ Great for transcripts; use `qwen3-4b` on 8 GB |
| **M2 / M2 Pro/Max** | 8–32 GB | ~8–12× real-time | ~8–15 s | ✅ Smooth, full pipeline |
| **M3 / M3 Pro/Max** | 16–36 GB | ~10–14× real-time | ~6–12 s | ✅ Comfortable everywhere |
| **M4 / M4 Pro/Max** | 16–48 GB | ~12–16× real-time | ~5–9 s | 🚀 Fast |
| **M5 / M5 Pro/Max** | 24–48 GB+ | ~14–18× real-time | ~4–8 s | 🚀🚀 Effortless |

<sub>Numbers are rough guidance, not benchmarks — they vary with thermals,
other apps, meeting length, and whether speaker detection is on. A 30-min meeting
typically transcribes in ~2–4 minutes on most M-series Macs.</sub>

## 🚀 Quick start

```bash
git clone <your-fork-url> meetre && cd meetre
bash install.sh
```

The installer needs **no admin password**. It:

- uses your `python3`/`git` if present, else downloads a **local, relocatable
  Python** into `.runtime/` (no sudo),
- creates a venv and installs meetre + the menu-bar app,
- adds `meetre` to your `~/.local/bin` (PATH),
- asks for an optional **HuggingFace token** (for speaker detection),
- registers a **login startup item** and launches the **✦ menu bar**.

Then look for the **✦** icon in your top-right menu bar and click **● Record…**.

> First recording asks for **Screen Recording** permission (for system audio) and,
> if you save to Notes, **Automation** permission. Grant both once.

## 🎛 The menu bar

Click **✦** →

- **● Record…** — a settings popup: meeting name, model, language, **summary model**,
  system-audio + speaker toggles, a **speaker slider**, and an editable **AI prompt**
  (with **Reset prompt**). Every choice is remembered next time.
- Live status while working: `⏺ 02:14`, `⬇ Whisper ████░░░░ 42%`, `✦ Transcribing…`.
- **Summarize last → Apple Notes (local)**, **Check for updates**, **Start at login**.

The app lives **only** in the menu bar (no Dock icon) and keeps running after you
close the terminal.

## ⌨️ CLI

```bash
meetre                       # interactive menu
meetre menubar               # launch the menu-bar app (detached)

meetre record --name "Standup"
meetre transcribe call.mp3   # re-transcribe an audio file
meetre localsummary          # summarise a transcript in-place (offline)
meetre summarize             # latest transcript → Apple Notes (local)
meetre list / open / devices
meetre model large-v3-turbo  # tiny | base | small | medium | large-v3 | large-v3-turbo
meetre persons on            # speaker detection
meetre speakers 3-6          # auto | exact (4) | range (3-6)
meetre update                # git pull + reinstall
meetre config                # view / edit all settings
```

## 🗣 Speaker detection

Uses `pyannote` (the headcount hint greatly improves accuracy):

```bash
pip install -e '.[persons]'
meetre config hf_token <token>     # free, from huggingface.co/settings/tokens
meetre persons on
meetre speakers 3-6
```

You must accept the model terms at
`huggingface.co/pyannote/speaker-diarization-3.1` and `…/segmentation-3.0` first.

## 🧠 Local summaries

Every recording is summarised on-device and the summary is embedded at the top of
the transcript **and** saved to Apple Notes — generated **once**, no cloud.

```bash
meetre config summary_model qwen3-8b   # ~4.7 GB, best (default)
meetre config summary_model qwen3-4b   # ~2.5 GB, faster / lighter
meetre config summary_model gemma3-4b  # ~2.6 GB, 140+ languages
meetre config auto_summarize off       # transcript only
```

The prompt is fully editable in the menu bar (or `meetre config summary_prompt "…"`).

## 🔐 Privacy

Everything — recording, transcription, diarization, summarization — runs locally
via MLX and PyTorch on your Mac. No audio or text is ever uploaded. The only
network calls are the **one-time model downloads** from HuggingFace and the
optional **`git pull`** auto-update.

## ⚙️ Configuration

Stored at `~/.config/meetre/config.json`. Keys: `model`, `language`,
`person_detection`, `num_speakers`, `min_speakers`, `max_speakers`,
`transcripts_dir`, `audio_backup_dir`, `mic_device`, `system_device`,
`capture_system`, `native_system`, `hf_token`, `compute_type`, `summary_model`,
`auto_summarize`, `auto_notes`, `summary_prompt`, `auto_update`.

## 🔄 Updating

The menu bar runs `git pull` on every launch; you can also use **Check for
updates** or `meetre update`. Add a remote first:

```bash
git remote add origin <your-repo-url>
```

## 🧩 How it works

```
recorder.py    mic (sounddevice) + system audio (ScreenCaptureKit via a Swift helper) → mixed 16 kHz WAV
transcriber.py MLX whisper (large-v3-turbo)  ·  pyannote diarization
summarizer.py  MLX-LM (Qwen3 / Gemma) with an editable prompt
transcript.py  Markdown writer (summary + timestamped, speaker-labelled body)
integrations.py Apple Notes (AppleScript)
menubar.py     rumps + AppKit status-bar app
updater.py / autostart.py   git-pull self-update + login LaunchAgent
```

## 🤝 Contributing

PRs welcome! It's a small, hackable Python codebase.

```bash
bash install.sh                 # or: python3 -m venv .venv && .venv/bin/pip install -e '.[menubar]'
.venv/bin/python -m py_compile src/meetre/*.py   # quick sanity check
```

Ideas: more summary models, Linux/Intel fallbacks, smarter diarization, exporters.

## 📄 License

MIT — see [LICENSE](LICENSE). Built with ❤️ on Apple Silicon.
