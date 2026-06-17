"""meetre command-line interface.

Run ``meetre`` with no arguments for the interactive menu, or use subcommands:

    meetre record [--name NAME] [--persons/--no-persons]
    meetre list
    meetre open
    meetre devices
    meetre model [SIZE]
    meetre persons [on|off]
    meetre config [KEY VALUE]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import recorder as rec
from . import transcriber, ui
from .config import MODELS, Config
from .transcript import write_transcript

# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def _resolve_devices(cfg: Config):
    """Figure out which mic / system devices to record from.

    Returns ``(mic, system_device, native_system)``. When native capture is on
    (the default), ``system_device`` is ignored and system audio is taken from
    ScreenCaptureKit instead of a loopback device.
    """
    mic = cfg.mic_device if cfg.mic_device is not None else rec.default_input_device()
    system = None
    native = False
    if cfg.capture_system:
        if cfg.native_system:
            native = True
        else:
            system = cfg.system_device if cfg.system_device is not None else rec.find_loopback_device()
    return mic, system, native


def _ensure_model_cli(repo: str, label: str) -> None:
    """Download a model if missing, showing a rich progress bar."""
    from . import downloads

    if downloads.is_cached(repo):
        return
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

    with Progress(
        TextColumn("[cyan]{task.description}"), BarColumn(),
        DownloadColumn(), console=ui.console,
    ) as prog:
        task = prog.add_task(f"Downloading {label}", total=None)

        def cb(frac, done, total):
            if total:
                prog.update(task, total=total, completed=done)

        downloads.ensure_model(repo, cb)


def _transcribe_and_write(
    cfg: Config, audio_path: Path, *, title: str, started, duration: float,
    persons: Optional[bool],
) -> Optional[Path]:
    """Shared pipeline: transcribe → (optional) diarize → write transcript."""
    backend = transcriber.available_backend()
    use_persons = cfg.person_detection if persons is None else persons

    if backend == "mlx-whisper":
        _ensure_model_cli(transcriber.mlx_repo(cfg.model), f"Whisper {cfg.model}")
    with ui.console.status(f"[cyan]Transcribing with {cfg.model} ({backend})…", spinner="dots"):
        segments, used_backend = transcriber.transcribe(
            audio_path, model=cfg.model, language=cfg.language,
            compute_type=cfg.compute_type,
        )

    if not segments:
        ui.warn("No speech detected — nothing to transcribe.")
        return None

    if use_persons:
        try:
            with ui.console.status("[cyan]Detecting speakers…", spinner="dots"):
                segments = transcriber.diarize(
                    audio_path, segments, cfg.hf_token,
                    num_speakers=cfg.num_speakers,
                    min_speakers=cfg.min_speakers,
                    max_speakers=cfg.max_speakers,
                )
            ui.ok("Speaker detection complete")
        except RuntimeError as e:
            ui.warn(str(e))
            use_persons = False

    # --- Optional local-LLM summary (generated once, reused for Notes) ---
    summary = ""
    if cfg.auto_summarize:
        summary = _make_summary(cfg, segments)

    path = write_transcript(
        segments, cfg.transcripts_path,
        title=title, started_at=started, duration=duration,
        model=cfg.model, backend=used_backend, person_detection=use_persons,
        summary=summary,
    )
    ui.ok(f"Transcript saved: [bold]{path}[/bold]")

    # --- Auto-save the same summary + transcript to Apple Notes ---
    if cfg.auto_notes:
        from . import integrations

        try:
            integrations.add_to_apple_notes(
                path.stem, _segments_to_text(segments), summary_md=summary or None)
            ui.ok("Saved to Apple Notes")
        except RuntimeError as e:
            ui.warn(f"Apple Notes: {e}")

    return path


def _segments_to_text(segments) -> str:
    lines = []
    for s in segments:
        prefix = f"{s.speaker}: " if getattr(s, "speaker", None) else ""
        lines.append(f"{prefix}{s.text}")
    return "\n".join(lines)


def _make_summary(cfg: Config, segments) -> str:
    """Generate a local-LLM summary; returns '' on any failure (non-fatal)."""
    from . import summarizer

    if not summarizer.available():
        ui.warn("Summarization skipped — mlx-lm not installed (pip install mlx-lm).")
        return ""
    try:
        _ensure_model_cli(summarizer.resolve_model(cfg.summary_model), f"Summary {cfg.summary_model}")
        with ui.console.status(f"[cyan]Summarizing with {cfg.summary_model}…", spinner="dots"):
            summary = summarizer.summarize(
                _segments_to_text(segments), model=cfg.summary_model,
                language=cfg.language, prompt=cfg.summary_prompt or None,
            )
        ui.ok("Summary generated")
        return summary
    except Exception as e:  # noqa: BLE001
        ui.warn(f"Summary failed: {e}")
        return ""


def do_record(cfg: Config, name: Optional[str] = None, persons: Optional[bool] = None) -> None:
    backend = transcriber.available_backend()
    if backend is None:
        ui.error("No transcription backend installed.")
        ui.console.print(
            "  [bold]pip install mlx-whisper[/bold]   (Apple Silicon, recommended)\n"
            "  [bold]pip install faster-whisper[/bold]  (Intel / fallback)"
        )
        return

    mic, system, native = _resolve_devices(cfg)
    if mic is None and system is None and not native:
        ui.error("No audio input device found.")
        return

    if cfg.capture_system and not native and system is None:
        ui.warn(
            "System-audio capture is on but no loopback device was found — "
            "recording mic only. Install BlackHole or enable native capture "
            "(meetre config native_system on) to capture other participants."
        )

    sources = []
    if mic is not None:
        sources.append("🎤 mic")
    if native:
        sources.append("↩ system audio (ScreenCaptureKit)")
    elif system is not None:
        sources.append("↩ system audio")
    ui.info(f"Recording from: {', '.join(sources)}")

    # Warn before recording if person detection is requested but can't run.
    want_persons = cfg.person_detection if persons is None else persons
    if want_persons:
        ready, reason = transcriber.diarization_ready(cfg.hf_token)
        if not ready:
            ui.warn(f"Person detection enabled but not ready: {reason}")
            ui.warn("Recording will proceed; transcript will have no speaker labels.")

    title = name or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    import tempfile

    tmp_wav = Path(tempfile.gettempdir()) / f"meetre_{stamp}.wav"

    recorder = rec.Recorder(mic_device=mic, system_device=system, native_system=native)
    recorder.start(tmp_wav)
    for msg in recorder.start_errors:
        ui.warn(f"System audio unavailable: {msg}")

    stop_flag = threading.Event()

    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_flag.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    ui.console.print("\n[bold red]● REC[/bold red]  press [bold]Enter[/bold] to stop\n")

    with Live(refresh_per_second=4, console=ui.console) as live:
        while not stop_flag.is_set():
            elapsed = recorder.seconds
            m, s = divmod(int(elapsed), 60)
            live.update(Text(f"  ● {m:02d}:{s:02d}  recording…", style="red bold"))
            time.sleep(0.25)

    ui.info("Finishing recording…")
    final_audio = recorder.stop()
    duration = recorder.seconds
    ui.ok(f"Captured {int(duration)}s of audio")

    # --- Save compressed MP3 backup ---
    from .transcript import _slugify

    mp3_path = cfg.audio_backup_path / f"{stamp}_{_slugify(title)}.mp3"
    try:
        rec.save_mp3(final_audio, mp3_path)
        ui.ok(f"Audio backup: [bold]{mp3_path}[/bold]")
    except Exception as e:  # noqa: BLE001
        ui.warn(f"Could not write MP3 backup: {e}")

    # --- Transcribe (+ optional diarize) and write transcript ---
    written = _transcribe_and_write(
        cfg, final_audio, title=title, started=started,
        duration=duration, persons=persons,
    )
    # (Summary + Apple Notes are handled automatically in _transcribe_and_write.)
    # Drop the bulky WAV; the MP3 backup is the retained copy.
    try:
        final_audio.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def do_transcribe(cfg: Config, file: str, name: Optional[str] = None,
                  persons: Optional[bool] = None) -> None:
    """Re-transcribe an existing audio file (MP3, WAV, …)."""
    src = Path(file).expanduser()
    if not src.exists():
        ui.error(f"File not found: {src}")
        return
    if transcriber.available_backend() is None:
        ui.error("No transcription backend installed (pip install mlx-whisper).")
        return

    import tempfile

    import soundfile as sf

    ui.info(f"Loading {src.name}…")
    try:
        audio = transcriber._load_audio_16k(src)  # 16 kHz mono float32
    except Exception as e:  # noqa: BLE001
        ui.error(f"Could not read audio: {e}")
        return

    # Normalise to a temp 16 kHz WAV so both whisper and pyannote can read it.
    tmp_wav = Path(tempfile.gettempdir()) / f"meetre_retx_{src.stem}.wav"
    sf.write(str(tmp_wav), audio, 16_000, subtype="PCM_16")
    duration = len(audio) / 16_000
    title = name or src.stem
    started = datetime.fromtimestamp(src.stat().st_mtime)

    _transcribe_and_write(
        cfg, tmp_wav, title=title, started=started,
        duration=duration, persons=persons,
    )
    try:
        tmp_wav.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def do_local_summarize(cfg: Config, target: Optional[str] = None) -> None:
    """Summarize an existing transcript with the local LLM, in-place.

    Unlike ``do_summarize`` (which forwards to Claude Desktop), this runs fully
    offline via mlx-lm and writes a summary section into the transcript file.
    """
    from . import summarizer

    if not summarizer.available():
        ui.error("Local summary needs mlx-lm: pip install mlx-lm")
        return

    files = ui.list_transcripts(cfg.transcripts_path)
    if target is None:
        ui.transcripts_table(cfg.transcripts_path)
        if not files:
            return
        target = Prompt.ask("Transcript number or path", default="1")

    # Accept a list index or a file path.
    path = None
    if target.isdigit() and 1 <= int(target) <= len(files):
        path = files[int(target) - 1]
    else:
        p = Path(target).expanduser()
        path = p if p.exists() else cfg.transcripts_path / target
    if not path or not path.exists():
        ui.error(f"Transcript not found: {target}")
        return

    raw = path.read_text()
    # Body = everything after the "## Transcript" marker, else after first "---".
    if "## Transcript" in raw:
        header, body = raw.split("## Transcript", 1)
        body = body.strip()
    elif "\n---\n" in raw:
        head, body = raw.split("\n---\n", 1)
        header, body = head + "\n---\n", body.strip()
    else:
        header, body = "", raw

    try:
        _ensure_model_cli(summarizer.resolve_model(cfg.summary_model), f"Summary {cfg.summary_model}")
        with ui.console.status(f"[cyan]Summarizing with {cfg.summary_model}…", spinner="dots"):
            summary = summarizer.summarize(body, model=cfg.summary_model,
                                           language=cfg.language, prompt=cfg.summary_prompt or None)
    except Exception as e:  # noqa: BLE001
        ui.error(f"Summary failed: {e}")
        return

    new_text = f"{header.rstrip()}\n\n{summary.strip()}\n\n---\n\n## Transcript\n\n{body}\n"
    path.write_text(new_text)
    ui.ok(f"Summary added to [bold]{path}[/bold]")


def do_list(cfg: Config) -> None:
    ui.transcripts_table(cfg.transcripts_path)


def do_open(cfg: Config) -> None:
    cfg.transcripts_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["open", str(cfg.transcripts_path)])
    ui.ok(f"Opened {cfg.transcripts_path}")


def do_menubar(cfg: Config) -> None:
    """Launch the menu-bar app detached so the terminal can be closed."""
    import os

    log_dir = Path(os.path.expanduser("~/.cache/meetre"))
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / "menubar.out.log", "a")
    err = open(log_dir / "menubar.err.log", "a")
    subprocess.Popen(
        [sys.executable, "-m", "meetre.menubar"],
        stdout=out, stderr=err, stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from the terminal (survives closing it)
    )
    ui.ok("meetre menu bar launched (✦ top-right). You can close this terminal.")


def do_update(cfg: Config) -> None:
    from . import updater

    with ui.console.status("[cyan]Checking for updates (git pull)…", spinner="dots"):
        result = updater.update()
    if result.get("error"):
        ui.warn(f"Update: {result['error']}")
    elif result.get("updated"):
        ui.ok("Updated to the latest version. Restart meetre to apply.")
    else:
        ui.ok("Already up to date.")


def _latest_transcript(cfg: Config) -> Optional[Path]:
    files = sorted(cfg.transcripts_path.glob("*.md"), reverse=True)
    return files[0] if files else None


def do_summarize(cfg: Config, file: Optional[str] = None, notes: bool = True) -> None:
    """Summarize a transcript with the local LLM and save it to Apple Notes.

    Generates the summary fully offline (mlx-lm) and writes a note containing
    the summary + full transcript. With no ``file``, the most recent transcript
    is used. ``notes=False`` just prints the summary without touching Notes.
    """
    from . import integrations, summarizer

    path = Path(file).expanduser() if file else _latest_transcript(cfg)
    if path is None or not path.exists():
        ui.error("No transcript found to summarize.")
        return
    title = path.stem
    raw = path.read_text()
    body = summarizer.transcript_body(raw)

    # Reuse the summary already embedded in the transcript if present.
    summary = summarizer.extract_summary(raw)
    if summary:
        ui.info("Reusing existing summary from transcript")
    elif summarizer.available():
        try:
            _ensure_model_cli(summarizer.resolve_model(cfg.summary_model), f"Summary {cfg.summary_model}")
            with ui.console.status(f"[cyan]Summarizing with {cfg.summary_model}…", spinner="dots"):
                summary = summarizer.summarize(body, model=cfg.summary_model,
                                               language=cfg.language, prompt=cfg.summary_prompt or None)
            ui.ok("Local summary generated")
        except Exception as e:  # noqa: BLE001
            ui.warn(f"Summary failed: {e}")
    else:
        ui.warn("mlx-lm not installed — saving transcript without a summary.")

    if not notes:
        ui.console.print(summary or "[dim](no summary)[/dim]")
        return

    try:
        integrations.add_to_apple_notes(title, body, summary_md=summary or None)
        ui.ok("Saved to Apple Notes (summary + transcript)")
    except RuntimeError as e:
        ui.error(str(e))


def do_devices(cfg: Config) -> None:
    try:
        devices = rec.list_devices()
    except Exception as e:  # noqa: BLE001
        ui.error(f"Could not query audio devices: {e}")
        return
    ui.devices_table(devices)


def do_model(cfg: Config, size: Optional[str] = None) -> None:
    if size is None:
        ui.console.print(f"Current model: [bold]{cfg.model}[/bold]")
        ui.console.print(f"Available: {', '.join(MODELS)}")
        size = Prompt.ask("Select model", choices=MODELS, default=cfg.model)
    if size not in MODELS:
        ui.error(f"Unknown model '{size}'. Choose from: {', '.join(MODELS)}")
        return
    cfg.model = size
    cfg.save()
    ui.ok(f"Model set to {size}")


def do_persons(cfg: Config, value: Optional[str] = None) -> None:
    if value is None:
        cfg.person_detection = not cfg.person_detection
    else:
        cfg.person_detection = value.lower() in ("on", "true", "yes", "1")
    cfg.save()
    state = "enabled" if cfg.person_detection else "disabled"
    ui.ok(f"Person detection {state}")
    if cfg.person_detection and not cfg.hf_token:
        ui.warn(
            "Set a HuggingFace token (meetre config hf_token <token>) and install "
            "the extra: pip install 'meetre[persons]'"
        )


def _parse_speakers(spec: str):
    """Parse a speaker spec into (num, min, max).

    Accepts ``auto`` (or empty), an exact count ``"4"``, or a range ``"3-6"``
    (``"3-"`` and ``"-6"`` are open-ended). Returns ints or None for each.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "auto", "none"):
        return None, None, None
    if "-" in spec:
        lo, _, hi = spec.partition("-")
        lo_i = int(lo) if lo.strip() else None
        hi_i = int(hi) if hi.strip() else None
        if lo_i and hi_i and lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        return None, lo_i, hi_i
    return int(spec), None, None


def speakers_label(cfg: Config) -> str:
    if cfg.num_speakers:
        return f"exactly {cfg.num_speakers}"
    if cfg.min_speakers or cfg.max_speakers:
        lo = cfg.min_speakers if cfg.min_speakers else "?"
        hi = cfg.max_speakers if cfg.max_speakers else "?"
        return f"estimate {lo}–{hi}"
    return "auto-estimate"


def do_speakers(cfg: Config, spec: Optional[str] = None) -> None:
    if spec is None:
        ui.console.print(f"Speaker count: [bold]{speakers_label(cfg)}[/bold]")
        spec = Prompt.ask(
            "Set speakers (auto | exact e.g. 4 | range e.g. 3-6)", default="auto"
        )
    try:
        num, lo, hi = _parse_speakers(spec)
    except ValueError:
        ui.error(f"Invalid speaker spec '{spec}'. Use 'auto', '4', or '3-6'.")
        return
    cfg.num_speakers, cfg.min_speakers, cfg.max_speakers = num, lo, hi
    cfg.save()
    ui.ok(f"Speaker estimation: {speakers_label(cfg)}")
    if not cfg.person_detection:
        ui.warn("Person detection is off — enable it with `meetre persons on`.")


def do_config(cfg: Config, key: Optional[str] = None, value: Optional[str] = None) -> None:
    editable = ["model", "language", "transcripts_dir", "audio_backup_dir",
                "mic_device", "system_device", "capture_system", "native_system",
                "person_detection", "num_speakers", "min_speakers",
                "max_speakers", "hf_token", "compute_type",
                "summary_model", "auto_summarize", "auto_notes", "summary_prompt",
                "auto_update"]
    if key is None:
        from rich.table import Table

        t = Table(title="Configuration", border_style="cyan")
        t.add_column("Key", style="bold")
        t.add_column("Value")
        for k in editable:
            t.add_row(k, str(getattr(cfg, k)))
        from .config import CONFIG_PATH

        ui.console.print(t)
        ui.console.print(f"[dim]Config file: {CONFIG_PATH}[/dim]")
        return
    if key not in editable:
        ui.error(f"Unknown key '{key}'. Editable: {', '.join(editable)}")
        return
    if value is None:
        value = Prompt.ask(f"New value for {key}")
    # Coerce types based on current value.
    current = getattr(cfg, key)
    if isinstance(current, bool):
        coerced = value.lower() in ("on", "true", "yes", "1")
    elif isinstance(current, int) or key in ("mic_device", "system_device",
                                              "num_speakers", "min_speakers", "max_speakers"):
        coerced = int(value) if value not in ("", "none", "None") else None
    elif value in ("", "none", "None"):
        coerced = None
    else:
        coerced = value
    setattr(cfg, key, coerced)
    cfg.save()
    ui.ok(f"{key} = {coerced}")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

MENU = [
    ("r", "Record a meeting", "record"),
    ("t", "Re-transcribe an audio file (MP3/WAV)", "transcribe"),
    ("z", "Summarize a transcript → Apple Notes (local LLM)", "summarize"),
    ("u", "Summarize a transcript in-place (local LLM)", "localsummary"),
    ("b", "Launch the menu-bar app", "menubar"),
    ("l", "List transcripts", "list"),
    ("o", "Open transcripts folder", "open"),
    ("d", "Show audio devices", "devices"),
    ("m", "Select model", "model"),
    ("p", "Toggle person detection", "persons"),
    ("s", "Set speaker count (auto / 4 / 3-6)", "speakers"),
    ("c", "Edit configuration", "config"),
    ("q", "Quit", "quit"),
]


def interactive(cfg: Config) -> None:
    ui.banner()
    while True:
        cfg = Config.load()
        ui.console.print()
        ui.status_panel(cfg, transcriber.available_backend())
        ui.console.print()
        for key, label, _ in MENU:
            ui.console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")
        ui.console.print()
        choice = Prompt.ask("Select", default="r").strip().lower()
        action = next((a for k, _, a in MENU if k == choice or a == choice), None)

        if action == "quit":
            ui.console.print("[dim]bye 👋[/dim]")
            return
        elif action == "record":
            name = Prompt.ask("Meeting name", default="")
            do_record(cfg, name=name or None)
        elif action == "transcribe":
            f = Prompt.ask("Path to audio file (MP3/WAV)", default="")
            if f:
                do_transcribe(cfg, f)
            else:
                ui.warn("No file given")
        elif action == "summarize":
            f = Prompt.ask("Transcript path (blank = latest)", default="")
            do_summarize(cfg, f or None)
        elif action == "localsummary":
            do_local_summarize(cfg)
        elif action == "menubar":
            do_menubar(cfg)
        elif action == "list":
            do_list(cfg)
        elif action == "open":
            do_open(cfg)
        elif action == "devices":
            do_devices(cfg)
        elif action == "model":
            do_model(cfg)
        elif action == "persons":
            do_persons(cfg)
        elif action == "speakers":
            do_speakers(cfg)
        elif action == "config":
            key = Prompt.ask("Config key (blank to view all)", default="")
            do_config(cfg, key or None)
        else:
            ui.warn("Unknown option")
        ui.console.print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meetre", description="macOS meeting recorder + transcripts")
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser("record", help="Record a meeting and transcribe it")
    pr.add_argument("--name", help="Meeting name")
    pr.add_argument("--persons", dest="persons", action="store_true", help="Enable person detection")
    pr.add_argument("--no-persons", dest="persons", action="store_false", help="Disable person detection")
    pr.set_defaults(persons=None)

    pt = sub.add_parser("transcribe", help="Re-transcribe an existing audio file (MP3/WAV)")
    pt.add_argument("file", help="Path to the audio file")
    pt.add_argument("--name", help="Meeting name (defaults to the file name)")
    pt.add_argument("--persons", dest="persons", action="store_true", help="Enable person detection")
    pt.add_argument("--no-persons", dest="persons", action="store_false", help="Disable person detection")
    pt.set_defaults(persons=None)

    pz = sub.add_parser("summarize", help="Summarize a transcript locally → Apple Notes")
    pz.add_argument("file", nargs="?", help="Transcript .md path (defaults to the latest)")
    pz.add_argument("--no-notes", dest="notes", action="store_false",
                    help="Just print the summary; don't write to Apple Notes")
    pz.set_defaults(notes=True)

    pu = sub.add_parser("localsummary", help="Summarize a transcript locally (offline LLM)")
    pu.add_argument("file", nargs="?", help="Transcript .md path or list number (defaults to prompt)")

    sub.add_parser("menubar", help="Launch the macOS menu-bar app (default)")
    sub.add_parser("cli", help="Start the interactive text menu")
    sub.add_parser("update", help="Update meetre to the latest version (git pull)")
    sub.add_parser("list", help="List saved transcripts")
    sub.add_parser("open", help="Open the transcripts folder in Finder")
    sub.add_parser("devices", help="List audio input devices")

    pm = sub.add_parser("model", help="Select transcription model")
    pm.add_argument("size", nargs="?", choices=MODELS)

    pp = sub.add_parser("persons", help="Toggle person detection")
    pp.add_argument("value", nargs="?", choices=["on", "off"])

    ps = sub.add_parser("speakers", help="Set speaker count: auto | N | N-M (e.g. 3-6)")
    ps.add_argument("spec", nargs="?", help="auto, an exact count, or a range like 3-6")

    pc = sub.add_parser("config", help="View or edit configuration")
    pc.add_argument("key", nargs="?")
    pc.add_argument("value", nargs="?")

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load()

    try:
        if args.command is None:
            do_menubar(cfg)  # default: launch the menu-bar app
        elif args.command == "cli":
            interactive(cfg)
        elif args.command == "record":
            do_record(cfg, name=args.name, persons=args.persons)
        elif args.command == "transcribe":
            do_transcribe(cfg, args.file, name=args.name, persons=args.persons)
        elif args.command == "summarize":
            do_summarize(cfg, args.file, notes=args.notes)
        elif args.command == "localsummary":
            do_local_summarize(cfg, args.file)
        elif args.command == "menubar":
            do_menubar(cfg)
        elif args.command == "update":
            do_update(cfg)
        elif args.command == "list":
            do_list(cfg)
        elif args.command == "open":
            do_open(cfg)
        elif args.command == "devices":
            do_devices(cfg)
        elif args.command == "model":
            do_model(cfg, args.size)
        elif args.command == "persons":
            do_persons(cfg, args.value)
        elif args.command == "speakers":
            do_speakers(cfg, args.spec)
        elif args.command == "config":
            do_config(cfg, args.key, args.value)
    except KeyboardInterrupt:
        ui.console.print("\n[dim]interrupted[/dim]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
