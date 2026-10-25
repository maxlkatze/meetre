"""The post-recording pipeline, free of any UI.

Transcribe → (speakers) → name speakers → summary → title, publishing the
transcript (.md + Apple Note) progressively after each stage. Hosts — the rumps
menu bar and the headless JSON-RPC engine — subclass :class:`Hooks` to show
stages, progress, notifications and the streaming summary their own way.

Everything here may run on a worker thread; hooks are called from it too.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


def summary_error_hint(e: Exception) -> str:
    """Turn a raw summarizer error into an actionable message.

    The most common one is a llama.cpp too old for a newer architecture
    ("unknown model architecture"), which otherwise surfaces as a fleeting
    failure and an empty summary.
    """
    msg = str(e)
    low = msg.lower()
    if "unknown model architecture" in low or "unknown architecture" in low:
        return ("This model needs a newer llama.cpp: update it (brew upgrade llama.cpp, "
                "or delete ~/models/.llama.cpp so meetre fetches the latest), or pick "
                "another summary model. (" + msg + ")")
    return msg


class Hooks:
    """Pipeline → host reporting. Every method is a no-op by default."""

    def stage(self, text: str) -> None:
        """A new indeterminate stage ("Transcribing…")."""

    def progress(self, label: str, frac: Optional[float], icon: str = "🔍") -> None:
        """``frac`` None → just a stage label; a number → a progress bar."""

    def download(self, label: str, frac: float) -> None:
        """Model download progress (0–1)."""

    def download_done(self) -> None:
        """The download bar can go."""

    def notify(self, title: str, subtitle: str, msg: str) -> None:
        """A user-facing notification."""

    def summary_begin(self, model: str) -> None:
        """A summary starts streaming (``model`` = alias for display)."""

    def summary_feed(self, phase: str, text: str, stage: str = "") -> None:
        """Streaming summary text; see :func:`meetre.summarizer.summarize`."""

    def summary_finish(self, text: str) -> None:
        """The summary is complete (``text`` may be empty on failure)."""

    def published(self, out: dict, stage: str) -> None:
        """The transcript was (re)written: ``out["path"]``, ``out["note"]``;
        ``stage`` is "transcript", "speakers" or "done"."""


def start_live(cfg, recorder, on_update: Callable):
    """Start live captions for ``recorder``; None when unsupported."""
    from . import transcriber
    from .live import LiveTranscriber

    if not transcriber.live_supported(cfg.model):
        return None
    lbl = transcriber._labels(cfg.language)
    labels = {"mic": (cfg.my_name or "").strip() or lbl["local"],
              "system": lbl["remote"]}
    live = LiveTranscriber(recorder.live_paths(), cfg.model, cfg.language,
                           on_update, labels)
    live.start()
    return live


def begin_recording(cfg, path: Path):
    """Build and start a :class:`meetre.recorder.Recorder` per ``cfg``.

    Raises RuntimeError with a user-facing message when no source can start.
    Non-fatal source failures are left in ``recorder.start_errors``.
    """
    from . import recorder as rec
    from . import transcriber

    if transcriber.available_backend() is None:
        raise RuntimeError("No transcription backend installed (pip install mlx-whisper).")
    mic = cfg.mic_device if cfg.mic_device is not None else rec.default_input_device()
    native = bool(cfg.capture_system)
    if mic is None and not native:
        raise RuntimeError("No audio input device found.")
    recorder = rec.Recorder(mic_device=mic, system_device=None, native_system=native,
                            echo_cancellation=getattr(cfg, "voice_processing_mic", False))
    recorder.start(path)
    return recorder


def save_stems(stems: dict, mp3: Path) -> None:
    """Keep each source's 16 kHz stem next to the MP3 backup as
    ``<backup>.<role>.flac`` (lossless, so they stay sample-aligned)."""
    import soundfile as sf

    for role, stem in stems.items():
        if role in ("mic", "system") and Path(stem).exists():
            data, sr = sf.read(str(stem), dtype="float32", always_2d=False)
            sf.write(str(mp3.with_suffix(f".{role}.flac")), data, sr, format="FLAC")


def embed_summary(path: Path, summary: str) -> None:
    """Insert or replace the summary section of a transcript .md in place.

    Layout (see write_transcript): metadata ``---`` [summary ``---`` ``##
    Transcript``] body.
    """
    md = path.read_text()
    summary = summary.strip()
    if "## Transcript" in md:
        head, body = md.split("## Transcript", 1)
        meta = head.split("\n---\n", 1)[0]
    elif "\n---\n" in md:
        meta, body = md.split("\n---\n", 1)
    else:
        meta, body = "", md
    body = body.strip("\n")
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(f"{meta.rstrip()}\n\n---\n\n{summary}\n\n---\n\n## Transcript\n\n{body}\n")
    tmp.replace(path)


def backup_stamp(name: str) -> Optional[datetime]:
    """The recording start encoded in a backup's name
    (``2026-10-13_14-01-59_<slug>.mp3``), or None."""
    try:
        return datetime.strptime(name[:19], "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def backup_stems(mp3: Path) -> dict:
    """The per-source stems saved next to ``mp3`` by :func:`save_stems`
    (empty unless both are there — one alone misses half the meeting)."""
    stems = {role: mp3.with_suffix(f".{role}.flac") for role in ("mic", "system")}
    return stems if all(p.exists() for p in stems.values()) else {}


def decode_sources(tmp: Path, sources: dict):
    """Decode ``{role: audio file}`` (``mic``/``system`` stems, or one ``mix``)
    to 16 kHz WAVs in ``tmp``. Returns ``(mix_wav, stem_wavs, seconds)``."""
    import numpy as np
    import soundfile as sf

    stems, tracks = {}, []
    for role, src in sources.items():
        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16_000 and len(data):
            n = int(round(len(data) * 16_000 / sr))
            data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                             np.arange(len(data)), data).astype("float32")
        wav = tmp / f"{role}.wav"
        sf.write(str(wav), data, 16_000, subtype="PCM_16")
        if role != "mix":
            stems[role] = wav
        tracks.append(data)
    mix = tmp / "mix.wav"
    if stems:
        out = np.zeros(max(len(t) for t in tracks), dtype="float32")
        for t in tracks:
            out[: len(t)] += t
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 1.0:
            out /= peak
        sf.write(str(mix), out, 16_000, subtype="PCM_16")
    duration = max(len(t) for t in tracks) / 16_000
    return mix, stems, duration


def orphan_backups(cfg) -> List[dict]:
    """Audio backups without a transcript (the meeting failed, was deleted or
    never finished), newest first — candidates for :meth:`Pipeline.import_audio`."""
    backup = cfg.audio_backup_path
    if not backup.is_dir():
        return []
    root = cfg.transcripts_path
    have = {p.name[:19] for p in root.glob("*.md")} if root.is_dir() else set()
    out = []
    for mp3 in sorted(backup.glob("*.mp3"), reverse=True):
        started = backup_stamp(mp3.name)
        if started is None or mp3.name[:19] in have:
            continue
        out.append({"path": str(mp3), "name": mp3.name,
                    "started_at": started.isoformat(timespec="seconds"),
                    "size": mp3.stat().st_size, "stems": sorted(backup_stems(mp3))})
    return out


class Pipeline:
    """One processing job over the current config. Create a fresh one per job
    so it picks up config changes made in between."""

    def __init__(self, cfg, hooks: Optional[Hooks] = None):
        self.cfg = cfg
        self.hooks = hooks or Hooks()

    # -- helpers -----------------------------------------------------------------

    def tx_progress(self, done, total):
        """Adapter: transcription reports (seconds_done, total_seconds)."""
        self.hooks.progress("Transcribing", (done / total) if total else None, icon="✏️")

    def ensure_model(self, repo: str, label: str) -> None:
        """Download a model if needed, streaming progress to the host."""
        from . import downloads

        if downloads.is_cached(repo):
            return
        try:
            downloads.ensure_model(repo, lambda frac, *_: self.hooks.download(label, frac))
        finally:
            self.hooks.download_done()

    @staticmethod
    def _text(segments) -> str:
        return "\n".join(
            (f"{s.speaker}: " if getattr(s, "speaker", None) else "") + s.text
            for s in segments
        )

    # -- recording → transcript ----------------------------------------------------

    def process(self, recorder, title: str, started: datetime,
                auto_title: bool = False) -> Optional[dict]:
        """Stop ``recorder`` and run the whole pipeline. Returns the published
        ``out`` dict (None when nothing was saved). Never raises: failures are
        crash-logged and reported through ``hooks.notify``."""
        from . import recorder as rec
        from . import summarizer, transcriber
        from .transcript import _slugify

        cfg, hooks = self.cfg, self.hooks
        try:
            final_audio = recorder.stop()
            stems = dict(getattr(recorder, "stems", {}))
            duration = recorder.seconds

            # MP3 backup
            try:
                mp3 = cfg.audio_backup_path / f"{started.strftime('%Y-%m-%d_%H-%M-%S')}_{_slugify(title)}.mp3"
                rec.save_mp3(final_audio, mp3)
                # Per-source audio (lossless, sample-aligned) so the meeting
                # can be transcribed again with you vs. the call kept apart.
                save_stems(stems, mp3)
            except Exception:  # noqa: BLE001
                import sys

                from . import crashlog

                crashlog._write_exception("MP3 backup failed", *sys.exc_info())

            # Ensure the transcription model is present (shows a download bar).
            hooks.stage("Loading transcription model…")
            _tx_label = "Parakeet" if transcriber.is_parakeet(cfg.model) else "Whisper"
            self.ensure_model(transcriber.mlx_repo(cfg.model), f"{_tx_label} {cfg.model}")
            use_persons = cfg.person_detection

            # Progressive output: the transcript (.md + Apple Note) is written as
            # soon as transcription finishes, then rewritten in place once the
            # speakers are identified and again once the summary/title are in.
            out = {"title": title, "started": started, "duration": duration,
                   "backend": None, "path": None, "note": None}
            will_summarize = cfg.auto_summarize and summarizer.available()

            def early(segs, persons):
                if use_persons:
                    pending = ("⏳ Sprecher werden erkannt … Zusammenfassung folgt."
                               if will_summarize else "⏳ Sprecher werden erkannt …")
                else:
                    pending = "⏳ Zusammenfassung wird erstellt …" if will_summarize else None
                self.publish(out, segs, persons, pending=pending, stage="transcript")
                hooks.notify("meetre", f"📝 {out['title']} — transcript ready",
                             "Speakers and summary follow; the transcript updates itself."
                             if pending else out["path"].name)

            segments, backend, use_persons = self.transcribe_audio(
                final_audio, stems, on_early=early)
            if not segments:
                from . import crashlog

                crashlog.note("No speech detected", f"model={cfg.model} "
                              f"duration={duration:.1f}s stems={sorted(stems)}")
                hooks.notify("meetre", "Done", "No speech detected.")
                return None
            out["backend"] = backend

            if use_persons:
                self.name_speakers(segments)

            # Speakers are in: refresh the transcript while the summary runs.
            if cfg.person_detection and use_persons:
                self.publish(out, segments, use_persons,
                             pending="⏳ Zusammenfassung wird erstellt …"
                             if will_summarize else None, stage="speakers")

            # Generate the summary ONCE; reuse it for the transcript and Notes.
            summary = self.generate_summary(segments)

            # Secondary action: when no meeting name was given (e.g. "Record
            # with previous"), derive a concise title from the summary so the
            # transcript and Apple Note get a meaningful name instead of a
            # bare timestamp.
            if auto_title:
                title = self.generate_title(summary, segments, fallback=title)

            out["title"] = title
            hooks.stage("Saving…")
            self.publish(out, segments, use_persons, summary=summary, stage="done")
            path = out["path"]
            mins = int(duration // 60)
            done_msg = ("Summary + transcript ready" if summary
                        else "Transcript ready")
            hooks.notify("meetre", f"✓ {title} — done",
                         f"{done_msg} · {mins} min · {path.name}")

            for p in [final_audio, *stems.values()]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
            return out
        except Exception as e:  # noqa: BLE001
            import sys

            from . import crashlog

            crashlog._write_exception("Recording failed", *sys.exc_info())
            hooks.notify("meetre", "Recording failed", str(e))
            return None

    def transcribe_audio(self, mix: Path, stems: dict, on_early=None):
        """Transcribe a recording, splitting speakers when person detection is
        on. Returns ``(segments, backend, persons)``.

        With both stems, each source is transcribed on its own (you vs. the
        call); otherwise the mix is transcribed and pyannote labels it.
        ``on_early(segments, persons)`` gets the transcript before the slow
        speaker analysis."""
        from . import transcriber

        cfg, hooks = self.cfg, self.hooks
        use_persons = cfg.person_detection
        hooks.stage("Transcribing…")
        if "mic" in stems and "system" in stems:
            def on_transcribed(segs):
                if on_early is not None:
                    on_early(segs, True)
                if use_persons:
                    hooks.stage("Detecting speakers…")

            segments, backend = transcriber.transcribe_attributed(
                mix, stems, model=cfg.model, language=cfg.language,
                compute_type=cfg.compute_type, detect_speakers=use_persons,
                hf_token=cfg.hf_token, num_speakers=cfg.num_speakers,
                min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers,
                progress=self.tx_progress, diar_progress=hooks.progress,
                vad=cfg.vad, word_align=cfg.word_timestamps,
                merged_analysis=cfg.merged_analysis,
                on_transcribed=on_transcribed,
            )
            return segments, backend, True
        segments, backend = transcriber.transcribe(
            mix, model=cfg.model, language=cfg.language,
            compute_type=cfg.compute_type, progress=self.tx_progress,
            vad=cfg.vad, word_align=cfg.word_timestamps,
        )
        if segments and on_early is not None:
            on_early(segments, False)
        if segments and use_persons:
            hooks.stage("Detecting speakers…")
            try:
                segments = transcriber.diarize(
                    mix, segments, cfg.hf_token,
                    num_speakers=cfg.num_speakers,
                    min_speakers=cfg.min_speakers,
                    max_speakers=cfg.max_speakers,
                    progress=hooks.progress,
                )
            except RuntimeError as e:
                hooks.notify("meetre", "Speaker detection skipped", str(e))
                use_persons = False
        return segments, backend, use_persons

    def publish(self, out, segments, persons, *, summary="", pending=None,
                stage="done") -> None:
        """Write (or rewrite) the transcript .md and its Apple Note.

        ``out`` carries the meeting metadata plus the current ``path``/``note``
        so later stages update the same file and note in place. A changed
        title (auto-titling) renames the file.
        """
        from . import integrations, summarizer
        from .transcript import move_words, write_transcript

        cfg = self.cfg
        path = write_transcript(
            segments, cfg.transcripts_path, title=out["title"],
            started_at=out["started"], duration=out["duration"],
            model=cfg.model, backend=out["backend"] or "?",
            person_detection=persons, summary=summary,
        )
        old = out["path"]
        if old is not None and old != path:
            old.unlink(missing_ok=True)
            move_words(old, None)
        out["path"] = path

        if cfg.auto_notes and out["note"] != "failed":
            body = summarizer.transcript_body(path.read_text())
            try:
                if out["note"] is None:
                    out["note"] = integrations.add_to_apple_notes(
                        out["title"], body, summary_md=summary or None,
                        when=out["started"], pending=pending)
                else:
                    integrations.update_apple_note(
                        out["note"], out["title"], body, summary_md=summary or None,
                        when=out["started"], pending=pending)
            except RuntimeError as e:
                out["note"] = "failed"  # don't retry (and re-notify) every stage
                self.hooks.notify("meetre", "Apple Notes failed", str(e))
        self.hooks.published(out, stage)

    # -- LLM stages ------------------------------------------------------------------

    def name_speakers(self, segments) -> None:
        """Replace anonymous speaker labels with real names, in place.

        Your own side ("Me"/"Ich") becomes ``cfg.my_name``; the other labels are
        named by the local LLM from explicit evidence in the conversation
        (introductions, being addressed by name), verified against the text.
        """
        from . import summarizer, transcriber

        cfg = self.cfg
        lbl = transcriber._labels(cfg.language)
        mapping = {}
        if (cfg.my_name or "").strip():
            mapping[lbl["local"]] = cfg.my_name.strip()
        labels = list(dict.fromkeys(s.speaker for s in segments
                                    if s.speaker and s.speaker not in mapping))
        if (cfg.name_speakers and labels and summarizer.available()
                and cfg.summary_model):
            text = "\n".join(f"{s.speaker}: {s.text}" for s in segments if s.speaker)
            try:
                self.hooks.stage("Naming speakers…")
                self.ensure_model(summarizer.resolve_model(cfg.summary_model),
                                  f"Summary {cfg.summary_model}")
                found = summarizer.infer_speaker_names(
                    text, labels, model=cfg.summary_model, language=cfg.language)
                taken = set(mapping.values())
                mapping.update({k: v for k, v in found.items() if v not in taken})
            except Exception:  # noqa: BLE001 — naming is best-effort
                pass
        for s in segments:
            if s.speaker in mapping:
                s.speaker = mapping[s.speaker]

    def generate_summary(self, segments) -> str:
        """Run the local LLM over segments once; '' if disabled/unavailable."""
        from . import summarizer

        if not self.cfg.auto_summarize:
            return ""
        if not summarizer.available():
            self.hooks.notify("meetre", "Summary skipped", "Install mlx-lm for summaries.")
            return ""
        try:
            return self.summarize_text(self._text(segments))
        except Exception as e:  # noqa: BLE001
            self.hooks.notify("meetre", "Summary failed", summary_error_hint(e))
            return ""

    def summarize_text(self, text: str) -> str:
        """Ensure the model, then summarizer.summarize streamed through hooks.
        Raises on failure."""
        from . import summarizer

        cfg = self.cfg
        self.hooks.stage("Loading summary model…")
        self.ensure_model(summarizer.resolve_model(cfg.summary_model),
                          f"Summary {cfg.summary_model}")
        self.hooks.stage("Summarizing…")
        alias = cfg.summary_model
        if not alias or alias == "auto":
            alias = summarizer.default_model()
        summary = ""
        self.hooks.summary_begin(alias)
        try:
            summary = summarizer.summarize(
                text, model=cfg.summary_model, language=cfg.language,
                prompt=cfg.summary_prompt or None, on_progress=self.hooks.summary_feed)
            return summary
        finally:
            self.hooks.summary_finish(summary)

    def generate_title(self, summary, segments, fallback: str) -> str:
        """Derive a short meeting title from the summary (or transcript).

        Returns ``fallback`` (the timestamp title) if summaries are disabled,
        the backend is unavailable, or generation fails — never raises."""
        from . import summarizer

        if not self.cfg.auto_summarize or not summarizer.available():
            return fallback
        source = (summary or "").strip() or self._text(segments)
        if not source.strip():
            return fallback
        try:
            self.hooks.stage("Naming meeting…")
            title = summarizer.generate_title(
                source, model=self.cfg.summary_model, language=self.cfg.language)
            return title.strip() or fallback
        except Exception:  # noqa: BLE001
            return fallback

    # -- existing transcripts ----------------------------------------------------------

    def summarize_to_notes(self, path: Path) -> None:
        """Manual 'Summarize last': reuse the embedded summary, else generate,
        then save summary + transcript to Apple Notes."""
        from . import integrations, summarizer

        md = path.read_text()
        body = summarizer.transcript_body(md)
        # Recover the meeting title + date from the transcript metadata so the
        # Apple Note gets "MEETRE>{title}-{date}" rather than the filename stem.
        mtitle, when = summarizer.meeting_meta(md)
        title = mtitle or path.stem

        # Reuse the summary already embedded in the transcript if present.
        summary = summarizer.extract_summary(md)
        if not summary and self.cfg.auto_summarize and summarizer.available():
            try:
                summary = self.summarize_text(body)
            except Exception as e:  # noqa: BLE001
                self.hooks.notify("meetre", "Summary failed", str(e))

        self.hooks.stage("Saving to Apple Notes…")
        try:
            integrations.add_to_apple_notes(title, body, summary_md=summary or None, when=when)
            self.hooks.notify("meetre", "Apple Notes",
                              "Saved summary + transcript." if summary
                              else "Saved transcript (no summary).")
        except RuntimeError as e:
            self.hooks.notify("meetre", "Apple Notes failed", str(e))

    def resummarize(self, path: Path) -> str:
        """Generate a fresh summary for a saved transcript and embed it in the
        .md in place. Raises on failure; returns the summary."""
        from . import summarizer

        if not summarizer.available():
            raise RuntimeError("Summaries need mlx-lm: pip install mlx-lm")
        body = summarizer.transcript_body(path.read_text())
        summary = self.summarize_text(body)
        if not summary.strip():
            raise RuntimeError("The model produced no summary — try again or pick another model.")
        self.hooks.stage("Saving…")
        embed_summary(path, summary)
        return summary

    def retranscribe(self, path: Path) -> Path:
        """Transcribe a saved meeting again from its audio backup and rewrite
        the .md in place (same title; fresh speakers and, with auto-summarize,
        a fresh summary). Uses the per-source stems when they were kept, else
        the mixed MP3. Raises on failure; returns the transcript path."""
        import tempfile

        from . import summarizer, transcriber
        from .transcript import move_words, write_transcript

        cfg = self.cfg
        md = path.read_text()
        title, _ = summarizer.meeting_meta(md)
        stamp = path.stem[:19]
        try:
            started = datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            raise RuntimeError("Can't tell when this meeting was recorded.")
        backup = cfg.audio_backup_path
        mp3 = next(iter(sorted(backup.glob(f"{stamp}_*.mp3"))), None)
        found = backup_stems(mp3) if mp3 else {
            role: p for role in ("mic", "system")
            for p in sorted(backup.glob(f"{stamp}_*.{role}.flac"))[:1]}
        if not found and mp3 is None:
            raise RuntimeError("No audio backup for this meeting.")

        self.hooks.stage("Loading transcription model…")
        _tx_label = "Parakeet" if transcriber.is_parakeet(cfg.model) else "Whisper"
        self.ensure_model(transcriber.mlx_repo(cfg.model), f"{_tx_label} {cfg.model}")
        with tempfile.TemporaryDirectory(prefix="meetre_retx_") as tmp:
            mix, stems, duration = decode_sources(Path(tmp), found or {"mix": mp3})
            segments, backend, persons = self.transcribe_audio(mix, stems)
        if not segments:
            raise RuntimeError("No speech detected in the audio backup.")
        if persons:
            self.name_speakers(segments)
        summary = self.generate_summary(segments) if cfg.auto_summarize \
            else summarizer.extract_summary(md) or ""
        self.hooks.stage("Saving…")
        new = write_transcript(
            segments, cfg.transcripts_path, title=title or path.stem,
            started_at=started, duration=duration, model=cfg.model,
            backend=backend or "?", person_detection=persons, summary=summary)
        if new != path:
            path.unlink(missing_ok=True)
            move_words(path, None)
        return new

    def import_audio(self, src: Path) -> Path:
        """Turn an audio file into a meeting — an orphaned backup (the meeting
        failed or was deleted) or any MP3/WAV/FLAC/… from elsewhere.

        A backup keeps its recording time and title and uses its per-source
        stems when present; other files are backed up first so "Transcribe
        again" works for them later. Raises on failure; returns the transcript."""
        import tempfile

        from . import recorder as rec
        from . import transcriber
        from .transcript import _slugify

        cfg = self.cfg
        src = Path(src).expanduser()
        if not src.is_file():
            raise RuntimeError(f"File not found: {src}")
        started = backup_stamp(src.name)
        backup = cfg.audio_backup_path
        in_backup = started is not None and src.parent.resolve() == backup.resolve()
        if started is None:
            started = datetime.fromtimestamp(src.stat().st_mtime).replace(microsecond=0)
        default = f"Meeting {started.strftime('%Y-%m-%d %H:%M')}"
        slug = src.stem[20:] if in_backup else src.stem
        # A meeting recorded without a name gets a title from its summary.
        auto_title = slug in ("", "meeting", _slugify(default))
        if auto_title:
            title = default
        elif in_backup:
            title = slug.replace("-", " ").strip().capitalize()
        else:
            title = src.stem
        sources = backup_stems(src) if in_backup else {}

        self.hooks.stage("Loading transcription model…")
        _tx_label = "Parakeet" if transcriber.is_parakeet(cfg.model) else "Whisper"
        self.ensure_model(transcriber.mlx_repo(cfg.model), f"{_tx_label} {cfg.model}")
        with tempfile.TemporaryDirectory(prefix="meetre_import_") as tmp:
            try:
                mix, stems, duration = decode_sources(Path(tmp), sources or {"mix": src})
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Can't read {src.name}: {e}")
            if not in_backup:
                stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
                rec.save_mp3(mix, backup / f"{stamp}_{_slugify(title)}.mp3")
            segments, backend, persons = self.transcribe_audio(mix, stems)
        if not segments:
            raise RuntimeError(f"No speech detected in {src.name}.")
        if persons:
            self.name_speakers(segments)
        summary = self.generate_summary(segments)
        if auto_title:
            title = self.generate_title(summary, segments, fallback=title)
        self.hooks.stage("Saving…")
        out = {"title": title, "started": started, "duration": duration,
               "backend": backend, "path": None, "note": None}
        self.publish(out, segments, persons, summary=summary, stage="done")
        return out["path"]
