"""Headless engine for the native app: JSON-RPC 2.0 over stdin/stdout.

``python -m meetre engine`` (or ``meetre engine``) reads one JSON request per
line on stdin and writes responses + event notifications, one JSON object per
line, to stdout. stderr is free-form logging. The protocol is documented in
docs/ROADMAP.md ("Engine protocol (v1)").

stdout must stay pure protocol, but MLX/HF/pyannote print to it freely — so on
start the real stdout fd is duplicated for protocol use and fd 1 (plus
``sys.stdout``) is pointed at stderr.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .pipeline import Hooks

PROTOCOL = 1
LANGUAGES = [{"code": "de", "label": "Deutsch"}, {"code": "en", "label": "English"},
             {"code": None, "label": "Auto-detect"}]
STREAM_INTERVAL = 0.1   # max rate of summary.stream / chat.delta events
TICK = 0.5              # state event cadence while recording/processing


class RPCError(Exception):
    def __init__(self, message: str, code: int = -32000, hint: Optional[str] = None):
        super().__init__(message)
        self.code, self.hint = code, hint


# ---------------------------------------------------------------------------
# Transcript .md parsing (library view)
# ---------------------------------------------------------------------------

_SECTION_KEYS = {
    "zusammenfassung": "summary", "summary": "summary",
    "entscheidungen": "decisions", "decisions": "decisions",
    "aufgaben": "tasks", "aufgaben / to-dos": "tasks", "to-dos": "tasks",
    "action items": "tasks", "tasks": "tasks",
    "offene fragen": "questions", "open questions": "questions",
}
_NONE = {"keine", "none", "keine.", "none.", "**keine**", "**none**", "—", "-"}
_STAMP = re.compile(r"^`\[(\d+(?::\d+){1,2})\]`\s*(.*)$")


def _seconds(ts: str) -> float:
    total = 0.0
    for part in ts.split(":"):
        total = total * 60 + float(part or 0)
    return total


def _bullets(text: str) -> List[str]:
    items: List[str] = []
    for line in text.splitlines():
        st = line.strip()
        if st[:2] in ("- ", "* "):
            item = st[2:].strip()
            if item.lower().strip("„“\"") not in _NONE:
                items.append(item)
        elif st and items and line.startswith((" ", "\t")):
            items[-1] += " " + st  # wrapped continuation
    return items


def _task(item: str) -> Dict[str, Optional[str]]:
    """'**Owner:** task' / 'Owner: task' / 'task — Owner' → {text, owner}."""
    m = re.match(r"^\*\*(.+?):?\*\*:?\s*(.+)$", item)
    if m:
        return {"text": m.group(2).strip(), "owner": m.group(1).strip().rstrip(":")}
    m = re.match(r"^([^:]{1,40}):\s+(.+)$", item)
    if m and len(m.group(1).split()) <= 4:
        return {"text": m.group(2).strip(), "owner": m.group(1).strip()}
    m = re.match(r"^(.+?)\s+—\s+([^—,]{1,40})(?:,.*)?$", item)
    if m:
        return {"text": m.group(1).strip(), "owner": m.group(2).strip()}
    return {"text": item, "owner": None}


def parse_sections(summary_md: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"summary": "", "decisions": [], "tasks": [], "questions": []}
    if not summary_md.strip():
        return out
    parts = re.split(r"^##\s+(.+)$", summary_md, flags=re.M)
    # parts = [pre, head1, body1, head2, body2, …]
    if parts[0].strip():
        out["summary"] = parts[0].strip()
    for head, body in zip(parts[1::2], parts[2::2]):
        key = _SECTION_KEYS.get(head.strip().lower().rstrip(":"))
        if key is None:
            key = next((v for k, v in _SECTION_KEYS.items()
                        if head.strip().lower().startswith(k)), None)
        if key == "summary":
            out["summary"] = (out["summary"] + "\n\n" + body.strip()).strip()
        elif key == "tasks":
            out["tasks"] += [_task(b) for b in _bullets(body)]
        elif key in ("decisions", "questions"):
            out[key] += _bullets(body)
    return out


def parse_turns(body: str) -> List[Dict[str, Any]]:
    """Transcript body → [{speaker, start, text}], consecutive lines of the
    same speaker merged. Without speaker headers each line is its own turn."""
    turns: List[Dict[str, Any]] = []
    speaker: Optional[str] = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = re.match(r"^\*\*(.+?)\*\*$", line)
        if head:
            speaker = head.group(1)
            continue
        m = _STAMP.match(line)
        start, text = (_seconds(m.group(1)), m.group(2)) if m else (None, line)
        if speaker is not None and turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": speaker, "start": start, "text": text})
    return turns


def _meta(md: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"title": None, "started_at": None, "duration": None,
                            "model": None, "backend": None, "speakers": []}
    for line in md.splitlines()[:16]:
        if meta["title"] is None and line.startswith("# "):
            # Older auto-titles could start with ":" — display them clean.
            meta["title"] = line[2:].strip().lstrip(":-–— ") or None
        m = re.match(r"^- \*\*(.+?):\*\*\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "date":
            try:
                meta["started_at"] = datetime.strptime(val, "%Y-%m-%d %H:%M:%S").isoformat()
            except ValueError:
                meta["started_at"] = val
        elif key == "duration":
            meta["duration"] = _seconds(val) if re.match(r"^\d+(:\d+)+$", val) else None
        elif key == "model":
            mm = re.match(r"^(.+?)\s+\((.+)\)$", val)
            meta["model"], meta["backend"] = (mm.group(1), mm.group(2)) if mm else (val, None)
        elif key == "speakers":
            meta["speakers"] = [s.strip() for s in val.split(",") if s.strip()]
    return meta


def meeting_summary(path: Path, md: Optional[str] = None) -> Dict[str, Any]:
    from . import summarizer

    md = path.read_text() if md is None else md
    meta = _meta(md)
    return {"id": path.stem, "title": meta["title"] or path.stem,
            "started_at": meta["started_at"], "duration": meta["duration"],
            "speakers": meta["speakers"], "has_summary": bool(summarizer.extract_summary(md)),
            "path": str(path)}


def meeting_full(path: Path) -> Dict[str, Any]:
    from . import summarizer

    md = path.read_text()
    meta = _meta(md)
    summary_md = summarizer.extract_summary(md)
    return {**meeting_summary(path, md), "summary_md": summary_md,
            "sections": parse_sections(summary_md),
            "turns": parse_turns(summarizer.transcript_body(md)),
            "model": meta["model"], "backend": meta["backend"]}


# ---------------------------------------------------------------------------
# Pipeline hooks → events
# ---------------------------------------------------------------------------

# Present while a recording is running or being processed: install.sh,
# the update migration and `meetre restart` refuse to quit meetre then.
BUSY_FILE = Path(os.path.expanduser("~/.cache/meetre/busy"))


class _StayAwake:
    """Keeps the display and Mac awake while recording (``caffeinate``).

    System audio comes through ScreenCaptureKit, which needs an active
    display: with the default 2-minute display sleep, just *listening* in a
    call long enough would silently stop capturing the other side. Tied to
    the engine's pid, so it can't outlive a crashed engine."""

    def __init__(self):
        self._proc = None

    def on(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            try:
                self._proc = subprocess.Popen(
                    ["/usr/bin/caffeinate", "-d", "-i", "-w", str(os.getpid())],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._proc = None

    def off(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


_awake = _StayAwake()


def _mark_busy(on: bool, what: str = "") -> None:
    try:
        if on:
            BUSY_FILE.parent.mkdir(parents=True, exist_ok=True)
            BUSY_FILE.write_text(f"{os.getpid()} {what}\n")
        else:
            BUSY_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _log(msg: str) -> None:
    """Timestamped milestone on stderr → the app's engine.log. Cheap enough to
    call at every stage, so a hang always shows where it stopped."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


class _EngineHooks(Hooks):
    def __init__(self, engine: "Engine", meeting_id: Optional[str] = None):
        self.e = engine
        self.meeting_id = meeting_id
        self._last = 0.0

    def stage(self, text):
        _log(f"stage: {text}")
        self.e._set(stage=text, progress=None, progress_label=None)

    def progress(self, label, frac, icon="🔍"):
        if frac is None:
            self.e._set(stage=label, progress=None, progress_label=None)
        else:
            self.e._set(progress=max(0.0, min(1.0, frac)), progress_label=label)

    def download(self, label, frac):
        self.e._set(progress=frac, progress_label=f"Downloading {label}")

    def download_done(self):
        self.e._set(progress=None, progress_label=None)

    def notify(self, title, subtitle, msg):
        _log(f"notify: {subtitle} — {msg}")
        self.e.event("notify", {"title": title, "subtitle": subtitle, "body": msg})

    def summary_begin(self, model):
        self._last = 0.0
        self.e.event("summary.stream", {"meeting_id": self.meeting_id, "phase": "thinking",
                                        "text": "", "stage": "", "model": model})

    def summary_feed(self, phase, text, stage=""):
        now = time.monotonic()
        if now - self._last < STREAM_INTERVAL:
            return
        self._last = now
        self.e.event("summary.stream", {"meeting_id": self.meeting_id, "phase": phase,
                                        "text": text, "stage": stage})

    def summary_finish(self, text):
        self.e.event("summary.stream", {"meeting_id": self.meeting_id, "phase": "done",
                                        "text": text or "", "stage": ""})

    def published(self, out, stage):
        path = out["path"]
        self.meeting_id = path.stem
        self.e.event("meeting.updated", {"id": path.stem, "stage": stage,
                                         "summary": meeting_summary(path)})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, out_fd: int):
        self._out = os.fdopen(out_fd, "w", buffering=1, encoding="utf-8")
        self._wlock = threading.Lock()
        self._slock = threading.RLock()
        self.cfg = Config.load()
        # state
        self.state = "idle"
        self.stage: Optional[str] = None
        self.progress: Optional[float] = None
        self.progress_label: Optional[str] = None
        self.title: Optional[str] = None
        self._since = 0.0
        self._recorder = None
        self._rec_meta = None   # (title, started, tmp_wav, auto_title)
        self._live = None
        # chat
        self._chat_stop = False
        self._chat_busy = False
        self._mcp = None
        self._mcp_key = None
        # calls
        from .callwatch import CallWatcher

        self._calls = CallWatcher(lambda: self.event("call", {"event": "started"}),
                                  lambda: self.event("call", {"event": "ended"}))
        if self.cfg.detect_calls:
            self._calls.start()
        threading.Thread(target=self._ticker, daemon=True, name="meetre-tick").start()

    # -- wire ------------------------------------------------------------------------

    def _write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._wlock:
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except (BrokenPipeError, ValueError):
                os._exit(0)  # the app is gone

    def event(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def serve(self) -> None:
        inflight: List[threading.Thread] = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                self._write({"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32700, "message": f"Parse error: {e}"}})
                continue
            t = threading.Thread(target=self._dispatch, args=(req,), daemon=True)
            t.start()
            inflight.append(t)
            inflight[:] = [x for x in inflight if x.is_alive()]
        # stdin closed: the app quit. Let quick in-flight requests answer first.
        deadline = time.monotonic() + 3.0
        for t in inflight:
            t.join(max(0.0, deadline - time.monotonic()))
        self._shutdown_now()

    def _dispatch(self, req: dict) -> None:
        rid, method = req.get("id"), req.get("method", "")
        params = req.get("params") or {}
        handler = getattr(self, "m_" + method.replace(".", "_"), None)
        try:
            if handler is None:
                raise RPCError(f"Method not found: {method}", -32601)
            if not isinstance(params, dict):
                raise RPCError("params must be an object", -32602)
            try:
                inspect.signature(handler).bind(**params)
            except TypeError as e:
                raise RPCError(f"Invalid params for {method}: {e}", -32602)
            result = handler(**params)
            if rid is not None:
                self._write({"jsonrpc": "2.0", "id": rid, "result": result if result is not None else {}})
        except RPCError as e:
            self._error(rid, e)
        except Exception as e:  # noqa: BLE001
            print(traceback.format_exc(), file=sys.stderr)
            self._error(rid, RPCError(str(e) or type(e).__name__))

    def _error(self, rid, e: RPCError) -> None:
        if rid is None:
            self.event("error", {"message": str(e), "hint": e.hint})
            return
        err: Dict[str, Any] = {"code": e.code, "message": str(e)}
        if e.hint:
            err["data"] = {"hint": e.hint}
        self._write({"jsonrpc": "2.0", "id": rid, "error": err})

    # -- state ------------------------------------------------------------------------

    def _state(self) -> dict:
        with self._slock:
            if self.state == "recording" and self._recorder is not None:
                elapsed = self._recorder.seconds
            elif self.state == "processing":
                elapsed = time.monotonic() - self._since
            else:
                elapsed = 0.0
            # "starting" is internal (a record.start in flight).
            state = "idle" if self.state == "starting" else self.state
            return {"state": state, "elapsed": round(elapsed, 1), "stage": self.stage,
                    "progress": self.progress, "progress_label": self.progress_label,
                    "title": self.title}

    def _set(self, **kw) -> None:
        with self._slock:
            for k, v in kw.items():
                setattr(self, k, v)
        self.event("state", self._state())

    def _ticker(self) -> None:
        while True:
            time.sleep(TICK)
            if self.state in ("recording", "processing"):
                self.event("state", self._state())

    # -- hello / config / catalog ---------------------------------------------------------

    def m_hello(self, client_version: Optional[str] = None) -> dict:
        from . import bundle

        return {"engine_version": bundle._version(), "protocol": PROTOCOL,
                "python": platform.python_version(), "platform": platform.platform()}

    def m_state_get(self) -> dict:
        return self._state()

    def m_config_get(self) -> dict:
        return self._config()

    def _config(self) -> dict:
        from . import autostart

        return {**asdict(self.cfg), "start_at_login": autostart.is_enabled()}

    def m_config_set(self, **values) -> dict:
        known = set(Config.__dataclass_fields__)  # type: ignore[attr-defined]
        values.pop("start_at_login", None)  # derived; set via autostart.set
        unknown = sorted(set(values) - known)
        if unknown:
            raise RPCError(f"Unknown config key(s): {', '.join(unknown)}", -32602)
        for k, v in values.items():
            setattr(self.cfg, k, v)
        self.cfg.save()
        if "detect_calls" in values:
            if self.cfg.detect_calls:
                self._calls.start()
            else:
                self._calls.stop()
        if "live_transcript" in values and self.state == "recording":
            if self.cfg.live_transcript:
                self._start_live()
            else:
                self._stop_live()
        return self._config()

    def m_catalog_get(self) -> dict:
        from . import config, downloads, summarizer, transcriber

        asr = []
        for m in config.MODELS:
            label = ("Parakeet TDT v3" if transcriber.is_parakeet(m)
                     else "Whisper " + m.replace("-", " "))
            asr.append({"id": m, "label": label,
                        "installed": downloads.is_cached(transcriber.mlx_repo(m))})
        default = summarizer.default_model()
        summ = [{"id": a, "label": a, "note": spec.note, "size_gb": spec.size_gb,
                 "fits": fits, "installed": summarizer.is_installed(a),
                 "recommended": a == default}
                for a, spec, fits in summarizer.model_catalog()]
        return {"asr_models": asr, "summary_models": summ, "languages": LANGUAGES,
                "default_summary_model": default,
                "default_prompt": summarizer.default_prompt(self.cfg.language),
                "summaries_available": summarizer.available(),
                "memory_gb": round(summarizer.system_memory_gb(), 1)}

    # -- recording -----------------------------------------------------------------------

    def m_record_start(self, title: Optional[str] = None,
                       options: Optional[dict] = None) -> dict:
        from . import pipeline

        with self._slock:
            if self.state != "idle":
                raise RPCError(f"Can't start recording while {self.state}.")
            self.state = "starting"  # reserve against a second start
        try:
            if options:
                self.m_config_set(**options)
            name = (title or "").strip()
            started = datetime.now()
            label = name or f"Meeting {started.strftime('%Y-%m-%d %H:%M')}"
            tmp_wav = Path(tempfile.gettempdir()) / f"meetre_{started.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
            try:
                recorder = pipeline.begin_recording(self.cfg, tmp_wav)
            except Exception as e:  # noqa: BLE001
                raise RPCError(str(e), hint="Check microphone and Screen Recording "
                               "permission in System Settings ▸ Privacy & Security.")
        except BaseException:
            with self._slock:
                self.state = "idle"
            raise
        for msg in recorder.start_errors:
            self.event("notify", {"title": "meetre", "subtitle": "System audio unavailable",
                                  "body": msg})
        with self._slock:
            self._recorder = recorder
            self._rec_meta = (label, started, tmp_wav, not name)
        _mark_busy(True, f"recording {label}")
        _awake.on()
        _log(f"record.start: {label!r} sources={[getattr(s, 'role', '?') for s in recorder._sources]} "
             f"live={self.cfg.live_transcript} errors={recorder.start_errors}")
        self._set(state="recording", stage=None, progress=None, progress_label=None,
                  title=label)
        if self.cfg.live_transcript:
            try:
                self._start_live()
            except Exception:  # noqa: BLE001 — captions are optional
                print(traceback.format_exc(), file=sys.stderr)
        return {"started_at": started.isoformat()}

    def _start_live(self) -> None:
        from . import pipeline

        if self._live is not None or self._recorder is None:
            return

        def on_update(lines, tentative, status):
            self.event("live.caption", {"lines": [list(l) for l in lines],
                                        "tentative": tentative, "status": status})

        self._live = pipeline.start_live(self.cfg, self._recorder, on_update)

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
            self.event("live.caption", {"lines": [], "tentative": "", "status": "stopped"})

    def m_record_stop(self) -> dict:
        with self._slock:
            if self.state != "recording" or self._recorder is None:
                raise RPCError("Not recording.")
            recorder, (title, started, _tmp, auto_title) = self._recorder, self._rec_meta
            self._since = time.monotonic()
        _log(f"record.stop after {recorder.seconds:.0f}s")
        _awake.off()
        self._stop_live()
        self._set(state="processing", stage="Finishing recording…")
        threading.Thread(target=self._process, args=(recorder, title, started, auto_title),
                         daemon=True, name="meetre-pipeline").start()
        return {}

    def _process(self, recorder, title, started, auto_title) -> None:
        from .pipeline import Pipeline

        try:
            Pipeline(self.cfg, _EngineHooks(self)).process(
                recorder, title, started, auto_title=auto_title)
        finally:
            with self._slock:
                self._recorder = None
                self._rec_meta = None
            _mark_busy(False)
            self._set(state="idle", stage=None, progress=None, progress_label=None,
                      title=None)

    def m_record_cancel(self) -> dict:
        with self._slock:
            if self.state != "recording" or self._recorder is None:
                raise RPCError("Not recording.")
            recorder, (_t, _s, tmp_wav, _a) = self._recorder, self._rec_meta
            self._recorder = None
            self._rec_meta = None
        self._stop_live()
        _mark_busy(False)
        _awake.off()
        try:
            out = recorder.stop()
            for p in [out, tmp_wav, *getattr(recorder, "stems", {}).values()]:
                Path(p).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — discarding is best-effort
            print(traceback.format_exc(), file=sys.stderr)
        self._set(state="idle", stage=None, progress=None, progress_label=None, title=None)
        return {}

    # -- meetings -----------------------------------------------------------------------

    def _files(self) -> List[Path]:
        root = self.cfg.transcripts_path
        return sorted(root.glob("*.md"), reverse=True) if root.is_dir() else []

    def _path(self, id: str) -> Path:
        if not id or "/" in id or id.startswith("."):
            raise RPCError(f"Invalid meeting id: {id!r}", -32602)
        p = self.cfg.transcripts_path / f"{id}.md"
        if not p.exists():
            raise RPCError(f"Meeting not found: {id}", -32004)
        return p

    def m_meetings_list(self, query: Optional[str] = None) -> list:
        q = (query or "").strip().lower()
        out = []
        for p in self._files():
            try:
                md = p.read_text()
            except OSError:
                continue
            if q and q not in md.lower() and q not in p.stem.lower():
                continue
            out.append(meeting_summary(p, md))
        return out

    def m_meetings_get(self, id: str) -> dict:
        return meeting_full(self._path(id))

    def m_meetings_summarize(self, id: str) -> dict:
        from .pipeline import Pipeline, summary_error_hint

        path = self._path(id)
        with self._slock:
            if self.state != "idle":
                raise RPCError(f"Busy ({self.state}) — try again when it's done.")
            self._since = time.monotonic()
        self._set(state="processing", stage="Summarizing…", title=meeting_summary(path)["title"])

        def work():
            hooks = _EngineHooks(self, meeting_id=id)
            try:
                Pipeline(self.cfg, hooks).resummarize(path)
                self.event("meeting.updated", {"id": id, "stage": "done",
                                               "summary": meeting_summary(path)})
            except Exception as e:  # noqa: BLE001
                print(traceback.format_exc(), file=sys.stderr)
                self.event("error", {"message": "Summary failed",
                                     "hint": summary_error_hint(e)})
            finally:
                self._set(state="idle", stage=None, progress=None, progress_label=None,
                          title=None)

        threading.Thread(target=work, daemon=True, name="meetre-summarize").start()
        return {}

    def m_meetings_rename(self, id: str, title: str) -> dict:
        from .transcript import _slugify

        path = self._path(id)
        title = (title or "").strip()
        if not title:
            raise RPCError("Title can't be empty.", -32602)
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("# "):
                lines[i] = f"# {title}"
                break
        else:
            lines.insert(0, f"# {title}")
        stamp = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", id)
        new = path.with_name(f"{stamp.group(1)}_{_slugify(title)}.md" if stamp
                             else f"{_slugify(title)}.md")
        if new != path and new.exists():
            raise RPCError("A meeting with that name already exists.")
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text("\n".join(lines))
        tmp.replace(path)
        if new != path:
            path.rename(new)
        return meeting_summary(new)

    def m_meetings_delete(self, id: str) -> dict:
        path = self._path(id)
        try:
            from Foundation import NSFileManager, NSURL

            ok, _url, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
                NSURL.fileURLWithPath_(str(path)), None, None)
            if not ok:
                raise RPCError(f"Could not move to Trash: {err}")
        except ImportError:
            raise RPCError("Moving to Trash needs pyobjc (Foundation).")
        return {}

    def m_meetings_reveal(self, id: str) -> dict:
        subprocess.run(["/usr/bin/open", "-R", str(self._path(id))], check=False)
        return {}

    def m_meetings_open_note(self, id: str) -> dict:
        from . import integrations, summarizer

        md = self._path(id).read_text()
        title, when = summarizer.meeting_meta(md)
        name = integrations.note_title(title or id, when)
        try:
            integrations._osascript(
                "on run argv\n  tell application \"Notes\"\n"
                "    show (first note whose name is (item 1 of argv))\n    activate\n"
                "  end tell\nend run\n", name)
        except RuntimeError as e:
            raise RPCError("No Apple Note found for this meeting.", hint=str(e))
        return {}

    # -- chat ---------------------------------------------------------------------------

    def _toolbox(self):
        """Tools for chat. No GUI here, so anything that would ask (unapproved
        folders, MCP tools not on the allow list) is denied."""
        from .mcp import MCPManager
        from .tools import ToolBox

        servers = {k: v for k, v in (self.cfg.mcp_servers or {}).items() if v.get("enabled", True)}
        key = json.dumps(servers, sort_keys=True)
        if key != self._mcp_key:
            if self._mcp is not None:
                self._mcp.stop_all()
            self._mcp, self._mcp_key = None, key
            if servers:
                self._mcp = MCPManager(servers)
                self._mcp.start_all()
        if not (self.cfg.tool_web_search or self.cfg.tool_files or self._mcp):
            return None
        return ToolBox(self.cfg, web=self.cfg.tool_web_search, files=self.cfg.tool_files,
                       mcp=self._mcp, ask=lambda *_: -1)

    def m_chat_send(self, message: str, history: Optional[list] = None,
                    meeting_id: Optional[str] = None) -> dict:
        from . import downloads, summarizer

        if self._chat_busy:
            raise RPCError("A chat answer is still being written.")
        transcript = self._path(meeting_id).read_text() if meeting_id else ""
        turns = [{"role": m["role"], "content": m["content"]} for m in (history or [])
                 if m.get("role") in ("user", "assistant")]
        turns.append({"role": "user", "content": message})
        self._chat_busy, self._chat_stop = True, False
        last = [0.0]

        def progress(phase, text):
            now = time.monotonic()
            if now - last[0] >= STREAM_INTERVAL:
                last[0] = now
                self.event("chat.delta", {"phase": phase, "text": text})

        def on_tool(label, result):
            self.event("chat.delta", {"phase": "tool", "text": label,
                                      "done": result is not None})

        try:
            repo = summarizer.resolve_model(self.cfg.summary_model)
            if not downloads.is_cached(repo):
                downloads.ensure_model(repo, lambda f, *_: self.event(
                    "chat.delta", {"phase": "downloading", "text": f"{f:.0%}"}))
            answer = summarizer.chat(
                turns, model=self.cfg.summary_model, language=self.cfg.language,
                transcript=transcript, on_progress=progress,
                should_stop=lambda: self._chat_stop, tools=self._toolbox(), on_tool=on_tool)
        finally:
            self._chat_busy = False
        self.event("chat.delta", {"phase": "done", "text": answer})
        return {"text": answer, "stopped": self._chat_stop}

    def m_chat_stop(self) -> dict:
        self._chat_stop = True
        return {}

    # -- models / app ---------------------------------------------------------------------

    def _repo(self, kind: str, id: str) -> str:
        from . import summarizer, transcriber

        if kind == "asr":
            return transcriber.mlx_repo(id)
        if kind == "summary":
            return summarizer.resolve_model(id)
        raise RPCError(f"Unknown model kind: {kind}", -32602)

    def m_models_ensure(self, kind: str, id: str) -> dict:
        from . import downloads

        repo = self._repo(kind, id)
        try:
            downloads.ensure_model(repo, lambda f, *_: self._set(
                progress=f, progress_label=f"Downloading {id}"))
        finally:
            self._set(progress=None, progress_label=None)
        return {}

    def m_models_uninstall(self, id: str) -> dict:
        from . import config, summarizer, transcriber

        target = transcriber.mlx_repo(id) if id in config.MODELS else id
        return {"freed_bytes": summarizer.uninstall_model(target)}

    def m_autostart_set(self, enabled: bool) -> dict:
        from . import autostart

        # Launched by Meetre.app: the login item starts the app, not the
        # legacy Python menu bar.
        app = os.environ.get("MEETRE_APP")
        autostart.enable(app=app) if enabled else autostart.disable()
        return {"enabled": autostart.is_enabled()}

    def m_shutdown(self) -> dict:
        threading.Timer(0.2, self._shutdown_now).start()
        return {}

    def _shutdown_now(self) -> None:
        try:
            self._stop_live()
            if self._recorder is not None and self.state == "recording":
                self._recorder.stop()  # finalise the WAV rather than leave a helper running
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._mcp is not None:
                self._mcp.stop_all()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)


def main() -> int:
    # Keep stdout for the protocol only: everything else that writes to fd 1
    # (C extensions, tqdm, print) lands on stderr.
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        from . import crashlog

        crashlog.install()
    except Exception:  # noqa: BLE001
        pass
    engine = Engine(proto_fd)
    engine.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
