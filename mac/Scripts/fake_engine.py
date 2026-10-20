"""Fake meetre engine for UI work: speaks the docs/ROADMAP.md protocol over
stdio with fixture data, simulated recording, live captions, processing
progress, summary streaming and chat. Stdlib only.

Run by the app with MEETRE_FAKE_ENGINE=1 (or --fake-engine).
Env: FAKE_CALL=1 emits a Teams "call started" event after 3 s.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta

LOCK = threading.Lock()


def send(obj):
    with LOCK:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def notify(method, params):
    send({"jsonrpc": "2.0", "method": method, "params": params})


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


NOW = datetime.now()
# Invented demo data (public screenshots / UI work) — never real meetings.
MEETINGS = [
    {"id": "roadmap", "title": "Q4 roadmap review",
     "started_at": iso(NOW.replace(hour=9, minute=30)), "duration": 1712,
     "speakers": ["Me", "Speaker 1", "Speaker 2"], "has_summary": True},
    {"id": "onboarding", "title": "Onboarding flow — design crit",
     "started_at": iso(NOW.replace(hour=11, minute=5)), "duration": 846,
     "speakers": ["Me", "Speaker 1"], "has_summary": True},
    {"id": "weekly", "title": "Weekly product sync",
     "started_at": iso((NOW - timedelta(days=1)).replace(hour=10, minute=0)), "duration": 1885,
     "speakers": ["Me", "Speaker 1", "Speaker 2"], "has_summary": True},
    {"id": "interview", "title": "Customer interview — Lumen Labs",
     "started_at": iso((NOW - timedelta(days=4)).replace(hour=14, minute=30)), "duration": 2402,
     "speakers": ["Me", "Speaker 1"], "has_summary": True},
    {"id": "hiring", "title": "Hiring debrief",
     "started_at": iso((NOW - timedelta(days=9)).replace(hour=16, minute=15)), "duration": 540,
     "speakers": ["Me", "Speaker 1", "Speaker 2"], "has_summary": False},
]
MEETINGS.sort(key=lambda m: m["started_at"], reverse=True)

SECTIONS = {
    "summary": "The Q4 plan is too wide for the team: three bets compete for the same two "
               "engineers. Offline mode moves to Q1 so search and the new onboarding can ship "
               "before the December freeze.",
    "decisions": [
        "Offline mode moves to Q1 — it needs the sync rewrite first.",
        "Search ships behind a flag on Nov 14, then rolls out to 100% after a week.",
        "Onboarding v2 replaces the tour; the old tour is removed, not hidden.",
    ],
    "tasks": [
        {"text": "Write the Q1 brief for offline mode", "owner": "Me", "due": "Friday"},
        {"text": "Set up the search rollout flag and dashboard", "owner": "Speaker 2"},
        {"text": "Share onboarding v2 prototype with support", "owner": "Speaker 1", "due": "Nov 7"},
        {"text": "Book the December freeze review", "owner": None},
    ],
    "questions": ["Do we need legal review for the new data export before the freeze?"],
}
SUMMARY_MD = (
    "## Summary\n" + SECTIONS["summary"] + "\n\n## Decisions\n"
    + "\n".join("- " + d for d in SECTIONS["decisions"]) + "\n\n## Action items\n"
    + "\n".join(f"- **{t['owner'] or '–'}:** {t['text']}" for t in SECTIONS["tasks"])
    + "\n\n## Open questions\n" + "\n".join("- " + q for q in SECTIONS["questions"])
)
TURNS = [
    {"speaker": "Me", "start": 0, "text": "Let's be honest about Q4: we have three big bets and two engineers."},
    {"speaker": "Me", "start": 9, "text": "Something has to move, and I think it's offline mode."},
    {"speaker": "Speaker 1", "start": 21, "text": "Agreed. Offline needs the sync rewrite anyway, that's a Q1 thing."},
    {"speaker": "Speaker 2", "start": 34, "text": "Then search can go out on the 14th behind a flag. I'd watch it for a week."},
    {"speaker": "Me", "start": 46, "text": "Good. And onboarding v2 replaces the tour completely?"},
    {"speaker": "Speaker 1", "start": 52, "text": "Yes. I'll share the prototype with support by the 7th."},
    {"speaker": "Speaker 2", "start": 61, "text": "Open point: does the new data export need a legal review before the freeze?"},
    {"speaker": "Me", "start": 70, "text": "Let's find out this week. I'll write the offline brief by Friday."},
]
CAPTIONS = [
    ("Me", "Okay, let's start — I sent the agenda this morning."),
    ("Speaker", "Got it. Can we begin with search? The flag is almost ready."),
    ("Me", "Sure. Then onboarding, and offline mode last."),
    ("Speaker", "Offline will need the sync rewrite first, I'm pretty sure."),
    ("Me", "Right. Let's decide that today and keep Q4 realistic."),
]

CONFIG = {
    "model": "parakeet-tdt-v3", "language": "en", "person_detection": True,
    "capture_system": True, "num_speakers": None, "summary_model": "k2-horizon-7b",
    "auto_summarize": True, "auto_notes": True, "summary_prompt": "", "detect_calls": True,
    "summary_overlay": True, "live_transcript": True, "live_transcript_expanded": os.environ.get("FAKE_MINIMIZED") != "1",
    "summary_overlay_expanded": True,
    "name_speakers": True, "my_name": "", "vad": True, "word_timestamps": True,
    "hf_token": "", "transcripts_dir": os.path.expanduser("~/Documents/meetre/transcripts"),
}
CATALOG = {
    "asr_models": [
        {"id": "parakeet-tdt-v3", "label": "Parakeet TDT v3", "installed": True},
        {"id": "large-v3-turbo", "label": "Whisper large-v3 turbo", "installed": False},
        {"id": "small", "label": "Whisper small", "installed": False},
    ],
    "summary_models": [
        {"id": "k2-horizon-7b", "label": "K2 Horizon 7B", "note": "reasoning — excellent summary writing",
         "size_gb": 7.2, "fits": True, "installed": True, "recommended": True},
        {"id": "qwen3.5-35b", "label": "Qwen3.5 35B", "note": "MoE, ~3B active", "size_gb": 19.6,
         "fits": True, "installed": False, "recommended": False},
        {"id": "qwen3.5-397b", "label": "Qwen3.5 397B", "note": "Mac Studio Ultra only", "size_gb": 210,
         "fits": False, "installed": False, "recommended": False},
    ],
    "languages": [{"code": "de", "label": "Deutsch"}, {"code": "en", "label": "English"},
                  {"code": None, "label": "Auto-detect"}],
    "default_prompt": "Fasse das Meeting oben für jemanden zusammen, der nicht dabei war. …",
}

STATE = {"state": "idle", "elapsed": 0.0, "stage": None, "progress": None,
         "progress_label": None, "title": None}
REC = {"t0": None, "title": ""}
CHAT_STOP = threading.Event()


def emit_state():
    notify("state", dict(STATE))


def recorder_loop():
    i = 0
    while True:
        time.sleep(0.5)
        if STATE["state"] == "recording":
            STATE["elapsed"] = time.monotonic() - REC["t0"]
            emit_state()
            i += 1
            n = min(len(CAPTIONS), i // 6)
            tentative = CAPTIONS[n][1][: (i % 6) * 12] if n < len(CAPTIONS) else ""
            notify("live.caption", {"lines": [list(c) for c in CAPTIONS[:n]],
                                    "tentative": tentative, "status": "listening"})
        else:
            i = 0


def process(title):
    stages = [("Loading transcription model…", None), ("Transcribing", 0.0)]
    STATE.update(state="processing", title=title or "Meeting", progress=None)
    for label, _ in stages[:1]:
        STATE.update(stage=label, progress=None, progress_label=None)
        emit_state()
        time.sleep(1)
    for p in range(0, 101, 10):
        STATE.update(stage="Transcribing", progress=p / 100, progress_label="Transcribing")
        emit_state()
        time.sleep(0.25)
    mid = "new-" + str(int(time.time()))
    m = {"id": mid, "title": title or "Design review", "started_at": iso(datetime.now()),
         "duration": STATE["elapsed"], "speakers": ["Me", "Speaker"], "has_summary": False}
    MEETINGS.insert(0, m)
    notify("meeting.updated", {"id": mid, "stage": "transcript", "summary": m})
    STATE.update(stage="Detecting speakers…", progress=None, progress_label=None)
    emit_state()
    time.sleep(1.5)
    STATE.update(stage="Summarizing…")
    emit_state()
    stream_summary(mid)
    m["has_summary"] = True
    notify("meeting.updated", {"id": mid, "stage": "done", "summary": m})
    STATE.update(state="idle", stage=None, progress=None, title=None, elapsed=0.0)
    emit_state()
    notify("notify", {"title": "meetre", "subtitle": f"✓ {m['title']} — done", "body": "Summary + transcript ready"})


def stream_summary(mid):
    notify("summary.stream", {"meeting_id": mid, "phase": "thinking", "text": "", "stage": ""})
    time.sleep(1.2)
    text = ""
    for word in SUMMARY_MD.split(" "):
        text += word + " "
        notify("summary.stream", {"meeting_id": mid, "phase": "writing", "text": text, "stage": ""})
        time.sleep(0.03)
    notify("summary.stream", {"meeting_id": mid, "phase": "done", "text": SUMMARY_MD, "stage": ""})


def full(mid):
    m = next((x for x in MEETINGS if x["id"] == mid), MEETINGS[0])
    has = m.get("has_summary")
    return {**m, "summary_md": SUMMARY_MD if has else "", "sections": SECTIONS if has else None,
            "turns": TURNS, "model": "parakeet-tdt-v3", "backend": "parakeet-tdt-v3", "path": None}


def handle(method, p):
    if method == "hello":
        return {"engine_version": "fake-0.1", "protocol": 1, "python": sys.version.split()[0], "platform": "darwin"}
    if method == "state.get":
        return dict(STATE)
    if method == "config.get":
        return CONFIG
    if method == "config.set":
        CONFIG.update(p)
        return CONFIG
    if method == "catalog.get":
        return CATALOG
    if method == "record.start":
        REC.update(t0=time.monotonic(), title=p.get("title") or "")
        STATE.update(state="recording", elapsed=0.0, stage=None, progress=None, title=REC["title"])
        emit_state()
        return {"started_at": iso(datetime.now())}
    if method == "record.stop":
        threading.Thread(target=process, args=(REC["title"],), daemon=True).start()
        return {}
    if method == "record.cancel":
        STATE.update(state="idle", elapsed=0.0)
        emit_state()
        return {}
    if method == "meetings.list":
        q = (p.get("query") or "").lower()
        return [m for m in MEETINGS if q in m["title"].lower()]
    if method == "meetings.get":
        return full(p["id"])
    if method == "meetings.summarize":
        threading.Thread(target=lambda: (stream_summary(p["id"]),
                                         notify("meeting.updated", {"id": p["id"], "stage": "done",
                                                                    "summary": next(m for m in MEETINGS if m["id"] == p["id"])})),
                         daemon=True).start()
        return {}
    if method == "meetings.retranscribe":
        notify("meeting.updated", {"id": p["id"], "stage": "done",
                                   "summary": next(m for m in MEETINGS if m["id"] == p["id"])})
        return {}
    if method == "meetings.rename":
        m = next(x for x in MEETINGS if x["id"] == p["id"])
        m["title"] = p["title"]
        return m
    if method == "meetings.delete":
        MEETINGS[:] = [m for m in MEETINGS if m["id"] != p["id"]]
        return {}
    if method in ("meetings.reveal", "meetings.open_note", "models.ensure", "chat.stop"):
        if method == "chat.stop":
            CHAT_STOP.set()
        return {}
    if method == "autostart.set":
        return {"enabled": p.get("enabled", False)}
    if method == "chat.send":
        CHAT_STOP.clear()
        answer = ("You decided to **move offline mode to Q1** and ship **search behind a flag on "
                  "Nov 14**; onboarding v2 replaces the tour.\n\n"
                  "- Offline brief — **Me**, by Friday\n- Rollout flag + dashboard — **Speaker 2**")
        text = ""
        time.sleep(0.6)
        for w in answer.split(" "):
            if CHAT_STOP.is_set():
                break
            text += w + " "
            notify("chat.delta", {"text": text})
            time.sleep(0.04)
        return {"text": text.strip()}
    if method == "shutdown":
        threading.Timer(0.1, lambda: os._exit(0)).start()
        return {}
    raise KeyError(method)


def serve(msg):
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    try:
        result = handle(method, params)
        if mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": result})
    except KeyError:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown method {method}"}})
    except Exception as e:  # noqa: BLE001
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": str(e)}})


def main():
    threading.Thread(target=recorder_loop, daemon=True).start()
    if os.environ.get("FAKE_CALL") == "1":
        threading.Timer(3, lambda: notify("call", {"event": "started"})).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        # Long calls (chat) must not block the reader.
        threading.Thread(target=serve, args=(msg,), daemon=True).start()


if __name__ == "__main__":
    main()
