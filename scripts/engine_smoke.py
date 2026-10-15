"""Smoke-test the JSON-RPC engine: spawn it, call the read-only methods, print.

    .venv/bin/python scripts/engine_smoke.py

Exits non-zero on the first error response. Doesn't record, summarize or chat.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-m", "meetre", "engine"], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        self.n = 0
        self.events = []

    def call(self, method, **params):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n,
                                       "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit(f"engine exited during {method}")
            msg = json.loads(line)  # every stdout line must be valid JSON
            if "id" not in msg:
                self.events.append(msg["method"])
                continue
            if msg["id"] != self.n:
                continue
            if "error" in msg:
                raise SystemExit(f"{method} failed: {msg['error']}")
            return msg["result"]


def show(label, value, limit=300):
    text = json.dumps(value, ensure_ascii=False)
    print(f"✓ {label:14} {text[:limit]}{'…' if len(text) > limit else ''}")


def main():
    c = Client()
    t0 = time.monotonic()
    show("hello", c.call("hello", client_version="smoke"))
    show("state.get", c.call("state.get"))
    cfg = c.call("config.get")
    show("config.get", {k: cfg[k] for k in ("model", "language", "summary_model",
                                             "live_transcript", "detect_calls")})
    show("config.set", c.call("config.set", language=cfg["language"])["language"])  # no-op write
    cat = c.call("catalog.get")
    show("catalog.get", {"asr": [m["id"] for m in cat["asr_models"]],
                         "summary": [(m["id"], m["fits"], m["installed"])
                                     for m in cat["summary_models"]][:4],
                         "default": cat["default_summary_model"]})
    meetings = c.call("meetings.list")
    show("meetings.list", {"count": len(meetings), "newest": meetings[0] if meetings else None})
    if meetings:
        newest = next((m for m in meetings if m["has_summary"]), meetings[0])
        m = c.call("meetings.get", id=newest["id"])
        show("meetings.get", {"title": m["title"], "duration": m["duration"],
                              "sections": m["sections"], "turns": len(m["turns"]),
                              "first_turn": m["turns"][0] if m["turns"] else None}, 900)
        show("search", len(c.call("meetings.list", query=newest["title"].split()[0])))
    try:
        c.call("meetings.get", id="does-not-exist")
        raise SystemExit("expected an error for a missing meeting")
    except SystemExit as e:
        if "Meeting not found" not in str(e):
            raise
        print("✓ missing id     → error as expected")
    c.call("shutdown")
    print(f"all good in {time.monotonic() - t0:.1f}s; events seen: {sorted(set(c.events)) or 'none'}")


if __name__ == "__main__":
    main()
