"""Minimal MCP (Model Context Protocol) client over stdio — no extra deps.

Servers are configured like Claude Desktop's ``mcpServers``:
``{"name": {"command": "npx", "args": [...], "env": {...}, "enabled": true}}``.
Only the parts meetre needs are implemented: initialize, tools/list (paged)
and tools/call, as newline-delimited JSON-RPC 2.0.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

PROTOCOL_VERSION = "2025-06-18"
CLAUDE_DESKTOP_CONFIG = (Path.home() / "Library" / "Application Support" / "Claude"
                         / "claude_desktop_config.json")

_login_path: Optional[str] = None


def _search_path() -> str:
    """A usable PATH for launching servers (npx, uvx, node, …).

    Under launchd ("start at login") PATH is empty, so ask the login shell
    (by absolute path) and add the usual install locations.
    """
    global _login_path
    if _login_path is None:
        try:
            _login_path = subprocess.run(
                ["/bin/zsh", "-lc", "printf %s \"$PATH\""], capture_output=True,
                text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            _login_path = ""
    home = Path.home()
    extra = ["/opt/homebrew/bin", "/usr/local/bin", str(home / ".local" / "bin"),
             str(home / ".cargo" / "bin"), str(home / ".bun" / "bin"),
             "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    parts = [p for p in (_login_path + ":" + os.environ.get("PATH", "")).split(":") if p]
    return ":".join(dict.fromkeys(parts + extra))


def import_claude_desktop() -> Dict[str, dict]:
    """The ``mcpServers`` from Claude Desktop's config ({} if none)."""
    try:
        data = json.loads(CLAUDE_DESKTOP_CONFIG.read_text())
    except (OSError, ValueError):
        return {}
    servers = data.get("mcpServers") or {}
    return {k: dict(v, enabled=True) for k, v in servers.items()
            if isinstance(v, dict) and v.get("command")}


class MCPError(RuntimeError):
    pass


class MCPServer:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.tools: List[dict] = []
        self.error: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._pending: Dict[int, dict] = {}
        self._lock = threading.Lock()
        self._stderr_tail: List[str] = []

    # -- lifecycle ----------------------------------------------------------

    def start(self, timeout: float = 30) -> None:
        env = dict(os.environ)
        env["PATH"] = _search_path()
        env.update({k: str(v) for k, v in (self.spec.get("env") or {}).items()})
        cmd = self.spec["command"]
        exe = shutil.which(cmd, path=env["PATH"]) or cmd
        try:
            self._proc = subprocess.Popen(
                [exe, *map(str, self.spec.get("args") or [])], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(Path.home()))
        except OSError as e:
            raise MCPError(f"can't start '{cmd}': {e.strerror or e}") from e
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._stderr_loop, daemon=True).start()
        self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "meetre", "version": "1"}}, timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools, cursor = [], None
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout)
            tools += res.get("tools") or []
            cursor = res.get("nextCursor")
            if not cursor:
                break
        self.tools = tools

    def stop(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
                p.terminate()
                p.wait(timeout=3)
            except Exception:  # noqa: BLE001
                p.kill()

    # -- JSON-RPC -----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise MCPError(self._died())
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        self._proc.stdin.flush()

    def _died(self) -> str:
        tail = " ".join(self._stderr_tail[-3:]).strip()
        return f"server '{self.name}' is not running" + (f": {tail}" if tail else "")

    def request(self, method: str, params: dict, timeout: float = 60) -> dict:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            slot = {"event": threading.Event(), "msg": None}
            self._pending[rid] = slot
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        if not slot["event"].wait(timeout):
            self._pending.pop(rid, None)
            raise MCPError(f"{self.name}: '{method}' timed out")
        msg = slot["msg"]
        if msg is None:
            raise MCPError(self._died())
        if "error" in msg:
            raise MCPError(f"{self.name}: {msg['error'].get('message', msg['error'])}")
        return msg.get("result") or {}

    def _read_loop(self) -> None:
        proc = self._proc
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # servers sometimes log to stdout
            if "id" in msg and ("result" in msg or "error" in msg):
                slot = self._pending.pop(msg["id"], None)
                if slot:
                    slot["msg"] = msg
                    slot["event"].set()
            elif "id" in msg and "method" in msg:
                # Server→client requests (sampling, roots…): politely decline.
                try:
                    self._send({"jsonrpc": "2.0", "id": msg["id"],
                                "error": {"code": -32601, "message": "not supported"}})
                except MCPError:
                    pass
        for slot in list(self._pending.values()):  # process exited
            slot["event"].set()
        self._pending.clear()

    def _stderr_loop(self) -> None:
        for line in self._proc.stderr:
            self._stderr_tail = (self._stderr_tail + [line.decode(errors="replace").strip()])[-20:]

    def call(self, tool: str, arguments: dict, timeout: float = 120) -> str:
        res = self.request("tools/call", {"name": tool, "arguments": arguments}, timeout)
        parts = []
        for c in res.get("content") or []:
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "resource":
                r = c.get("resource") or {}
                parts.append(r.get("text") or f"[resource {r.get('uri', '')}]")
            else:
                parts.append(f"[{c.get('type')} content omitted]")
        if res.get("structuredContent") and not parts:
            parts.append(json.dumps(res["structuredContent"])[:8000])
        text = "\n".join(parts).strip()
        return ("ERROR: " + text) if res.get("isError") else text


class MCPManager:
    """Starts the enabled servers (in parallel) and routes tool calls."""

    def __init__(self, servers: Dict[str, dict]):
        self.servers = {n: MCPServer(n, s) for n, s in servers.items()
                        if s.get("enabled", True) and s.get("command")}

    def start_all(self) -> None:
        def run(s):
            try:
                s.start()
            except Exception as e:  # noqa: BLE001
                s.error = str(e)
                s.stop()

        threads = [threading.Thread(target=run, args=(s,), daemon=True)
                   for s in self.servers.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(45)

    def stop_all(self) -> None:
        for s in self.servers.values():
            s.stop()
