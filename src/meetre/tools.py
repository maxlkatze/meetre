"""Tools the chat model can call: web search, file reading, MCP servers.

Every file access is gated by a folder permission. A folder the user has
"always allowed" (``cfg.tool_folders``) covers its subfolders; anything else
pops a dialog — Allow once / Always allow / Deny — on the main thread while
the generating worker waits. Secrets (SSH keys, keychains, meetre's own config
with the HF token, browser profiles …) are never readable. MCP tools ask once
per tool unless "always allowed" (``cfg.mcp_allowed_tools``).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

HOME = Path.home()

# Never readable, whatever the user allowed (prefix match on resolved paths).
_BLOCKED = [HOME / p for p in (
    ".ssh", ".gnupg", ".aws", ".config/gh", ".netrc", ".config/meetre", ".docker",
    ".kube", "Library/Keychains", "Library/Cookies", "Library/Mail", "Library/Messages",
    "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
    "Library/Safari", "Library/Application Support/com.apple.TCC",
    ".cache/huggingface/token", ".zsh_history", ".bash_history",
)]
_SECRET_NAMES = re.compile(r"(^\.env(\..*)?$|id_rsa|id_ed25519|\.pem$|\.p12$|\.key$|credentials)",
                           re.IGNORECASE)

MAX_RESULT_CHARS = 8000
TEXT_EXTS_BINARY_SNIFF = 2048


def _builtin_schemas(web: bool, files: bool) -> List[dict]:
    def fn(name, desc, props, required):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

    s = []
    if web:
        s += [
            fn("web_search", "Search the web. Returns titles, URLs and snippets.",
               {"query": {"type": "string", "description": "Search query"}}, ["query"]),
            fn("fetch_url", "Read the text of a web page.",
               {"url": {"type": "string", "description": "http(s) URL"}}, ["url"]),
        ]
    if files:
        s += [
            fn("list_files", "List the contents of a folder on this Mac "
               "(e.g. ~/Downloads, ~/Documents). The user may be asked for permission.",
               {"path": {"type": "string", "description": "Folder path, ~ allowed"}}, ["path"]),
            fn("find_files", "Find files by name pattern inside a folder (recursive).",
               {"folder": {"type": "string", "description": "Folder to search, ~ allowed"},
                "pattern": {"type": "string", "description": "Glob like *.pdf or *budget*"}},
               ["folder", "pattern"]),
            fn("read_file", "Read a text file (txt, md, csv, json, code, PDF text …). "
               "Long files are paged with 'offset'.",
               {"path": {"type": "string", "description": "File path, ~ allowed"},
                "offset": {"type": "integer", "description": "Character offset to start at"}},
               ["path"]),
        ]
    return s


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)[:48]


def ask_main_thread(title: str, message: str, buttons: List[str]) -> int:
    """Show an NSAlert on the main thread and wait; returns the button index
    (-1 when no GUI is running, e.g. the CLI — treated as "deny")."""
    try:
        from PyObjCTools import AppHelper
    except ImportError:
        return -1
    done, result = threading.Event(), {"i": -1}

    def show():
        try:
            from AppKit import NSAlert, NSApp

            NSApp.activateIgnoringOtherApps_(True)
            alert = NSAlert.alloc().init()
            alert.setMessageText_(title)
            alert.setInformativeText_(message)
            for b in buttons:
                alert.addButtonWithTitle_(b)
            result["i"] = int(alert.runModal()) - 1000  # NSAlertFirstButtonReturn
        finally:
            done.set()

    AppHelper.callAfter(show)
    return result["i"] if done.wait(600) else -1


class ToolBox:
    """The tools for one chat window session."""

    def __init__(self, cfg, web: bool = True, files: bool = True, mcp=None,
                 ask: Callable[[str, str, List[str]], int] = ask_main_thread):
        self.cfg = cfg
        self.web, self.files = web, files
        self.mcp = mcp                 # MCPManager or None
        self.ask = ask
        self._once_folders: set = set()   # "Allow once" grants for this session
        self._once_tools: set = set()
        self._mcp_routes: Dict[str, tuple] = {}

    # -- schemas ------------------------------------------------------------

    def schemas(self) -> List[dict]:
        out = _builtin_schemas(self.web, self.files)
        self._mcp_routes = {}
        if self.mcp is not None:
            for server in self.mcp.servers.values():
                for t in server.tools:
                    name = f"{_safe_name(server.name)}__{_safe_name(t['name'])}"
                    self._mcp_routes[name] = (server, t["name"])
                    out.append({"type": "function", "function": {
                        "name": name,
                        "description": (t.get("description") or "")[:600]
                        + f" (MCP server: {server.name})",
                        "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                    }})
        return out

    @staticmethod
    def label(name: str, args: dict) -> str:
        """Short human description for the chat UI ("🔎 web_search: …")."""
        icons = {"web_search": "🔎", "fetch_url": "🌐", "list_files": "📁",
                 "find_files": "📁", "read_file": "📄"}
        arg = next((str(v) for v in args.values() if v not in (None, "")), "")
        return f"{icons.get(name, '🔧')} {name.replace('__', ' › ')}  {arg[:80]}".rstrip()

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "web_search" and self.web:
                return self._web_search(str(args.get("query", "")))
            if name == "fetch_url" and self.web:
                return self._fetch(str(args.get("url", "")))
            if name == "list_files" and self.files:
                return self._list(str(args.get("path", "")))
            if name == "find_files" and self.files:
                return self._find(str(args.get("folder", "")), str(args.get("pattern", "*")))
            if name == "read_file" and self.files:
                return self._read(str(args.get("path", "")), int(args.get("offset") or 0))
            if name in self._mcp_routes:
                return self._mcp_call(name, args)
            return f"ERROR: unknown or disabled tool '{name}'"
        except PermissionError as e:
            return f"ERROR: permission denied — {e}. Do not retry; tell the user."
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    # -- web ----------------------------------------------------------------

    def _web_search(self, query: str) -> str:
        from . import websearch

        hits = websearch.search(query, 6)
        if not hits:
            return "No results."
        return "\n\n".join(f"[{i}] {h.title}\n{h.url}\n{h.snippet}"
                           for i, h in enumerate(hits, 1))

    def _fetch(self, url: str) -> str:
        from . import websearch

        if not re.match(r"^https?://", url):
            return "ERROR: only http(s) URLs can be fetched"
        return websearch.page_text(url, MAX_RESULT_CHARS) or "(page has no readable text)"

    # -- files --------------------------------------------------------------

    def _resolve(self, raw: str) -> Path:
        if not raw.strip():
            raise ValueError("empty path")
        p = Path(os.path.expanduser(raw.strip()))
        if not p.is_absolute():
            p = HOME / p
        return p.resolve()

    @staticmethod
    def _blocked(p: Path) -> bool:
        if _SECRET_NAMES.search(p.name):
            return True
        return any(p == b or b in p.parents for b in _BLOCKED)

    def _allowed_folders(self) -> List[Path]:
        return [Path(os.path.expanduser(f)).resolve() for f in (self.cfg.tool_folders or [])]

    def _check(self, path: Path, folder: Path) -> None:
        """Raise PermissionError unless ``folder`` is granted (or the user grants it)."""
        if self._blocked(path) or self._blocked(folder):
            raise PermissionError(f"{path} is a protected location")
        granted = self._allowed_folders() + list(self._once_folders)
        if any(folder == g or g in folder.parents for g in granted):
            return
        shown = str(folder).replace(str(HOME), "~", 1)
        choice = self.ask(
            "Allow K2 to read files?",
            f"The chat model wants to read files in:\n\n{shown}\n\n"
            "Contents are only used on this Mac by the local model "
            "(unless web search sends a query mentioning them).",
            ["Allow once", "Always allow", "Deny"])
        if choice == 0:
            self._once_folders.add(folder)
        elif choice == 1:
            folders = list(self.cfg.tool_folders or [])
            folders.append(str(folder).replace(str(HOME), "~", 1))
            self.cfg.tool_folders = sorted(set(folders))
            self.cfg.save()
        else:
            raise PermissionError(f"the user denied access to {shown}")

    def _list(self, raw: str) -> str:
        p = self._resolve(raw)
        if not p.is_dir():
            return f"ERROR: not a folder: {p}"
        self._check(p, p)
        rows = []
        for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))[:300]:
            if c.name.startswith("."):
                continue
            try:
                size = "" if c.is_dir() else f"  {c.stat().st_size:,} B"
            except OSError:
                size = ""
            rows.append(f"{c.name}{'/' if c.is_dir() else ''}{size}")
        return f"{p}:\n" + ("\n".join(rows) or "(empty)")

    def _find(self, raw: str, pattern: str) -> str:
        p = self._resolve(raw)
        if not p.is_dir():
            return f"ERROR: not a folder: {p}"
        self._check(p, p)
        hits, seen = [], 0
        for root, dirs, names in os.walk(p):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "Library", ".git")]
            for n in names:
                seen += 1
                if fnmatch.fnmatch(n.lower(), pattern.lower()):
                    hits.append(str(Path(root) / n))
                    if len(hits) >= 100:
                        return "\n".join(hits) + "\n(… more results truncated)"
            if seen > 50_000:
                break
        return "\n".join(hits) or "No matching files."

    def _read(self, raw: str, offset: int = 0) -> str:
        p = self._resolve(raw)
        if not p.is_file():
            return f"ERROR: not a file: {p}"
        self._check(p, p.parent)
        if p.suffix.lower() == ".pdf":
            text = _pdf_text(p)
        else:
            with open(p, "rb") as fh:
                head = fh.read(TEXT_EXTS_BINARY_SNIFF)
            if b"\0" in head:
                return f"ERROR: {p.name} is a binary file, not text"
            text = p.read_text(errors="replace")
        chunk = text[offset:offset + MAX_RESULT_CHARS]
        more = len(text) - (offset + len(chunk))
        tail = f"\n\n[… {more:,} more chars — call read_file with offset={offset + len(chunk)}]" \
            if more > 0 else ""
        return chunk + tail

    # -- MCP ----------------------------------------------------------------

    def _mcp_call(self, name: str, args: dict) -> str:
        server, tool = self._mcp_routes[name]
        key = f"{server.name}/{tool}"
        if key not in (self.cfg.mcp_allowed_tools or []) and key not in self._once_tools:
            preview = json.dumps(args, ensure_ascii=False)[:400]
            choice = self.ask(
                f"Allow K2 to use “{tool}”?",
                f"MCP server: {server.name}\n\nArguments:\n{preview}",
                ["Allow once", "Always allow", "Deny"])
            if choice == 0:
                self._once_tools.add(key)
            elif choice == 1:
                self.cfg.mcp_allowed_tools = sorted(set((self.cfg.mcp_allowed_tools or []) + [key]))
                self.cfg.save()
            else:
                raise PermissionError(f"the user denied {key}")
        return server.call(tool, args)[:MAX_RESULT_CHARS * 2]


def _pdf_text(p: Path) -> str:
    """PDF text via PDFKit (ships with macOS — no extra dependency)."""
    import objc
    from Foundation import NSURL

    objc.loadBundle("PDFKit", {}, bundle_path="/System/Library/Frameworks/PDFKit.framework")
    PDFDocument = objc.lookUpClass("PDFDocument")
    doc = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(p)))
    if doc is None:
        raise ValueError(f"can't open PDF {p.name}")
    return str(doc.string() or "")


# -- parsing model output ------------------------------------------------------

_CALL = re.compile(r"<(?:ifm\|)?tool_call>\s*(.*?)\s*</(?:ifm\|)?tool_call>", re.DOTALL)
_CALL_BLOCK = re.compile(r"<(?:ifm\|)?tool_calls>.*?(?:</(?:ifm\|)?tool_calls>|$)"
                         r"|<(?:ifm\|)?tool_call>.*?(?:</(?:ifm\|)?tool_call>|$)", re.DOTALL)


_ARG = re.compile(r"<(?:ifm\|)?arg_key>(.*?)</(?:ifm\|)?arg_key>\s*"
                  r"<(?:ifm\|)?arg_value>(.*?)</(?:ifm\|)?arg_value>", re.DOTALL)


def _arg_value(v: str):
    v = v.strip()
    try:
        return json.loads(v)  # numbers, booleans, objects
    except ValueError:
        return v


def parse_tool_calls(text: str) -> List[dict]:
    """[{name, arguments}] from model output.

    K2's native form is ``<ifm|tool_call>name <ifm|arg_key>k</…><ifm|arg_value>v</…>``;
    JSON bodies (``{"name": …, "arguments": …}``, Qwen's <tool_call>) also work.
    """
    calls = []
    for m in _CALL.finditer(text):
        body = m.group(1).strip()
        if body.startswith("{"):
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if not isinstance(obj, dict) or not obj.get("name"):
                continue
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"name": str(obj["name"]),
                          "arguments": args if isinstance(args, dict) else {}})
        else:
            name = re.split(r"\s|<", body, 1)[0].strip()
            if name:
                calls.append({"name": name,
                              "arguments": {k.strip(): _arg_value(v) for k, v in _ARG.findall(body)}})
    return calls


def strip_tool_calls(text: str) -> str:
    """Model output with any (partial) tool-call block removed, for display."""
    return _CALL_BLOCK.sub("", text).strip()


def has_tool_call_start(text: str) -> bool:
    return bool(re.search(r"<(?:ifm\|)?tool_call", text))
