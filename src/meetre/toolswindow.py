"""Tools & MCP window: what the chat model may use.

Built-in tools (web search, file reading — both on by default), the folders it
may read without asking, MCP servers (➕ popup with presets, or import from
Claude Desktop) and the MCP tools that were "always allowed".
"""

from __future__ import annotations

import shlex
import threading
from typing import Callable, Optional

W, H = 540, 620
PAD = 18

# Popular stdio MCP servers. ``{folder}`` is replaced with ~/Documents.
PRESETS = [
    ("Custom…", "", "", ""),
    ("Filesystem", "filesystem", "npx", "-y @modelcontextprotocol/server-filesystem ~/Documents"),
    ("Memory (knowledge graph)", "memory", "npx", "-y @modelcontextprotocol/server-memory"),
    ("Fetch (web pages)", "fetch", "uvx", "mcp-server-fetch"),
    ("Git", "git", "uvx", "mcp-server-git"),
    ("Sequential thinking", "thinking", "npx", "-y @modelcontextprotocol/server-sequential-thinking"),
    ("Time", "time", "uvx", "mcp-server-time"),
]

_CTRL = None


def _controller_class():
    global _CTRL
    if _CTRL is not None:
        return _CTRL
    import objc

    class _ToolsController(objc.lookUpClass("NSObject")):
        def act_(self, sender):
            self.owner.on_action(int(sender.tag()), sender)

        def windowWillClose_(self, note):
            self.owner.on_close()

    _CTRL = _ToolsController
    return _CTRL


class ToolsWindow:
    """Settings for chat tools. ``on_change()`` fires after every change."""

    # action tags
    WEB, FILES, ADD_FOLDER, IMPORT, PLUS, POP_ADD, POP_CANCEL, POP_PRESET, RESET = range(9)
    QUICK = 100      # + index into quick folders
    DEL_FOLDER = 200  # + index
    TOGGLE_MCP = 300  # + index
    DEL_MCP = 400     # + index
    TEST_MCP = 500    # + index

    QUICK_FOLDERS = ["~/Downloads", "~/Documents", "~/Desktop"]

    def __init__(self, cfg, on_change: Optional[Callable[[], None]] = None):
        self.cfg = cfg
        self.on_change = on_change or (lambda: None)
        self._status = {}   # server name -> "✓ 4 tools" / "✗ error" / "…"
        self._popover = None
        self._lock = threading.Lock()
        self._dirty = False
        self._build_window()
        self._render()
        from .chat import _common_mode_timer

        self._timer = _common_mode_timer(self._tick, 0.25)

    # -- window -----------------------------------------------------------------

    def _build_window(self) -> None:
        from AppKit import NSApp, NSAppearance, NSBackingStoreBuffered, NSMakeRect, NSWindow

        self._ctrl = _controller_class().alloc().init()
        self._ctrl.owner = self
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), 1 | 2, NSBackingStoreBuffered, False)
        win.setTitle_("meetre — Tools & MCP")
        win.setReleasedWhenClosed_(False)
        win.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        win.setDelegate_(self._ctrl)
        win.center()
        self._win = win
        NSApp.activateIgnoringOtherApps_(True)
        win.makeKeyAndOrderFront_(None)

    def show(self) -> None:
        from AppKit import NSApp

        self._render()
        NSApp.activateIgnoringOtherApps_(True)
        self._win.makeKeyAndOrderFront_(None)

    @property
    def is_open(self) -> bool:
        return self._win is not None and bool(self._win.isVisible())

    def on_close(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._win = None

    def _tick(self, _t=None) -> None:
        with self._lock:
            dirty, self._dirty = self._dirty, False
        if dirty and self._win is not None and self._popover is None:
            self._render()

    # -- layout helpers ----------------------------------------------------------

    def _render(self) -> None:
        from AppKit import NSBox, NSMakeRect, NSView

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        self._root = root
        y = H - PAD

        def section(title, sub=""):
            nonlocal y
            y -= 22
            self._label(title, PAD, y, 300, bold=True, size=13)
            if sub:
                y -= 18
                self._label(sub, PAD, y, W - 2 * PAD, size=11, alpha=0.55)
            y -= 6

        def divider():
            nonlocal y
            y -= 12
            box = NSBox.alloc().initWithFrame_(NSMakeRect(PAD, y, W - 2 * PAD, 1))
            box.setBoxType_(2)  # separator
            root.addSubview_(box)

        # Built-in tools
        section("Built-in tools", "K2 decides on its own when a question needs them.")
        y -= 24
        self._check("Web search & reading web pages (sends only search queries/URLs)",
                    PAD, y, self.cfg.tool_web_search, self.WEB)
        y -= 24
        self._check("Read files on this Mac (asks before each new folder)",
                    PAD, y, self.cfg.tool_files, self.FILES)
        divider()

        # Folders
        section("Always-allowed folders",
                "Other folders trigger an Allow once / Always allow / Deny prompt.")
        folders = list(self.cfg.tool_folders or [])
        if not folders:
            y -= 20
            self._label("None yet.", PAD, y, 300, size=12, alpha=0.5)
        for i, f in enumerate(folders[:6]):
            y -= 26
            self._label(f"📁  {f}", PAD, y + 4, W - 2 * PAD - 90, size=12)
            self._button("Remove", W - PAD - 80, y, 80, self.DEL_FOLDER + i, small=True)
        y -= 32
        x = PAD
        for i, q in enumerate(self.QUICK_FOLDERS):
            if q in folders:
                continue
            b = self._button(f"+ {q[2:]}", x, y, 96, self.QUICK + i, small=True)
            x += b.frame().size.width + 6
        self._button("Choose folder…", x, y, 120, self.ADD_FOLDER, small=True)
        divider()

        # MCP servers
        y -= 26
        self._label("MCP servers", PAD, y + 2, 200, bold=True, size=13)
        plus = self._button("+", W - PAD - 34, y - 2, 34, self.PLUS)
        plus.setToolTip_("Add an MCP server")
        self._plus = plus
        self._button("Import from Claude Desktop", W - PAD - 34 - 196, y - 2, 190,
                     self.IMPORT, small=True)
        y -= 4
        servers = list((self.cfg.mcp_servers or {}).items())
        if not servers:
            y -= 22
            self._label("No servers. Click + to add one (e.g. Filesystem, Memory, Git).",
                        PAD, y, W - 2 * PAD, size=12, alpha=0.5)
        for i, (name, spec) in enumerate(servers[:7]):
            y -= 28
            self._check(name, PAD, y + 4, spec.get("enabled", True), self.TOGGLE_MCP + i, width=150)
            cmd = " ".join([spec.get("command", "")] + [str(a) for a in spec.get("args") or []])
            status = self._status.get(name, "")
            self._label((status + "  " if status else "") + cmd, PAD + 154, y + 6,
                        W - 2 * PAD - 154 - 130, size=11, alpha=0.6)
            self._button("Test", W - PAD - 126, y, 56, self.TEST_MCP + i, small=True)
            self._button("Remove", W - PAD - 66, y, 66, self.DEL_MCP + i, small=True)
        divider()

        allowed = self.cfg.mcp_allowed_tools or []
        y -= 26
        self._label(f"{len(allowed)} MCP tool(s) always allowed", PAD, y + 4, 300, size=12, alpha=0.7)
        if allowed:
            self._button("Reset approvals", W - PAD - 130, y, 130, self.RESET, small=True)

        self._win.setContentView_(root)

    def _label(self, text, x, y, w, *, bold=False, size=12, alpha=0.9):
        from AppKit import NSColor, NSFont, NSMakeRect, NSTextField

        f = NSTextField.labelWithString_(text)
        f.setFrame_(NSMakeRect(x, y, w, size + 6))
        f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        f.setTextColor_(NSColor.colorWithWhite_alpha_(1.0, alpha))
        f.cell().setLineBreakMode_(5)  # truncate middle
        self._root.addSubview_(f)
        return f

    def _check(self, title, x, y, on, tag, width=None):
        from AppKit import NSButton, NSMakeRect

        b = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, width or W - 2 * PAD, 20))
        b.setButtonType_(3)
        b.setTitle_(title)
        b.setState_(1 if on else 0)
        b.setTag_(tag)
        b.setTarget_(self._ctrl)
        b.setAction_("act:")
        self._root.addSubview_(b)
        return b

    def _button(self, title, x, y, w, tag, small=False):
        from AppKit import NSButton, NSFont, NSMakeRect

        b = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 26 if small else 30))
        b.setTitle_(title)
        b.setBezelStyle_(1)
        if small:
            b.setFont_(NSFont.systemFontOfSize_(11))
            b.sizeToFit()
            f = b.frame()
            b.setFrame_(NSMakeRect(x, y, max(w if title in ("Remove", "Test") else 0, f.size.width), 26))
        b.setTag_(tag)
        b.setTarget_(self._ctrl)
        b.setAction_("act:")
        self._root.addSubview_(b)
        return b

    # -- actions -----------------------------------------------------------------

    def _save(self) -> None:
        self.cfg.save()
        self.on_change()
        self._render()

    def on_action(self, tag: int, sender) -> None:
        servers = list((self.cfg.mcp_servers or {}).keys())
        folders = list(self.cfg.tool_folders or [])
        if tag == self.WEB:
            self.cfg.tool_web_search = bool(sender.state())
        elif tag == self.FILES:
            self.cfg.tool_files = bool(sender.state())
        elif tag == self.ADD_FOLDER:
            self._choose_folder()
            return
        elif tag == self.IMPORT:
            self._import()
            return
        elif tag == self.PLUS:
            self._show_add_popover()
            return
        elif tag == self.POP_PRESET:
            self._apply_preset(int(sender.indexOfSelectedItem()))
            return
        elif tag == self.POP_ADD:
            self._add_from_popover()
            return
        elif tag == self.POP_CANCEL:
            self._close_popover()
            return
        elif tag == self.RESET:
            self.cfg.mcp_allowed_tools = []
        elif self.QUICK <= tag < self.QUICK + 10:
            self._add_folder(self.QUICK_FOLDERS[tag - self.QUICK])
        elif self.DEL_FOLDER <= tag < self.DEL_FOLDER + 100:
            folders.pop(tag - self.DEL_FOLDER)
            self.cfg.tool_folders = folders
        elif self.TOGGLE_MCP <= tag < self.TOGGLE_MCP + 100:
            name = servers[tag - self.TOGGLE_MCP]
            self.cfg.mcp_servers[name]["enabled"] = bool(sender.state())
        elif self.DEL_MCP <= tag < self.DEL_MCP + 100:
            name = servers[tag - self.DEL_MCP]
            self.cfg.mcp_servers.pop(name, None)
            self._status.pop(name, None)
        elif self.TEST_MCP <= tag < self.TEST_MCP + 100:
            self._test(servers[tag - self.TEST_MCP])
            return
        self._save()

    def _add_folder(self, path: str) -> None:
        self.cfg.tool_folders = sorted(set((self.cfg.tool_folders or []) + [path]))

    def _choose_folder(self) -> None:
        from AppKit import NSOpenPanel
        from pathlib import Path

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setPrompt_("Allow")
        panel.setMessage_("Choose a folder K2 may always read")
        if panel.runModal() == 1:
            p = str(panel.URL().path())
            home = str(Path.home())
            self._add_folder(p.replace(home, "~", 1) if p.startswith(home) else p)
            self._save()

    def _import(self) -> None:
        from . import mcp
        from .tools import ask_main_thread  # noqa: F401 (same alert style)

        found = mcp.import_claude_desktop()
        new = {k: v for k, v in found.items() if k not in (self.cfg.mcp_servers or {})}
        from AppKit import NSAlert

        alert = NSAlert.alloc().init()
        if not found:
            alert.setMessageText_("No Claude Desktop MCP servers found")
            alert.setInformativeText_(str(mcp.CLAUDE_DESKTOP_CONFIG))
        else:
            self.cfg.mcp_servers = {**(self.cfg.mcp_servers or {}), **new}
            alert.setMessageText_(f"Imported {len(new)} server(s)")
            alert.setInformativeText_(", ".join(new) or "All were already added.")
        alert.runModal()
        self._save()
        for name in new:
            self._test(name)

    def _test(self, name: str) -> None:
        from .mcp import MCPServer

        spec = (self.cfg.mcp_servers or {}).get(name)
        if not spec:
            return
        self._status[name] = "… connecting"
        self._render()

        def run():
            s = MCPServer(name, spec)
            try:
                s.start()
                status = f"✓ {len(s.tools)} tools"
            except Exception as e:  # noqa: BLE001
                status = f"✗ {str(e)[:60]}"
            finally:
                s.stop()
            with self._lock:
                self._status[name] = status
                self._dirty = True

        threading.Thread(target=run, daemon=True).start()

    # -- ➕ popover ---------------------------------------------------------------

    def _show_add_popover(self) -> None:
        from AppKit import (
            NSFont, NSMakeRect, NSPopover, NSPopUpButton, NSTextField, NSView,
            NSViewController,
        )

        PW, PH = 380, 262
        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, PW, PH))

        def lbl(text, y, bold=False):
            f = NSTextField.labelWithString_(text)
            f.setFrame_(NSMakeRect(16, y, PW - 32, 18))
            f.setFont_(NSFont.boldSystemFontOfSize_(13) if bold else NSFont.systemFontOfSize_(11))
            view.addSubview_(f)

        def field(y, placeholder):
            f = NSTextField.alloc().initWithFrame_(NSMakeRect(110, y, PW - 126, 24))
            f.setPlaceholderString_(placeholder)
            view.addSubview_(f)
            return f

        def row_label(text, y):
            f = NSTextField.labelWithString_(text)
            f.setFrame_(NSMakeRect(16, y + 3, 90, 18))
            f.setFont_(NSFont.systemFontOfSize_(12))
            view.addSubview_(f)

        lbl("Add MCP server", PH - 32, bold=True)
        row_label("Preset", PH - 66)
        pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(106, PH - 68, PW - 122, 26), False)
        pop.addItemsWithTitles_([p[0] for p in PRESETS])
        pop.setTag_(self.POP_PRESET)
        pop.setTarget_(self._ctrl)
        pop.setAction_("act:")
        view.addSubview_(pop)
        row_label("Name", PH - 100)
        self._f_name = field(PH - 100, "e.g. filesystem")
        row_label("Command", PH - 132)
        self._f_cmd = field(PH - 132, "npx, uvx, node, python3, /path/to/server")
        row_label("Arguments", PH - 164)
        self._f_args = field(PH - 164, "-y @modelcontextprotocol/server-memory")
        row_label("Env", PH - 196)
        self._f_env = field(PH - 196, "KEY=value, OTHER=value (optional)")
        self._f_msg = NSTextField.labelWithString_("")
        self._f_msg.setFrame_(NSMakeRect(16, 16, 170, 30))
        self._f_msg.setFont_(NSFont.systemFontOfSize_(10.5))
        view.addSubview_(self._f_msg)

        from AppKit import NSButton

        for title, x, tag, key in (("Cancel", PW - 186, self.POP_CANCEL, "\x1b"),
                                   ("Add", PW - 96, self.POP_ADD, "\r")):
            b = NSButton.alloc().initWithFrame_(NSMakeRect(x, 12, 84, 32))
            b.setTitle_(title)
            b.setBezelStyle_(1)
            b.setKeyEquivalent_(key)
            b.setTag_(tag)
            b.setTarget_(self._ctrl)
            b.setAction_("act:")
            view.addSubview_(b)

        vc = NSViewController.alloc().init()
        vc.setView_(view)
        popover = NSPopover.alloc().init()
        popover.setContentViewController_(vc)
        popover.setContentSize_(view.frame().size)
        popover.setBehavior_(1)  # transient: closes on outside click
        popover.showRelativeToRect_ofView_preferredEdge_(self._plus.bounds(), self._plus, 1)
        self._popover = popover
        self._win.makeFirstResponder_(self._f_name)

    def _apply_preset(self, i: int) -> None:
        _title, name, cmd, args = PRESETS[i]
        self._f_name.setStringValue_(name)
        self._f_cmd.setStringValue_(cmd)
        self._f_args.setStringValue_(args)

    def _close_popover(self) -> None:
        if self._popover is not None:
            self._popover.performClose_(None)
            self._popover = None

    def _add_from_popover(self) -> None:
        import os

        name = str(self._f_name.stringValue()).strip()
        cmd = str(self._f_cmd.stringValue()).strip()
        try:
            args = [os.path.expanduser(a) for a in shlex.split(str(self._f_args.stringValue()))]
        except ValueError as e:
            self._f_msg.setStringValue_(f"Arguments: {e}")
            return
        env = {}
        for part in str(self._f_env.stringValue()).replace("\n", ",").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                env[k.strip()] = v.strip()
        if not name or not cmd:
            self._f_msg.setStringValue_("Name and command are required.")
            return
        if name in (self.cfg.mcp_servers or {}):
            self._f_msg.setStringValue_(f"“{name}” already exists.")
            return
        spec = {"command": cmd, "args": args, "enabled": True}
        if env:
            spec["env"] = env
        self.cfg.mcp_servers = {**(self.cfg.mcp_servers or {}), name: spec}
        self._close_popover()
        self._save()
        self._test(name)
