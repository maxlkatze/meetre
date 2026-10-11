"""Chat window: talk to the local summary model (K2 Horizon by default).

Pick a meeting from the dropdown to chat *about* it (the transcript becomes the
model's context), or "No meeting" for a free chat. Quick-prompt chips cover the
common asks. "Tools…" opens the Tools & MCP window: web search and file reading
(on by default, folder access asks first) plus MCP servers.

Thread model mirrors :mod:`meetre.overlay`: generation runs on a worker thread
that only stores state; a main-thread timer (:meth:`ChatWindow.tick`) does all
the AppKit work.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

W, H = 560, 640
PAD = 14

QUICK_PROMPTS = {
    "en": [("Summary", "Summarize this meeting in a few bullet points."),
           ("Action items", "List all action items as a checklist: who does what, by when."),
           ("Decisions", "Which decisions were made? One bullet each."),
           ("Follow-up email", "Draft a short follow-up email to the participants."),
           ("Who said what", "For each speaker, summarize their main points.")],
    "de": [("Zusammenfassung", "Fasse das Meeting in wenigen Stichpunkten zusammen."),
           ("Aufgaben", "Liste alle Aufgaben als Checkliste: wer macht was bis wann."),
           ("Entscheidungen", "Welche Entscheidungen wurden getroffen? Je ein Stichpunkt."),
           ("Follow-up-Mail", "Schreibe eine kurze Follow-up-E-Mail an die Teilnehmer."),
           ("Wer sagte was", "Fasse für jeden Sprecher seine wichtigsten Punkte zusammen.")],
}

_CTRL = None


def _controller_class():
    """AppKit action target (defined once; Obj-C classes can't be redefined)."""
    global _CTRL
    if _CTRL is not None:
        return _CTRL
    import objc

    class _ChatController(objc.lookUpClass("NSObject")):
        def send_(self, sender):
            self.owner.on_send()

        def stop_(self, sender):
            self.owner.on_stop()

        def newChat_(self, sender):
            self.owner.on_new_chat()

        def pickMeeting_(self, sender):
            self.owner.on_pick(int(sender.indexOfSelectedItem()))

        def chip_(self, sender):
            self.owner.on_chip(int(sender.tag()))

        def openTools_(self, sender):
            self.owner.on_open_tools()

        def copyLast_(self, sender):
            self.owner.on_copy_last()

        def windowWillClose_(self, note):
            self.owner.on_close()

    _CTRL = _ChatController
    return _CTRL


def _ensure_edit_menu() -> None:
    """Give the (menu-bar-only) app an Edit menu so ⌘C/⌘V/⌘A work in fields."""
    from AppKit import NSApp, NSMenu, NSMenuItem

    main = NSApp.mainMenu()
    if main is not None and main.itemWithTitle_("Edit") is not None:
        return
    if main is None:
        main = NSMenu.alloc().init()
        main.addItem_(NSMenuItem.alloc().init())  # app menu placeholder
        NSApp.setMainMenu_(main)
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, sel, key in [("Undo", "undo:", "z"), ("Redo", "redo:", "Z"),
                            ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                            ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]:
        edit.addItemWithTitle_action_keyEquivalent_(title, sel, key)
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
    item.setSubmenu_(edit)
    main.addItem_(item)


class ChatWindow:
    def __init__(self, cfg, model_label: str = ""):
        self.cfg = cfg
        self.model_label = model_label
        self.lang = "de" if cfg.language == "de" else "en"
        self._lock = threading.Lock()
        self.history: List[dict] = []      # [{role, content}] sent to the model
        self.steps: List[list] = []        # tool-call labels per assistant turn
        self._live_steps: List[str] = []   # … for the turn being generated
        self._mcp = None                   # running MCPManager
        self._mcp_key = None               # servers config it was started with
        self._tools_win = None
        self._live = None                  # (phase, text) while generating
        self._status = ""
        self._busy = False
        self._stop = False
        self._dirty = True
        self._files: List[Path] = []
        self._meeting: Optional[Path] = None
        self._window = None
        self._frame = 0
        self._build()

    # -- window ---------------------------------------------------------------

    def _build(self) -> None:
        from AppKit import (
            NSApp, NSAppearance, NSBackingStoreBuffered, NSButton, NSFont,
            NSMakeRect, NSPopUpButton, NSScrollView, NSTextField, NSTextView,
            NSWindow,
        )

        _ensure_edit_menu()
        ctrl = _controller_class().alloc().init()
        ctrl.owner = self
        self._ctrl = ctrl

        style = 1 | 2 | 4 | 8  # titled | closable | miniaturizable | resizable
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, NSBackingStoreBuffered, False)
        win.setTitle_(f"meetre — Chat with {self.model_label or 'K2'}")
        win.setReleasedWhenClosed_(False)
        win.setMinSize_((420, 420))
        win.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        win.setDelegate_(ctrl)
        win.center()
        root = win.contentView()

        # Autoresizing: 2 = width, 16 = height, 8 = keep top (flexible bottom margin).
        top = H - PAD - 26
        lbl = NSTextField.labelWithString_("Meeting")
        lbl.setFrame_(NSMakeRect(PAD, top + 4, 60, 18))
        lbl.setAutoresizingMask_(8)
        root.addSubview_(lbl)

        pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(PAD + 60, top, W - 310, 26), False)
        pop.setTarget_(ctrl)
        pop.setAction_("pickMeeting:")
        pop.setAutoresizingMask_(2 | 8)
        root.addSubview_(pop)
        self._pop = pop

        tools_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 232, top - 2, 112, 30))
        tools_btn.setTitle_("🔧 Tools…")
        tools_btn.setBezelStyle_(1)
        tools_btn.setToolTip_("Web search, file access and MCP servers")
        tools_btn.setTarget_(ctrl)
        tools_btn.setAction_("openTools:")
        tools_btn.setAutoresizingMask_(1 | 8)  # flexible left margin
        root.addSubview_(tools_btn)

        new = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 104, top - 2, 104, 30))
        new.setTitle_("New chat")
        new.setBezelStyle_(1)
        new.setTarget_(ctrl)
        new.setAction_("newChat:")
        new.setAutoresizingMask_(1 | 8)
        root.addSubview_(new)

        # Conversation view.
        conv_bottom = 118
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(PAD, conv_bottom, W - 2 * PAD, top - 10 - conv_bottom))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)
        scroll.setAutoresizingMask_(2 | 16)
        tv = NSTextView.alloc().initWithFrame_(scroll.contentView().bounds())
        tv.setEditable_(False)
        tv.setSelectable_(True)
        tv.setDrawsBackground_(False)
        tv.setRichText_(True)
        tv.setTextContainerInset_((4, 8))
        tv.setAutoresizingMask_(2)
        tv.setVerticallyResizable_(True)
        tv.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(tv)
        root.addSubview_(scroll)
        self._tv = tv

        # Quick-prompt chips.
        x = PAD
        self._chips = []
        for i, (title, _prompt) in enumerate(QUICK_PROMPTS[self.lang]):
            b = NSButton.alloc().initWithFrame_(NSMakeRect(x, 80, 10, 26))
            b.setTitle_(title)
            b.setBezelStyle_(15)  # inline / pill
            b.setFont_(NSFont.systemFontOfSize_(11))
            b.sizeToFit()
            f = b.frame()
            b.setFrame_(NSMakeRect(x, 84, f.size.width + 8, 22))
            b.setTag_(i)
            b.setTarget_(ctrl)
            b.setAction_("chip:")
            root.addSubview_(b)
            self._chips.append(b)
            x += f.size.width + 14

        # Input row.
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(PAD, 40, W - 2 * PAD - 92, 32))
        field.setPlaceholderString_("Ask K2 anything — ↩ to send")
        field.setFont_(NSFont.systemFontOfSize_(13))
        field.setTarget_(ctrl)
        field.setAction_("send:")
        field.setAutoresizingMask_(2)
        field.cell().setScrollable_(True)
        root.addSubview_(field)
        self._field = field

        send = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 88, 38, 88, 36))
        send.setTitle_("Send")
        send.setBezelStyle_(1)
        send.setTarget_(ctrl)
        send.setAction_("send:")
        send.setAutoresizingMask_(1)
        root.addSubview_(send)
        self._send_btn = send

        status = NSTextField.labelWithString_("")
        status.setFrame_(NSMakeRect(PAD, 12, W - 2 * PAD - 120, 16))
        status.setFont_(NSFont.systemFontOfSize_(10.5))
        status.setTextColor_(_white(0.5))
        status.setAutoresizingMask_(2)
        root.addSubview_(status)
        self._status_lbl = status

        copy = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 116, 6, 116, 26))
        copy.setTitle_("Copy last answer")
        copy.setBezelStyle_(15)
        copy.setFont_(NSFont.systemFontOfSize_(11))
        copy.setTarget_(ctrl)
        copy.setAction_("copyLast:")
        copy.setAutoresizingMask_(1)
        root.addSubview_(copy)

        self._window = win
        self._fill_meetings()
        self._timer = _common_mode_timer(self.tick, 1 / 15)
        NSApp.activateIgnoringOtherApps_(True)
        win.makeKeyAndOrderFront_(None)
        win.makeFirstResponder_(field)

    def show(self) -> None:
        from AppKit import NSApp

        self._fill_meetings(keep=True)
        NSApp.activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)

    @property
    def is_open(self) -> bool:
        return self._window is not None and bool(self._window.isVisible())

    def _fill_meetings(self, keep: bool = False) -> None:
        from . import summarizer

        current = self._meeting
        self._files = sorted(self.cfg.transcripts_path.glob("*.md"), reverse=True)[:60]
        titles = ["No meeting — free chat"]
        for p in self._files:
            try:
                title, date = summarizer.meeting_meta(p.read_text()[:2000])
            except OSError:
                title, date = None, None
            titles.append(f"{date or ''}  {title or p.stem}".strip())
        self._pop.removeAllItems()
        self._pop.addItemsWithTitles_(titles)
        if keep and current in self._files:
            idx = self._files.index(current) + 1
        else:
            idx = 1 if self._files else 0  # default: the latest meeting
        self._pop.selectItemAtIndex_(idx)
        self._meeting = self._files[idx - 1] if idx else None
        self._update_chips()
        self._dirty = True

    def _update_chips(self) -> None:
        for b in self._chips:
            b.setEnabled_(self._meeting is not None)

    # -- actions (main thread) -------------------------------------------------

    def on_pick(self, idx: int) -> None:
        new = self._files[idx - 1] if idx else None
        if new != self._meeting:
            self._meeting = new
            self.on_new_chat()  # a different meeting starts a fresh conversation
        self._update_chips()

    def on_open_tools(self) -> None:
        if self._tools_win is not None and self._tools_win.is_open:
            self._tools_win.show()
            return
        from .toolswindow import ToolsWindow

        self._tools_win = ToolsWindow(self.cfg, on_change=lambda: self._set_status(
            "Tools updated — used from the next question"))

    def on_new_chat(self) -> None:
        if self._busy:
            self._stop = True
        with self._lock:
            self.history, self.steps, self._live, self._live_steps = [], [], None, []
            self._status = ""
            self._dirty = True

    def on_chip(self, i: int) -> None:
        self._field.setStringValue_(QUICK_PROMPTS[self.lang][i][1])
        self.on_send()

    def on_stop(self) -> None:
        self._stop = True

    def on_send(self) -> None:
        if self._busy:
            self.on_stop()
            return
        text = str(self._field.stringValue()).strip()
        if not text:
            return
        self._field.setStringValue_("")
        with self._lock:
            self.history.append({"role": "user", "content": text})
            self._live, self._live_steps = ("thinking", ""), []
            self._busy, self._stop, self._dirty = True, False, True
        self._send_btn.setTitle_("Stop")
        threading.Thread(target=self._answer, args=(text, self._meeting),
                         daemon=True).start()

    def on_copy_last(self) -> None:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        last = next((m["content"] for m in reversed(self.history)
                     if m["role"] == "assistant"), "")
        if last:
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(last, NSPasteboardTypeString)
            self._set_status("Copied to clipboard")

    def on_close(self) -> None:
        self._stop = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._window = None

        def release():
            while self._busy:
                time.sleep(0.2)
            from . import summarizer

            summarizer.release_chat_model()
            if self._mcp is not None:
                self._mcp.stop_all()
                self._mcp = None

        threading.Thread(target=release, daemon=True).start()

    # -- worker ----------------------------------------------------------------

    def _set_status(self, s: str) -> None:
        with self._lock:
            self._status, self._dirty = s, True

    def _toolbox(self):
        """Tools for this question; (re)starts MCP servers when their config changed."""
        import json

        from .mcp import MCPManager
        from .tools import ToolBox

        servers = {k: v for k, v in (self.cfg.mcp_servers or {}).items() if v.get("enabled", True)}
        key = json.dumps(servers, sort_keys=True)
        if key != self._mcp_key:
            if self._mcp is not None:
                self._mcp.stop_all()
            self._mcp, self._mcp_key = None, key
            if servers:
                self._set_status(f"Starting {len(servers)} MCP server(s)…")
                self._mcp = MCPManager(servers)
                self._mcp.start_all()
                failed = [s.name for s in self._mcp.servers.values() if s.error]
                if failed:
                    self._set_status(f"MCP failed: {', '.join(failed)} (see Tools…)")
        if not (self.cfg.tool_web_search or self.cfg.tool_files or self._mcp):
            return None
        return ToolBox(self.cfg, web=self.cfg.tool_web_search,
                       files=self.cfg.tool_files, mcp=self._mcp)

    def _answer(self, question: str, meeting: Optional[Path]) -> None:
        from . import downloads, summarizer

        answer = ""
        started = time.monotonic()
        try:
            transcript = meeting.read_text() if meeting else ""
            toolbox = self._toolbox()
            repo = summarizer.resolve_model(self.cfg.summary_model)
            if not downloads.is_cached(repo):
                self._set_status(f"⬇ Downloading {self.model_label}…")
                downloads.ensure_model(repo, lambda f, *_: self._set_status(
                    f"⬇ Downloading {self.model_label}… {f:.0%}"))
            self._set_status(f"Loading {self.model_label}…")
            with self._lock:
                history = [dict(m) for m in self.history]

            def progress(phase, text):
                with self._lock:
                    self._live, self._dirty = (phase, text), True
                    secs = time.monotonic() - started
                    self._status = f"{self.model_label} · {phase}… {secs:.0f}s"

            def on_tool(label, result):
                with self._lock:
                    if result is None:
                        self._live_steps.append(label)
                        self._live = ("thinking", "")
                        self._status = f"{label[:60]} …"
                    elif result.startswith("ERROR"):
                        self._live_steps[-1] = f"{label}  ⚠︎ {result[7:80]}"
                    self._dirty = True

            answer = summarizer.chat(
                history, model=self.cfg.summary_model, language=self.cfg.language,
                transcript=transcript, on_progress=progress,
                should_stop=lambda: self._stop, tools=toolbox, on_tool=on_tool)
            if not answer and self._stop:
                answer = "_(stopped)_"
            elif not answer:
                answer = "_(no answer — try rephrasing)_"
            secs = time.monotonic() - started
            used = len(self._live_steps)
            self._set_status(f"{self.model_label} · {secs:.0f}s · "
                             + (f"{used} tool call(s)" if used else "on-device"))
        except Exception as e:  # noqa: BLE001
            answer = f"_Error: {e}_"
            self._set_status("Error")
        finally:
            with self._lock:
                # A "New chat" during generation already cleared the history.
                if self.history and self.history[-1]["role"] == "user":
                    self.history.append({"role": "assistant", "content": answer})
                    self.steps.append(list(self._live_steps))
                self._live, self._live_steps = None, []
                self._busy, self._dirty = False, True

    # -- rendering (main thread) ------------------------------------------------

    def tick(self, _timer=None) -> None:
        try:
            self._tick()
        except Exception:  # noqa: BLE001 — never let the window break the app
            pass

    def _tick(self) -> None:
        if self._window is None:
            return
        self._frame += 1
        with self._lock:
            dirty, self._dirty = self._dirty, False
            history = list(self.history)
            steps = list(self.steps)
            live_steps = list(self._live_steps)
            live, status, busy = self._live, self._status, self._busy
        if not dirty and not (busy and self._frame % 8 == 0):
            return
        self._send_btn.setTitle_("Stop" if busy else "Send")
        self._status_lbl.setStringValue_(status or f"{self.model_label} · on-device")
        self._render(history, steps, live, live_steps)

    def _render(self, history, steps, live, live_steps) -> None:
        from AppKit import (
            NSFontAttributeName, NSForegroundColorAttributeName, NSLinkAttributeName,
            NSMutableAttributedString, NSMutableParagraphStyle,
            NSParagraphStyleAttributeName, NSURL,
        )

        out = NSMutableAttributedString.alloc().init()
        para = NSMutableParagraphStyle.alloc().init()
        para.setLineSpacing_(2.5)
        para.setParagraphSpacing_(4)
        bullet = para.mutableCopy()
        bullet.setHeadIndent_(14)

        def add(s, font, color, style=para, link=None):
            attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color,
                     NSParagraphStyleAttributeName: style}
            if link:
                attrs[NSLinkAttributeName] = NSURL.URLWithString_(link)
            out.appendAttributedString_(type(out).alloc().initWithString_attributes_(s, attrs))

        def markdown(text, color):
            for line in text.strip("\n").split("\n"):
                st = line.replace("**", "").strip()
                if st.startswith("#"):
                    add(st.lstrip("#").strip() + "\n", _sans(13, 0.5), _white(0.95))
                elif st[:2] in ("- ", "* ") or st[:6] in ("- [ ] ", "- [x] "):
                    add("•  ", _sans(13), _white(0.5), bullet)
                    add(st[2:] + "\n", _sans(13), color, bullet)
                else:
                    add(line.replace("**", "") + "\n", _sans(13), color)

        if not history:
            hint = ("Ask about the selected meeting — or pick “No meeting” to chat "
                    "freely. K2 runs on this Mac and can search the web, read files "
                    "you allow and use MCP servers (🔧 Tools…).")
            add(hint + "\n", _sans(12.5), _white(0.45))

        def tool_steps(labels):
            for lab in labels:
                add(f"   {lab}\n", _sans(11), _white(0.45))

        answer_i = 0
        for m in history:
            if m["role"] == "user":
                add("You\n", _sans(10.5, 0.5), _white(0.45))
                add(m["content"] + "\n\n", _sans(13, 0.3), _white(0.95))
            else:
                add(f"{self.model_label}\n", _sans(10.5, 0.5), _accent())
                tool_steps(steps[answer_i] if answer_i < len(steps) else [])
                markdown(m["content"], _white(0.86))
                add("\n", _sans(6), _white(0.5))
                answer_i += 1

        if live is not None:
            phase, text = live
            add(f"{self.model_label}\n", _sans(10.5, 0.5), _accent())
            tool_steps(live_steps)
            dot = "●" if (self._frame // 8) % 2 == 0 else "○"
            if phase == "thinking":
                tail = text.strip()[-600:]
                add(f"{dot} thinking…\n", _sans(11), _white(0.5))
                if tail:
                    add(tail + "\n", _sans(11), _white(0.35))
            else:
                markdown(text, _white(0.86))
                add("▍", _sans(13), _white(0.6))

        self._tv.textStorage().setAttributedString_(out)
        self._tv.scrollRangeToVisible_((out.length(), 0))


def _common_mode_timer(fn, interval):
    """A started rumps timer that keeps firing while menus are open."""
    import rumps

    t = rumps.Timer(fn, interval)
    t.start()
    try:
        from Foundation import NSRunLoop, NSRunLoopCommonModes

        nstimer = getattr(t, "_nstimer", None)
        if nstimer is not None:
            NSRunLoop.currentRunLoop().addTimer_forMode_(nstimer, NSRunLoopCommonModes)
    except Exception:  # noqa: BLE001
        pass
    return t


def _white(alpha):
    from AppKit import NSColor

    return NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha)


def _accent():
    from AppKit import NSColor

    return NSColor.colorWithSRGBRed_green_blue_alpha_(0.45, 0.68, 1.0, 1.0)


def _sans(size, weight=0.0):
    from AppKit import NSFont

    return NSFont.systemFontOfSize_weight_(size, weight)
