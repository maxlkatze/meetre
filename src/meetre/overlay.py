"""Live overlays: translucent, click-through HUDs in the top-right corner.

:class:`TranscriptOverlay` shows live captions while recording;
:class:`SummaryOverlay` streams the summary afterwards (same window, so the
two never compete for the corner).

While the local LLM writes the summary, its output streams into a small
borderless panel (graphite, grayscale type). The panel ignores
the mouse, never takes focus, floats over every Space and full-screen app, and
fades out a few seconds after the summary is done.

Thread model: :meth:`begin`, :meth:`feed` and :meth:`finish` may be called from
any thread (the summarizer streams from a worker); they only store state.
:meth:`tick` — driven by a main-thread timer — does all the AppKit work.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Optional

try:  # CGColor() without the Quartz bindings yields an opaque pointer; that's fine.
    import objc

    warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)
except Exception:  # noqa: BLE001
    pass

W, H = 440, 300
MARGIN = 14
OPACITY = 0.97
HOLD_AFTER_DONE = 8.0   # seconds the finished summary stays visible
MAX_CHARS = 2400        # tail of the text that is rendered (keeps frames cheap)

# Neutral grayscale palette (white at varying strength over a graphite panel).
_BG = (0.105, 0.105, 0.115)
_TEXT, _SUBTLE, _MUTED, _FAINT = 0.92, 0.62, 0.45, 0.14


def _color(rgb, alpha=1.0):
    from AppKit import NSColor

    return NSColor.colorWithSRGBRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], alpha)


def _white(alpha):
    from AppKit import NSColor

    return NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha)


def _sans(size, weight=0.0):
    from AppKit import NSFont

    return NSFont.systemFontOfSize_weight_(size, weight)


class SummaryOverlay:
    TITLE = "MEETRE  ·  LIVE SUMMARY"
    HOLD = HOLD_AFTER_DONE

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None       # None | "active" | "done"
        self._phase = "thinking"
        self._text = ""
        self._stage = ""
        self._model = ""
        self._started = 0.0
        self._done_at = 0.0
        self._dirty = False
        self._panel = None
        self._shown = False
        self._frame = 0

    # -- any thread ---------------------------------------------------------

    def begin(self, model: str = "") -> None:
        with self._lock:
            self._state = "active"
            self._phase = "thinking"
            self._text = ""
            self._stage = ""
            self._model = model
            self._started = time.monotonic()
            self._dirty = True

    def feed(self, phase: str, text: str, stage: str = "") -> None:
        with self._lock:
            if self._state != "active":
                return
            self._phase, self._text, self._stage = phase, text, stage
            self._dirty = True

    def finish(self, text: Optional[str] = None) -> None:
        """Mark the summary complete (``text`` = final cleaned summary, if any).
        An empty result hides the overlay right away."""
        with self._lock:
            if self._state != "active":
                return
            if text is not None:
                self._text = text
            self._phase = "done"
            self._state = "done"
            self._done_at = time.monotonic() - (self.HOLD if not self._text.strip() else 0)
            self._dirty = True

    # -- main thread --------------------------------------------------------

    def tick(self, _timer=None) -> None:
        try:
            self._tick()
        except Exception:  # noqa: BLE001 — never let the HUD break the app
            pass

    def _tick(self) -> None:
        with self._lock:
            state, phase, text, stage = self._state, self._phase, self._text, self._stage
            dirty, self._dirty = self._dirty, False
            started, done_at, model = self._started, self._done_at, self._model

        if state is None:
            return
        if state == "done" and time.monotonic() - done_at > self.HOLD:
            self._hide()
            with self._lock:
                if self._state == "done":
                    self._state = None
            return

        if self._panel is None:
            self._build()
        if not self._shown:
            self._show()

        self._frame += 1
        self._animate(phase)
        if dirty or self._frame % 8 == 0:  # also refresh cursor + timer
            self._render(phase, text, stage, started, model)

    # -- window ---------------------------------------------------------------

    def _build(self) -> None:
        from AppKit import (
            NSAppearance, NSBackingStoreBuffered, NSMakeRect, NSPanel, NSScreen,
            NSScrollView, NSTextField, NSTextView, NSView, NSVisualEffectView,
        )

        screen = NSScreen.mainScreen()
        vf = screen.visibleFrame()
        x = vf.origin.x + vf.size.width - W - MARGIN
        y = vf.origin.y + vf.size.height - H - MARGIN

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, W, H),
            0 | (1 << 7),  # borderless | non-activating panel
            NSBackingStoreBuffered, False)
        panel.setReleasedWhenClosed_(False)
        panel.setOpaque_(False)
        from AppKit import NSColor

        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)     # pure overlay: clicks pass through
        panel.setLevel_(25)                     # NSStatusWindowLevel
        panel.setHidesOnDeactivate_(False)
        # all Spaces | stationary | skip ⌘-Tab cycle | over full-screen apps
        panel.setCollectionBehavior_(1 | 16 | 64 | 256)
        panel.setAlphaValue_(0.0)

        root = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        root.setMaterial_(13)       # HUD window
        root.setBlendingMode_(0)    # behind window
        root.setState_(1)           # always active
        root.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameVibrantDark"))
        root.setWantsLayer_(True)
        # The rounded clip + border live on a container, so the panel can be
        # shrunk to a header strip (TranscriptOverlay minimize) by shifting the
        # fixed-size root down instead of re-laying out every subview.
        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        container.setWantsLayer_(True)
        layer = container.layer()
        layer.setCornerRadius_(14)
        layer.setMasksToBounds_(True)
        layer.setBorderWidth_(1.0)
        layer.setBorderColor_(_white(_FAINT).CGColor())
        container.addSubview_(root)
        panel.setContentView_(container)
        self._root = root

        tint = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        tint.setWantsLayer_(True)
        tint.layer().setBackgroundColor_(_color(_BG, 0.94).CGColor())
        root.addSubview_(tint)

        def label(frame, font, color, align=0):
            f = NSTextField.labelWithString_("")
            f.setFrame_(frame)
            f.setFont_(font)
            f.setTextColor_(color)
            f.setAlignment_(align)
            root.addSubview_(f)
            return f

        top = H - 30
        self._title = label(NSMakeRect(16, top, 260, 16), _sans(10, 0.4), _white(_SUBTLE))
        self._title.setAttributedStringValue_(self._kerned(self.TITLE,
                                                           _sans(10, 0.4), _white(_SUBTLE), 1.2))
        self._phase_lbl = label(NSMakeRect(W - 216, top, 200, 16), _sans(10, 0.4),
                                _white(_MUTED), align=2)  # right

        # Scan rail: a faint line with a bright "comet" sweeping along it.
        rail_y = top - 8
        rail = NSView.alloc().initWithFrame_(NSMakeRect(16, rail_y, W - 32, 1))
        rail.setWantsLayer_(True)
        rail.layer().setBackgroundColor_(_white(0.08).CGColor())
        root.addSubview_(rail)
        comet = NSView.alloc().initWithFrame_(NSMakeRect(16, rail_y - 0.5, 56, 2))
        comet.setWantsLayer_(True)
        cl = comet.layer()
        cl.setBackgroundColor_(_white(0.55).CGColor())
        cl.setCornerRadius_(1)
        cl.setShadowColor_(_white(0.6).CGColor())
        cl.setShadowRadius_(5)
        cl.setShadowOpacity_(1.0)
        cl.setShadowOffset_((0, 0))
        root.addSubview_(comet)
        self._comet, self._rail_y = comet, rail_y

        body_top = rail_y - 10
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, 30, W - 32, body_top - 30))
        scroll.setDrawsBackground_(False)
        scroll.setHasVerticalScroller_(False)
        scroll.setHasHorizontalScroller_(False)
        tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, W - 32, body_top - 30))
        tv.setEditable_(False)
        tv.setSelectable_(False)
        tv.setDrawsBackground_(False)
        tv.setRichText_(True)
        tv.setTextContainerInset_((0, 0))
        tv.textContainer().setLineFragmentPadding_(0)
        scroll.setDocumentView_(tv)
        root.addSubview_(scroll)
        self._tv = tv

        self._footer = label(NSMakeRect(16, 10, W - 32, 14), _sans(9.5), _white(_MUTED))
        self._panel = panel

    def _show(self) -> None:
        from AppKit import NSAnimationContext

        self._panel.orderFrontRegardless()
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.35)
        self._panel.animator().setAlphaValue_(OPACITY)
        NSAnimationContext.endGrouping()
        self._shown = True

    def _hide(self) -> None:
        if self._panel is None or not self._shown:
            return
        from AppKit import NSAnimationContext

        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.6)
        self._panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()
        self._shown = False
        # orderOut once the fade has finished (next idle tick is fine).
        try:
            self._panel.performSelector_withObject_afterDelay_("orderOut:", None, 0.7)
        except Exception:  # noqa: BLE001
            pass

    # -- drawing --------------------------------------------------------------

    @staticmethod
    def _kerned(text, font, color, kern=1.6, align=0):
        from AppKit import (
            NSAttributedString, NSFontAttributeName, NSForegroundColorAttributeName,
            NSKernAttributeName, NSMutableParagraphStyle, NSParagraphStyleAttributeName,
        )

        para = NSMutableParagraphStyle.alloc().init()
        para.setAlignment_(align)  # 0 left, 2 right
        return NSAttributedString.alloc().initWithString_attributes_(
            text, {NSFontAttributeName: font, NSForegroundColorAttributeName: color,
                   NSKernAttributeName: kern, NSParagraphStyleAttributeName: para})

    def _animate(self, phase: str) -> None:
        if phase == "done":
            self._comet.setHidden_(True)
            return
        self._comet.setHidden_(False)
        span = W - 32 - 56
        # ping-pong sweep, ~2 s per pass at 15 fps
        t = (self._frame % 60) / 30.0
        pos = t if t <= 1 else 2 - t
        ease = pos * pos * (3 - 2 * pos)
        self._comet.setFrameOrigin_((16 + span * ease, self._rail_y - 0.5))

    def _render(self, phase, text, stage, started, model) -> None:
        from AppKit import (
            NSFontAttributeName, NSForegroundColorAttributeName, NSKernAttributeName,
            NSMutableAttributedString, NSMutableParagraphStyle,
            NSParagraphStyleAttributeName,
        )

        blink = (self._frame // 8) % 2 == 0
        dot = "●" if blink or phase == "done" else "○"
        if phase == "done":
            tag, col = "✓ DONE", _SUBTLE
        elif phase == "writing":
            tag, col = f"{dot} WRITING", _SUBTLE
        else:
            tag, col = f"{dot} THINKING", _MUTED
        if stage and phase != "done":
            tag = f"{stage.upper()} · {tag}"
        self._phase_lbl.setAttributedStringValue_(
            self._kerned(tag, _sans(10, 0.4), _white(col), 1.2, align=2))

        out = NSMutableAttributedString.alloc().init()
        para = NSMutableParagraphStyle.alloc().init()
        para.setLineSpacing_(2.5)
        para.setParagraphSpacing_(3)
        # Bullets: wrapped lines hang under the text, not under the "▸".
        bullet = para.mutableCopy()
        bullet.setHeadIndent_(12)

        def add(s, font, color, kern=0.0, style=para):
            out.appendAttributedString_(type(out).alloc().initWithString_attributes_(
                s, {NSFontAttributeName: font, NSForegroundColorAttributeName: color,
                    NSParagraphStyleAttributeName: style, NSKernAttributeName: kern}))

        body = text[-MAX_CHARS:]
        if phase == "thinking":
            add(body.strip() or "Thinking…", _sans(11), _white(_MUTED))
        else:
            lines = body.strip("\n").split("\n")
            for i, raw in enumerate(lines):
                line = raw.replace("**", "").rstrip()
                nl = "\n" if i < len(lines) - 1 else ""
                st = line.lstrip()
                if st.startswith("#"):
                    add(st.lstrip("#").strip().upper() + nl, _sans(10, 0.5), _white(_SUBTLE), 1.0)
                elif st[:2] in ("- ", "* "):
                    add("•  ", _sans(12.5), _white(_MUTED), style=bullet)
                    add(st[2:] + nl, _sans(12.5), _white(_TEXT), style=bullet)
                else:
                    add(line + nl, _sans(12.5), _white(_TEXT))
        if phase != "done" and blink:
            add("▍", _sans(12.5), _white(_SUBTLE))

        self._tv.textStorage().setAttributedString_(out)
        self._tv.scrollRangeToVisible_((out.length(), 0))

        secs = time.monotonic() - started
        chars = len(text)
        size = f"{chars / 1000:.1f}k" if chars >= 1000 else str(chars)
        self._footer.setAttributedStringValue_(self._kerned(
            f"{secs:.0f}s  ·  {size} chars  ·  {model}  ·  on-device",
            _sans(9.5), _white(_MUTED), 0.2))


MINI_H = 46  # minimized: header + scan rail only

_target_cls = None


def _button_target_class():
    """NSObject subclass forwarding a button click to a Python callable
    (defined lazily: PyObjC class names must be unique per process)."""
    global _target_cls
    if _target_cls is None:
        from Foundation import NSObject

        class MeetreHudButtonTarget(NSObject):
            def clicked_(self, sender):
                self.fn()

        _target_cls = MeetreHudButtonTarget
    return _target_cls


class TranscriptOverlay(SummaryOverlay):
    """Live captions while recording: committed lines (speaker-labelled when
    both mic and system audio are captured) plus the in-flux tail, dimmed.
    Always pinned to the newest text. Fed by :class:`meetre.live.LiveTranscriber`.
    """

    TITLE = "MEETRE  ·  LIVE TRANSCRIPT"
    HOLD = 1.5

    def __init__(self, minimized: bool = False, on_minimize=None):
        super().__init__()
        self._lines: list = []
        self._tentative = ""
        self._minimized = minimized
        self._on_minimize = on_minimize  # callable(bool), e.g. persist to config
        self._btn = None
        self._hover = False

    # -- minimize button --------------------------------------------------------
    # The panel stays click-through everywhere except over this button: each
    # tick checks the pointer and only accepts mouse events while it hovers it.

    def _build(self) -> None:
        super()._build()
        from AppKit import NSButton, NSImage, NSMakeRect

        top = H - 30
        # Make room for the button right of the status label.
        self._phase_lbl.setFrame_(NSMakeRect(W - 240, top, 200, 16))
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 36, top - 3, 22, 22))
        btn.setBordered_(False)
        btn.setTitle_("")
        btn.setImagePosition_(1)  # image only
        target = _button_target_class().alloc().init()
        target.fn = self.toggle_minimized
        btn.setTarget_(target)
        btn.setAction_("clicked:")
        btn.setToolTip_("Minimize / expand live transcript")
        self._root.addSubview_(btn)
        self._btn, self._btn_target = btn, target
        self._btn_image = lambda name: NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            name, "Minimize")
        self._apply_minimized()

    def toggle_minimized(self) -> None:
        self._minimized = not self._minimized
        self._apply_minimized()
        if self._on_minimize is not None:
            try:
                self._on_minimize(self._minimized)
            except Exception:  # noqa: BLE001
                pass

    def _apply_minimized(self) -> None:
        from AppKit import NSMakeRect

        if self._panel is None:
            return
        h = MINI_H if self._minimized else H
        f = self._panel.frame()
        top = f.origin.y + f.size.height  # keep the top edge pinned
        self._root.setFrameOrigin_((0, h - H))
        self._panel.setFrame_display_(NSMakeRect(f.origin.x, top - h, W, h), True)
        img = self._btn_image("chevron.down" if self._minimized else "chevron.up")
        if img is not None:
            self._btn.setImage_(img)
        else:
            self._btn.setTitle_("+" if self._minimized else "–")
        self._style_button()

    def _style_button(self) -> None:
        try:
            self._btn.setContentTintColor_(_white(_TEXT if self._hover else _MUTED))
        except Exception:  # noqa: BLE001
            pass

    def _track_mouse(self) -> None:
        from AppKit import NSEvent, NSPointInRect

        if self._btn is None or not self._shown:
            return
        rect = self._panel.convertRectToScreen_(
            self._btn.convertRect_toView_(self._btn.bounds(), None))
        # A little slack around the 22 pt button makes it easy to hit.
        rect = ((rect.origin.x - 4, rect.origin.y - 4),
                (rect.size.width + 8, rect.size.height + 8))
        hover = bool(NSPointInRect(NSEvent.mouseLocation(), rect))
        if hover != self._hover:
            self._hover = hover
            self._panel.setIgnoresMouseEvents_(not hover)
            self._style_button()

    def _tick(self) -> None:
        super()._tick()
        self._track_mouse()

    def _hide(self) -> None:
        super()._hide()
        if self._panel is not None:
            self._hover = False
            self._panel.setIgnoresMouseEvents_(True)

    def begin(self, model: str = "") -> None:
        with self._lock:
            self._lines, self._tentative = [], ""
        super().begin(model)
        self.feed_lines([], "", "starting")

    def feed_lines(self, lines, tentative: str, status: str) -> None:
        with self._lock:
            if self._state != "active":
                return
            self._lines, self._tentative, self._phase = list(lines), tentative, status
            self._dirty = True

    def finish(self, text: Optional[str] = None) -> None:
        """Recording stopped: show "stopped" briefly, then fade out."""
        with self._lock:
            if self._state != "active":
                return
            self._phase, self._state = "done", "done"
            self._done_at = time.monotonic()
            self._dirty = True

    def _animate(self, phase: str) -> None:
        # The comet sweeps while a model is loading/downloading.
        if phase == "listening":
            self._comet.setHidden_(True)
        else:
            super()._animate(phase)

    def _render(self, phase, text, stage, started, model) -> None:
        from AppKit import (
            NSFontAttributeName, NSForegroundColorAttributeName, NSKernAttributeName,
            NSMutableAttributedString, NSMutableParagraphStyle,
            NSParagraphStyleAttributeName,
        )

        with self._lock:
            lines, tentative = list(self._lines), self._tentative

        blink = (self._frame // 8) % 2 == 0
        dot = "●" if blink else "○"
        if phase == "done":
            tag, col = "■ STOPPED", _MUTED
        elif phase == "listening":
            tag, col = f"{dot} LIVE", _SUBTLE
        else:
            tag, col = f"{dot} {phase.upper()}", _MUTED
        self._phase_lbl.setAttributedStringValue_(
            self._kerned(tag, _sans(10, 0.4), _white(col), 1.2, align=2))

        out = NSMutableAttributedString.alloc().init()
        para = NSMutableParagraphStyle.alloc().init()
        para.setLineSpacing_(2.5)
        para.setParagraphSpacing_(6)

        def add(s, font, color, kern=0.0):
            out.appendAttributedString_(type(out).alloc().initWithString_attributes_(
                s, {NSFontAttributeName: font, NSForegroundColorAttributeName: color,
                    NSParagraphStyleAttributeName: para, NSKernAttributeName: kern}))

        # Only the tail is rendered; walk back until MAX_CHARS is filled.
        shown, budget = [], MAX_CHARS - len(tentative)
        for who, said in reversed(lines):
            if budget <= 0:
                break
            shown.append((who, said[-budget:]))
            budget -= len(said)
        shown.reverse()

        for i, (who, said) in enumerate(shown):
            if i:
                add("\n", _sans(12.5), _white(_TEXT))
            if who:
                add(who.upper() + "  ", _sans(9.5, 0.5), _white(_MUTED), 1.0)
            add(said, _sans(12.5), _white(_TEXT))
        if tentative:
            add(("  " if shown else "") + tentative, _sans(12.5), _white(_SUBTLE))
        if not shown and not tentative:
            add("Listening…" if phase in ("listening", "done") else "Preparing live captions…",
                _sans(11), _white(_MUTED))
        if phase == "listening" and blink:
            add("▍", _sans(12.5), _white(_SUBTLE))

        self._tv.textStorage().setAttributedString_(out)
        self._tv.scrollRangeToVisible_((out.length(), 0))

        m, s = divmod(int(time.monotonic() - started), 60)
        words = sum(len(t.split()) for _, t in lines)
        self._footer.setAttributedStringValue_(self._kerned(
            f"{m:02d}:{s:02d}  ·  {words} words  ·  {model}  ·  on-device",
            _sans(9.5), _white(_MUTED), 0.2))
