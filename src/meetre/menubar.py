"""macOS menu-bar app for meetre.

A status-bar (top-bar) icon you click to record meetings, pick the model /
language / speakers, and summarise transcripts via Claude Desktop + Apple
Notes — all without the terminal.

Built on **rumps** (the status item + menu) plus a small **AppKit** settings
window (the pickers + speaker slider) shown before recording.

Launch with::

    meetre menubar      # or: meetre-menubar
"""

from __future__ import annotations

import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import rumps
except ImportError:  # pragma: no cover - optional GUI extra
    rumps = None

from .config import MODELS, Config

# Languages offered in the picker; None = auto-detect.
LANGUAGES = [("Deutsch", "de"), ("English", "en"), ("Auto-detect", None)]

def summary_choices():
    """Build the summary-model dropdown for THIS machine.

    Returns a list of ``(title, value, fits)``. ``value`` is a model alias,
    ``"auto"``, or None (= no summary). ``fits`` is False for models too large
    for the detected RAM — those are shown but grayed out / unselectable.
    """
    from . import summarizer

    total = summarizer.system_memory_gb()
    auto_alias = summarizer.default_model(total)
    items = [(f"Auto — best that fits ({auto_alias}, ~{summarizer.SUMMARY_MODELS[auto_alias].size_gb:.0f} GB)",
              "auto", True)]
    for alias, spec, fits in summarizer.model_catalog(total):
        suffix = "" if fits else " — needs more RAM"
        items.append((f"{alias}  (~{spec.size_gb:.0f} GB){suffix}", alias, fits))
    items.append(("Off — transcript only (no summary)", None, True))
    return items

# Idle icon: an AI sparkle. The spinner twinkles while working.
IDLE_TITLE = "✦"
SPINNER = ["✦", "✧", "⊹", "✧"]


def _ensure_bundle_identifier(bundle_id: str = "net.cubedpixels.meetre") -> None:
    """Give the process a bundle identifier so notifications work.

    Run unbundled as ``python -m meetre.menubar``, the main bundle has no
    CFBundleIdentifier, so ``NSUserNotificationCenter.defaultUserNotificationCenter()``
    returns nil and every ``rumps.notification`` is silently dropped. Injecting an
    id into the in-memory info dictionary makes the center valid and notifications
    deliver. No-op if an id is already present (e.g. a real .app bundle).
    """
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None and not info.get("CFBundleIdentifier"):
            info["CFBundleIdentifier"] = bundle_id
    except Exception:  # noqa: BLE001
        pass


def _summary_error_hint(e: Exception) -> str:
    """Turn a raw summarizer error into an actionable message.

    The most common one is an mlx-lm too old for a newer architecture (e.g.
    'Model type gemma4 not supported'), which otherwise surfaces as a fleeting
    failure and an empty summary. Point the user at the update path instead.
    """
    msg = str(e)
    low = msg.lower()
    if "not supported" in low or "model type" in low or "unknown model" in low:
        return ("This model needs a newer mlx-lm. Run 'meetre update', or pick "
                "another summary model. (" + msg + ")")
    return msg


def _app_version() -> str:
    """Installed package version, for the About submenu ('?' if unknown)."""
    try:
        from importlib.metadata import version

        return version("meetre")
    except Exception:  # noqa: BLE001
        return "?"


def _require_rumps():
    if rumps is None:
        raise SystemExit(
            "The menu-bar app needs the GUI extra:\n"
            "  pip install -e '.[menubar]'\n"
            "(installs rumps + pyobjc)"
        )


# ---------------------------------------------------------------------------
# Settings popup window (AppKit)
# ---------------------------------------------------------------------------

_CTRL_CLASS = None


def _settings_controller_class():
    """Define the AppKit action-target class exactly once and cache it.

    Re-defining an Obj-C class with the same name raises
    ``objc.error: _Ctrl is overriding existing Objective-C class``, which is why
    this must not live inside the window builder.
    """
    global _CTRL_CLASS
    if _CTRL_CLASS is not None:
        return _CTRL_CLASS
    import objc

    class _SettingsController(objc.lookUpClass("NSObject")):
        # Callbacks are attached per-instance as Python attributes.
        def sliderChanged_(self, sender):
            self.on_slider(int(sender.intValue()))

        def startClicked_(self, sender):
            self.on_start()

        def cancelClicked_(self, sender):
            self.on_cancel()

        def resetClicked_(self, sender):
            self.on_reset()

    _CTRL_CLASS = _SettingsController
    return _CTRL_CLASS

def _build_settings_window(cfg: Config, on_start, *, want_name: bool):
    """Construct and show the pre-recording settings window.

    ``on_start(values: dict)`` is called when the user clicks Start. Returns the
    window controller, which the caller must keep a reference to so it survives
    until dismissed.
    """
    import objc
    from AppKit import (
        NSApp, NSBackingStoreBuffered, NSButton, NSButtonTypeSwitch, NSColor,
        NSFont, NSMakeRect, NSPopUpButton, NSScrollView, NSSlider, NSTextField,
        NSTextView, NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskTitled,
    )
    from . import summarizer

    W, H = 560, 800
    PAD = 24
    row = [H - 48]  # mutable cursor, walking down from the top

    def next_y(step=40):
        row[0] -= step
        return row[0]

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, H),
        NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
        NSBackingStoreBuffered, False,
    )
    win.setTitle_("meetre — recording settings")
    # A programmatically created NSWindow defaults to releasedWhenClosed=True, so
    # AppKit releases it on close() / the red close button while pyobjc still
    # holds it (via the controller below). The second time the window is opened
    # the previous controller is GC'd and over-releases the already-freed window
    # → crash. Let Python own the lifetime instead.
    win.setReleasedWhenClosed_(False)
    win.center()
    content = win.contentView()

    def label(text, y, *, bold=False, size=13, color=None):
        lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(PAD, y, W - 2 * PAD, 20))
        lbl.setStringValue_(text)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        if color is not None:
            lbl.setTextColor_(color)
        content.addSubview_(lbl)
        return lbl

    # Heading
    label("Meeting recording settings", next_y(34), bold=True, size=15)

    # Meeting name
    name_field = None
    if want_name:
        label("Meeting name", next_y(36))
        name_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(PAD, next_y(26), W - 2 * PAD, 24))
        name_field.setPlaceholderString_("optional")
        content.addSubview_(name_field)

    # Model picker
    label("Model", next_y(38))
    model_pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
        NSMakeRect(PAD, next_y(28), W - 2 * PAD, 26), False)
    model_pop.addItemsWithTitles_(list(MODELS))
    if cfg.model in MODELS:
        model_pop.selectItemWithTitle_(cfg.model)
    content.addSubview_(model_pop)

    # Language picker
    label("Language", next_y(38))
    lang_pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
        NSMakeRect(PAD, next_y(28), W - 2 * PAD, 26), False)
    lang_pop.addItemsWithTitles_([name for name, _ in LANGUAGES])
    for i, (_, code) in enumerate(LANGUAGES):
        if code == cfg.language:
            lang_pop.selectItemAtIndex_(i)
    content.addSubview_(lang_pop)

    # Summary model picker. Sized for THIS machine: models too big for the
    # detected RAM are listed but grayed out. Last entry disables summaries.
    summary_items = summary_choices()
    label("Summary model", next_y(38))
    summary_pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
        NSMakeRect(PAD, next_y(28), W - 2 * PAD, 26), False)
    # Manage item enabled-state ourselves so we can gray out models that don't fit.
    summary_pop.menu().setAutoenablesItems_(False)
    cur_summary = None if not cfg.auto_summarize else cfg.summary_model
    for i, (title, val, fits) in enumerate(summary_items):
        summary_pop.addItemWithTitle_(title)
        item = summary_pop.itemAtIndex_(i)
        # Keep the currently-saved choice selectable even if it no longer fits.
        item.setEnabled_(bool(fits) or val == cur_summary)
        if val == cur_summary:
            summary_pop.selectItemAtIndex_(i)
    content.addSubview_(summary_pop)

    # Checkboxes: system audio + person detection + summarize after
    def checkbox(text, y, on):
        b = NSButton.alloc().initWithFrame_(NSMakeRect(PAD, y, W - 2 * PAD, 22))
        b.setButtonType_(NSButtonTypeSwitch)
        b.setTitle_(text)
        b.setState_(1 if on else 0)
        content.addSubview_(b)
        return b

    sysaudio_cb = checkbox("Capture system audio (ScreenCaptureKit, other participants)",
                           next_y(40), cfg.capture_system)
    persons_cb = checkbox("Detect speakers (person detection)", next_y(28), cfg.person_detection)
    summarize_cb = checkbox("After: save summary + transcript to Apple Notes",
                            next_y(28), cfg.auto_notes)

    # Speaker slider (0 = auto)
    spk_label = label("Speakers: auto", next_y(40), bold=True)
    slider = NSSlider.alloc().initWithFrame_(NSMakeRect(PAD, next_y(30), W - 2 * PAD, 24))
    slider.setMinValue_(0)
    slider.setMaxValue_(10)
    slider.setNumberOfTickMarks_(11)
    slider.setAllowsTickMarkValuesOnly_(True)
    init_spk = cfg.num_speakers or 0
    slider.setIntValue_(int(init_spk))

    def spk_text(n):
        return "Speakers: auto" if n == 0 else f"Speakers: exactly {n}"

    spk_label.setStringValue_(spk_text(int(init_spk)))

    # Editable AI summary prompt (persisted), pre-filled with saved or default.
    label("AI summary prompt", next_y(40), bold=True)
    default_prompt_text = summarizer.default_prompt(cfg.language)
    prompt_y = 60
    prompt_h = max(120, row[0] - prompt_y - 8)
    prompt_scroll = NSScrollView.alloc().initWithFrame_(
        NSMakeRect(PAD, prompt_y, W - 2 * PAD, prompt_h))
    prompt_scroll.setHasVerticalScroller_(True)
    prompt_scroll.setBorderType_(2)  # NSBezelBorder
    prompt_tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, W - 2 * PAD, prompt_h))
    prompt_tv.setRichText_(False)
    prompt_tv.setFont_(NSFont.systemFontOfSize_(11))
    prompt_tv.setString_(cfg.summary_prompt or default_prompt_text)
    prompt_scroll.setDocumentView_(prompt_tv)
    content.addSubview_(prompt_scroll)

    def _collect_and_start():
        values = {
            "name": (name_field.stringValue() if name_field else "") or None,
            "model": model_pop.titleOfSelectedItem(),
            "language": LANGUAGES[lang_pop.indexOfSelectedItem()][1],
            "summary_model": summary_items[summary_pop.indexOfSelectedItem()][1],
            "capture_system": bool(sysaudio_cb.state()),
            "person_detection": bool(persons_cb.state()),
            "summarize_after": bool(summarize_cb.state()),
            "speakers": int(slider.intValue()),
            "summary_prompt": str(prompt_tv.string()),
        }
        win.close()
        on_start(values)

    # The controller class is defined ONCE (Obj-C class names are global); each
    # window attaches its own callbacks as plain Python attributes.
    ctrl = _settings_controller_class().alloc().init()
    ctrl.win = win  # tie window lifetime to the controller the caller retains
    ctrl.on_slider = lambda n: spk_label.setStringValue_(spk_text(n))
    ctrl.on_start = _collect_and_start
    ctrl.on_cancel = lambda: win.close()
    ctrl.on_reset = lambda: prompt_tv.setString_(default_prompt_text)
    slider.setTarget_(ctrl)
    slider.setAction_("sliderChanged:")
    content.addSubview_(slider)

    # Buttons
    by = 16
    reset_btn = NSButton.alloc().initWithFrame_(NSMakeRect(PAD, by, 150, 32))
    reset_btn.setTitle_("Reset prompt")
    reset_btn.setBezelStyle_(1)
    reset_btn.setTarget_(ctrl)
    reset_btn.setAction_("resetClicked:")
    content.addSubview_(reset_btn)

    start_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 110, by, 110, 32))
    start_btn.setTitle_("Start" if want_name else "Save")
    start_btn.setBezelStyle_(1)  # rounded
    start_btn.setKeyEquivalent_("\r")
    start_btn.setTarget_(ctrl)
    start_btn.setAction_("startClicked:")
    content.addSubview_(start_btn)

    cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 220, by, 100, 32))
    cancel_btn.setTitle_("Cancel")
    cancel_btn.setBezelStyle_(1)
    cancel_btn.setKeyEquivalent_("\x1b")  # Esc
    cancel_btn.setTarget_(ctrl)
    cancel_btn.setAction_("cancelClicked:")
    content.addSubview_(cancel_btn)

    NSApp.activateIgnoringOtherApps_(True)
    win.makeKeyAndOrderFront_(None)
    win.setLevel_(3)  # floating, above other windows
    return ctrl


# ---------------------------------------------------------------------------
# Status-bar app
# ---------------------------------------------------------------------------

class MeetreApp(rumps.App if rumps else object):
    def __init__(self):
        # Must run before any notification: gives the unbundled process a bundle
        # identifier so the notification center isn't nil.
        _ensure_bundle_identifier()
        # quit_button=None: we add our own Quit item in _build_menu so it keeps
        # a stable position after every menu rebuild (rumps' auto button can
        # otherwise duplicate or jump).
        super().__init__("meetre", title=IDLE_TITLE, quit_button=None)
        self.cfg = Config.load()
        self.state = "idle"          # idle | recording | processing
        self._recorder = None
        self._rec_meta = None        # (title, started, tmp_wav)
        self._settings_ctrl = None   # keep the settings window alive
        self._stage_text = None      # current processing stage label
        self._download = None        # (label, fraction) while downloading
        self._spin = 0               # spinner frame counter
        self._notify_queue = []      # (title, subtitle, msg) from worker threads

        # Real image icon for the menu bar (template-tinted), reused on
        # notifications. Falls back to the Unicode glyph if it can't render.
        from . import icon as _icon

        self._icon_path = _icon.icon_path()
        if self._icon_path:
            try:
                self.template = True       # adapt to light/dark menu bar
                self.icon = self._icon_path
            except Exception:  # noqa: BLE001
                self._icon_path = None

        self._build_menu()
        # A single main-thread timer reflects state into the menu bar.
        self._timer = rumps.Timer(self._tick, 0.5)
        self._timer.start()
        # By default the NSTimer only fires in the default run-loop mode, so the
        # moment the user opens the menu the loop switches to event-tracking mode
        # and the title/spinner freeze and queued notifications stall until the
        # menu closes. Registering the same timer for the common modes keeps it
        # ticking while the menu is open.
        try:
            from Foundation import NSRunLoop, NSRunLoopCommonModes

            nstimer = getattr(self._timer, "_nstimer", None)
            if nstimer is not None:
                NSRunLoop.currentRunLoop().addTimer_forMode_(nstimer, NSRunLoopCommonModes)
        except Exception:  # noqa: BLE001
            pass
        # Check for updates (git pull) in the background on every launch.
        threading.Thread(target=self._do_update, daemon=True).start()

    # -- menu construction --------------------------------------------------

    def _build_menu(self):
        from . import summarizer

        self.menu.clear()
        # Informational status line (disabled item) updated by the timer.
        self.status_item = rumps.MenuItem("✦ Ready")
        self.status_item.set_callback(None)
        self.rec_item = rumps.MenuItem("● Record…", callback=self.on_record)
        self.stop_item = rumps.MenuItem("■ Stop", callback=self.on_stop)
        self.stop_item.set_callback(None)  # disabled until recording

        model_menu = rumps.MenuItem("Model")
        for m in MODELS:
            it = rumps.MenuItem(m, callback=self._make_model_cb(m))
            it.state = 1 if m == self.cfg.model else 0
            model_menu.add(it)

        lang_menu = rumps.MenuItem("Language")
        for name, code in LANGUAGES:
            it = rumps.MenuItem(name, callback=self._make_lang_cb(code))
            it.state = 1 if code == self.cfg.language else 0
            lang_menu.add(it)

        spk_menu = rumps.MenuItem("Speakers")
        for label, num in [("Auto", 0)] + [(str(n), n) for n in range(2, 9)]:
            it = rumps.MenuItem(label, callback=self._make_spk_cb(num))
            it.state = 1 if (self.cfg.num_speakers or 0) == num else 0
            spk_menu.add(it)

        # Summary-model submenu — sized for this machine; too-big models are
        # listed but disabled (grayed). ✓ marks models already downloaded.
        summary_menu = rumps.MenuItem("Summary model")
        auto_alias = summarizer.default_model()
        auto_it = rumps.MenuItem(f"Auto — best that fits ({auto_alias})",
                                 callback=self._make_summary_cb("auto"))
        auto_it.state = 1 if (self.cfg.auto_summarize and self.cfg.summary_model == "auto") else 0
        summary_menu.add(auto_it)
        for alias, spec, fits in summarizer.model_catalog():
            inst = " ✓" if summarizer.is_installed(alias) else ""
            if fits:
                it = rumps.MenuItem(f"{alias} (~{spec.size_gb:.0f} GB){inst}",
                                    callback=self._make_summary_cb(alias))
            else:
                it = rumps.MenuItem(f"{alias} (~{spec.size_gb:.0f} GB) — needs more RAM")
                it.set_callback(None)  # disabled / grayed out
            it.state = 1 if (self.cfg.auto_summarize and self.cfg.summary_model == alias) else 0
            summary_menu.add(it)
        off_it = rumps.MenuItem("Off — transcript only", callback=self._make_summary_cb(None))
        off_it.state = 1 if not self.cfg.auto_summarize else 0
        summary_menu.add(off_it)

        # Downloaded-models submenu — click a model to uninstall and free space.
        downloads_menu = rumps.MenuItem("Downloaded models")
        installed = summarizer.installed_models()
        if not installed:
            empty = rumps.MenuItem("(none downloaded yet)")
            empty.set_callback(None)
            downloads_menu.add(empty)
        else:
            for m in installed:
                it = rumps.MenuItem(f"{m['repo']} — {summarizer.human_size(m['size'])}  ⌫",
                                    callback=self._make_uninstall_cb(m["repo"], m["size"]))
                downloads_menu.add(it)
            downloads_menu.add(None)
            total = sum(m["size"] for m in installed)
            tot = rumps.MenuItem(f"Total: {summarizer.human_size(total)}")
            tot.set_callback(None)
            downloads_menu.add(tot)

        self.sysaudio_item = rumps.MenuItem("System audio (ScreenCaptureKit)",
                                            callback=self.on_toggle_sysaudio)
        self.sysaudio_item.state = 1 if self.cfg.capture_system else 0
        self.persons_item = rumps.MenuItem("Person detection", callback=self.on_toggle_persons)
        self.persons_item.state = 1 if self.cfg.person_detection else 0

        # "About meetre" groups the app-level actions (version, updates,
        # restart, start-at-login, quit) into one submenu so the top level stays
        # focused on recording and per-meeting options.
        from . import autostart

        about_menu = rumps.MenuItem("About meetre")
        ver = rumps.MenuItem(f"meetre {_app_version()}")
        ver.set_callback(None)
        about_menu.add(ver)
        tagline = rumps.MenuItem("Local meeting transcripts + summaries")
        tagline.set_callback(None)
        about_menu.add(tagline)
        about_menu.add(None)
        about_menu.add(rumps.MenuItem("Check for updates", callback=self.on_update))
        about_menu.add(rumps.MenuItem("Restart meetre", callback=self.on_restart))
        self.startup_item = rumps.MenuItem("Start at login", callback=self.on_toggle_startup)
        self.startup_item.state = 1 if autostart.is_enabled() else 0
        about_menu.add(self.startup_item)
        about_menu.add(None)
        about_menu.add(rumps.MenuItem("Quit meetre", callback=rumps.quit_application))

        self.menu = [
            self.status_item,
            None,
            self.rec_item,
            self.stop_item,
            None,
            model_menu,
            lang_menu,
            summary_menu,
            spk_menu,
            self.sysaudio_item,
            self.persons_item,
            None,
            rumps.MenuItem("Settings…", callback=self.on_settings),
            rumps.MenuItem("Summarize last → Apple Notes (local)", callback=self.on_summarize),
            rumps.MenuItem("Open transcripts folder", callback=self.on_open_folder),
            downloads_menu,
            None,
            about_menu,
        ]

    # -- picker callbacks ---------------------------------------------------

    def _make_model_cb(self, model):
        def cb(sender):
            self.cfg.model = model
            self.cfg.save()
            self._build_menu()
            self._post("meetre", "Model", f"Set to {model}")
        return cb

    def _make_lang_cb(self, code):
        def cb(sender):
            self.cfg.language = code
            self.cfg.save()
            self._build_menu()
        return cb

    def _make_summary_cb(self, value):
        """value: model alias, 'auto', or None (= disable summaries)."""
        def cb(sender):
            if value is None:
                self.cfg.auto_summarize = False
            else:
                self.cfg.auto_summarize = True
                self.cfg.summary_model = value
            self.cfg.save()
            self._build_menu()
        return cb

    def _make_uninstall_cb(self, repo, size):
        def cb(sender):
            from . import summarizer

            if rumps.alert(
                title="Uninstall model?",
                message=f"Delete {repo} and free {summarizer.human_size(size)}?\n"
                        "It will be re-downloaded automatically if needed again.",
                ok="Delete", cancel="Cancel",
            ) == 1:
                freed = summarizer.uninstall_model(repo)
                self._build_menu()
                self._post("meetre", "Model removed",
                           f"Freed {summarizer.human_size(freed)} — {repo}")
        return cb

    def _make_spk_cb(self, num):
        def cb(sender):
            self.cfg.num_speakers = num or None
            self.cfg.min_speakers = None
            self.cfg.max_speakers = None
            self.cfg.save()
            self._build_menu()
        return cb

    def on_toggle_sysaudio(self, sender):
        self.cfg.capture_system = not self.cfg.capture_system
        self.cfg.save()
        sender.state = 1 if self.cfg.capture_system else 0

    def on_toggle_persons(self, sender):
        self.cfg.person_detection = not self.cfg.person_detection
        self.cfg.save()
        sender.state = 1 if self.cfg.person_detection else 0

    # -- settings window ----------------------------------------------------

    def on_settings(self, sender=None):
        self.cfg = Config.load()
        self._settings_ctrl = _build_settings_window(
            self.cfg, self._apply_settings, want_name=False)

    def _apply_settings(self, values):
        self._save_values(values)
        self._build_menu()

    def _save_values(self, values):
        self.cfg.model = values["model"]
        self.cfg.language = values["language"]
        # Summary model: None selection = disable summaries (transcript only).
        if "summary_model" in values:
            sm = values["summary_model"]
            if sm is None:
                self.cfg.auto_summarize = False
            else:
                self.cfg.auto_summarize = True
                self.cfg.summary_model = sm
        self.cfg.capture_system = values["capture_system"]
        self.cfg.person_detection = values["person_detection"]
        # Persist the rest so the popup remembers state next time.
        if "summarize_after" in values:
            self.cfg.auto_notes = values["summarize_after"]
        if "summary_prompt" in values:
            # Store empty when it matches the default, so language changes still
            # pick up the right default prompt.
            from . import summarizer

            txt = (values["summary_prompt"] or "").strip()
            self.cfg.summary_prompt = "" if txt == summarizer.default_prompt(values["language"]).strip() else txt
        n = values["speakers"]
        self.cfg.num_speakers = n or None
        self.cfg.min_speakers = None
        self.cfg.max_speakers = None
        self.cfg.save()

    # -- recording ----------------------------------------------------------

    def on_record(self, sender=None):
        if self.state != "idle":
            return
        self.cfg = Config.load()
        self._settings_ctrl = _build_settings_window(
            self.cfg, self._start_recording, want_name=True)

    def _start_recording(self, values):
        from . import recorder as rec
        from . import transcriber

        self._save_values(values)
        self._summarize_after = values.get("summarize_after", False)

        if transcriber.available_backend() is None:
            rumps.alert("No transcription backend installed (pip install mlx-whisper).")
            return

        mic = self.cfg.mic_device if self.cfg.mic_device is not None else rec.default_input_device()
        native = bool(self.cfg.capture_system)
        if mic is None and not native:
            rumps.alert("No audio input device found.")
            return

        title = values.get("name") or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        started = datetime.now()
        stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
        tmp_wav = Path(tempfile.gettempdir()) / f"meetre_{stamp}.wav"

        recorder = rec.Recorder(mic_device=mic, system_device=None, native_system=native)
        try:
            recorder.start(tmp_wav)
        except Exception as e:  # noqa: BLE001
            rumps.alert("Could not start recording", str(e))
            return
        for msg in recorder.start_errors:
            self._post("meetre", "System audio unavailable", msg)

        self._recorder = recorder
        self._rec_meta = (title, started, tmp_wav)
        self.state = "recording"
        self.stop_item.set_callback(self.on_stop)
        self.rec_item.set_callback(None)

    def on_stop(self, sender=None):
        if self.state != "recording" or self._recorder is None:
            return
        self.state = "processing"
        self.stop_item.set_callback(None)
        # Finalise + transcribe off the main thread so the UI stays responsive.
        threading.Thread(target=self._finish, daemon=True).start()

    def _stage(self, text):
        """Set the current processing stage shown in the menu bar."""
        self._download = None
        self._stage_text = text

    def _ensure_model(self, repo, label):
        """Download a model if needed, streaming progress to the menu bar."""
        from . import downloads

        if downloads.is_cached(repo):
            return

        def cb(frac, done, total):
            self._download = (label, frac)

        try:
            downloads.ensure_model(repo, cb)
        finally:
            self._download = None

    def _finish(self):
        from . import recorder as rec
        from . import transcriber
        from .transcript import _slugify, write_transcript

        title, started, tmp_wav = self._rec_meta
        recorder = self._recorder
        try:
            final_audio = recorder.stop()
            stems = dict(getattr(recorder, "stems", {}))
            duration = recorder.seconds

            # MP3 backup
            try:
                mp3 = self.cfg.audio_backup_path / f"{started.strftime('%Y-%m-%d_%H-%M-%S')}_{_slugify(title)}.mp3"
                rec.save_mp3(final_audio, mp3)
            except Exception:  # noqa: BLE001
                pass

            # Ensure the whisper model is present (shows a download bar).
            self._stage("Loading transcription model…")
            self._ensure_model(transcriber.mlx_repo(self.cfg.model), f"Whisper {self.cfg.model}")
            use_persons = self.cfg.person_detection

            source_aware = "mic" in stems and "system" in stems
            if source_aware:
                # One transcription of the mix (clean timing); attribute each
                # segment to you vs. remote from the separated stems.
                self._stage("Transcribing…")
                segments, backend = transcriber.transcribe_attributed(
                    final_audio, stems, model=self.cfg.model, language=self.cfg.language,
                    compute_type=self.cfg.compute_type, detect_speakers=use_persons,
                    hf_token=self.cfg.hf_token, num_speakers=self.cfg.num_speakers,
                    min_speakers=self.cfg.min_speakers, max_speakers=self.cfg.max_speakers,
                )
                use_persons = True
            else:
                self._stage("Transcribing…")
                segments, backend = transcriber.transcribe(
                    final_audio, model=self.cfg.model, language=self.cfg.language,
                    compute_type=self.cfg.compute_type,
                )
                if segments and use_persons:
                    self._stage("Detecting speakers…")
                    try:
                        segments = transcriber.diarize(
                            final_audio, segments, self.cfg.hf_token,
                            num_speakers=self.cfg.num_speakers,
                            min_speakers=self.cfg.min_speakers,
                            max_speakers=self.cfg.max_speakers,
                        )
                    except RuntimeError as e:
                        self._notify("meetre", "Speaker detection skipped", str(e))
                        use_persons = False
            if not segments:
                self._notify("meetre", "Done", "No speech detected.")
                return

            # Generate the summary ONCE; reuse it for the transcript and Notes.
            summary = self._generate_summary(segments)

            path = write_transcript(
                segments, self.cfg.transcripts_path, title=title, started_at=started,
                duration=duration, model=self.cfg.model, backend=backend,
                person_detection=use_persons, summary=summary,
            )
            mins = int(duration // 60)
            done_msg = ("Summary + transcript ready" if summary
                        else "Transcript ready")
            self._notify("meetre", f"✓ {title} — done",
                         f"{done_msg} · {mins} min · {path.name}")

            # Auto-save the same summary + transcript to Apple Notes.
            if self.cfg.auto_notes:
                from . import integrations, summarizer

                self._stage("Saving to Apple Notes…")
                try:
                    integrations.add_to_apple_notes(
                        path.stem, summarizer.transcript_body(path.read_text()),
                        summary_md=summary or None)
                    self._notify("meetre", "Apple Notes", "Saved summary + transcript.")
                except RuntimeError as e:
                    self._notify("meetre", "Apple Notes failed", str(e))

            for p in [final_audio, *stems.values()]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            self._notify("meetre", "Recording failed", str(e))
        finally:
            self._recorder = None
            self._rec_meta = None
            self.state = "idle"
            self.rec_item.set_callback(self.on_record)

    # -- summarize / misc ---------------------------------------------------

    def on_summarize(self, sender=None):
        files = sorted(self.cfg.transcripts_path.glob("*.md"), reverse=True)
        if not files:
            rumps.alert("No transcripts to summarize yet.")
            return

        def work():
            self.state = "processing"
            try:
                self._summarize_path(files[0])
            finally:
                self.state = "idle"
                self._stage_text = None
                self._download = None

        threading.Thread(target=work, daemon=True).start()

    def _generate_summary(self, segments) -> str:
        """Run the local LLM over segments once; '' if disabled/unavailable."""
        from . import summarizer

        if not self.cfg.auto_summarize:
            return ""
        if not summarizer.available():
            self._notify("meetre", "Summary skipped", "Install mlx-lm for summaries.")
            return ""
        text = "\n".join(
            (f"{s.speaker}: " if getattr(s, "speaker", None) else "") + s.text
            for s in segments
        )
        try:
            self._stage("Loading summary model…")
            self._ensure_model(summarizer.resolve_model(self.cfg.summary_model),
                               f"Summary {self.cfg.summary_model}")
            self._stage("Summarizing…")
            return summarizer.summarize(
                text, model=self.cfg.summary_model, language=self.cfg.language,
                prompt=self.cfg.summary_prompt or None)
        except Exception as e:  # noqa: BLE001
            self._notify("meetre", "Summary failed", _summary_error_hint(e))
            return ""

    def _summarize_path(self, path: Path):
        """Manual 'Summarize last': reuse the embedded summary, else generate."""
        from . import integrations, summarizer

        md = path.read_text()
        title = path.stem
        body = summarizer.transcript_body(md)

        # Reuse the summary already embedded in the transcript if present.
        summary = summarizer.extract_summary(md)
        if not summary and self.cfg.auto_summarize and summarizer.available():
            try:
                self._stage("Loading summary model…")
                self._ensure_model(summarizer.resolve_model(self.cfg.summary_model),
                                   f"Summary {self.cfg.summary_model}")
                self._stage("Summarizing…")
                summary = summarizer.summarize(
                    body, model=self.cfg.summary_model, language=self.cfg.language,
                    prompt=self.cfg.summary_prompt or None)
            except Exception as e:  # noqa: BLE001
                self._notify("meetre", "Summary failed", str(e))

        self._stage("Saving to Apple Notes…")
        try:
            integrations.add_to_apple_notes(title, body, summary_md=summary or None)
            self._notify("meetre", "Apple Notes",
                         "Saved summary + transcript." if summary
                         else "Saved transcript (no summary).")
        except RuntimeError as e:
            self._notify("meetre", "Apple Notes failed", str(e))

    def on_open_folder(self, sender=None):
        import subprocess
        self.cfg.transcripts_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(self.cfg.transcripts_path)])

    # -- update / autostart -------------------------------------------------

    def on_update(self, sender=None):
        threading.Thread(target=self._do_update, args=(True,), daemon=True).start()

    def on_restart(self, sender=None):
        """Relaunch the app in place (e.g. to apply a downloaded update)."""
        if self.state != "idle":
            if rumps.alert(
                title="Restart meetre?",
                message="A recording or processing job is still running. "
                        "Restart anyway and discard it?",
                ok="Restart", cancel="Cancel",
            ) != 1:
                return
        import os
        import sys

        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:  # noqa: BLE001
            rumps.alert("Could not restart", str(e))

    def _do_update(self, interactive=False):
        from . import updater

        result = updater.update()
        if result.get("updated"):
            self._notify("meetre", "Update installed",
                         "Restart meetre to apply the new version.")
        elif interactive:
            err = result.get("error")
            self._notify("meetre", "Up to date", err or "Already on the latest version.")

    def on_toggle_startup(self, sender=None):
        from . import autostart

        if autostart.is_enabled():
            autostart.disable()
            sender.state = 0
            self._notify("meetre", "Start at login", "Disabled.")
        else:
            autostart.enable()
            sender.state = 1
            self._notify("meetre", "Start at login", "Enabled.")

    # -- title timer --------------------------------------------------------

    def _set_status(self, text):
        try:
            self.status_item.title = text
        except Exception:  # noqa: BLE001
            pass

    def _notify(self, title, subtitle, msg):
        """Thread-safe notification: queued here, delivered on the main thread.

        Calling rumps/AppKit from a worker thread can segfault, so background
        code uses this and the main-thread timer (_tick) actually posts it.
        """
        self._notify_queue.append((title, subtitle, msg))

    def _post(self, title, subtitle, msg):
        """Post a notification now (main thread only), with the app icon.

        Uses the native center (shows the app icon) when it's available, and
        falls back to AppleScript otherwise so a notification always appears.
        """
        try:
            from Foundation import NSUserNotificationCenter

            if NSUserNotificationCenter.defaultUserNotificationCenter() is not None:
                try:
                    rumps.notification(title, subtitle, msg, icon=self._icon_path)
                except TypeError:
                    rumps.notification(title, subtitle, msg)  # older rumps
                return
        except Exception:  # noqa: BLE001
            pass
        self._osascript_notify(title, subtitle, msg)

    @staticmethod
    def _osascript_notify(title, subtitle, msg):
        """Fallback notification via AppleScript (works for unbundled apps)."""
        import subprocess

        def esc(s):
            return (s or "").replace("\\", "\\\\").replace('"', '\\"')

        script = (f'display notification "{esc(msg)}" '
                  f'with title "{esc(title)}" subtitle "{esc(subtitle)}"')
        try:
            subprocess.run(["osascript", "-e", script], check=False, timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def _tick(self, _timer):
        # Deliver any queued notifications on the main thread.
        while self._notify_queue:
            title, subtitle, msg = self._notify_queue.pop(0)
            self._post(title, subtitle, msg)
        self._spin = (self._spin + 1) % len(SPINNER)
        if self.state == "recording" and self._recorder is not None:
            m, s = divmod(int(self._recorder.seconds), 60)
            self.title = f"⏺ {m:02d}:{s:02d}"
            self._set_status(f"⏺ Recording…  {m:02d}:{s:02d}")
        elif self.state == "processing":
            if self._download is not None:
                from .downloads import bar

                label, frac = self._download
                pct = int(frac * 100)
                self.title = f"⬇ {pct}%"
                self._set_status(f"⬇ {label}  {bar(frac)} {pct}%")
            else:
                star = SPINNER[self._spin]
                stage = self._stage_text or "Working…"
                self.title = star
                self._set_status(f"{star} {stage}")
        else:
            # With an image icon the status bar already shows the glyph, so the
            # idle title is blank; otherwise fall back to the Unicode sparkle.
            self.title = "" if self._icon_path else IDLE_TITLE
            self._set_status("✦ Ready")


def run():
    _require_rumps()
    # Re-exec from inside meetre.app so notifications/app switcher show "meetre"
    # with our icon instead of "python3.12". No-op once already in the bundle;
    # if it can't build, we keep running in place. Must happen before any
    # AppKit/notification setup.
    try:
        from . import bundle

        bundle.relaunch_into_bundle()
    except Exception:  # noqa: BLE001
        pass
    # Capture native segfaults and uncaught exceptions to <repo>/crashlogs/.
    try:
        from . import crashlog

        crashlog.install()
    except Exception:  # noqa: BLE001
        pass
    # Run as a menu-bar-only accessory: no Dock icon, no Python app window.
    try:
        from AppKit import NSApplication

        NSApplication.sharedApplication().setActivationPolicy_(1)  # Accessory
    except Exception:  # noqa: BLE001
        pass
    MeetreApp().run()


if __name__ == "__main__":
    run()
