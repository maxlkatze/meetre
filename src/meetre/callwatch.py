"""Detect when a Microsoft Teams call starts.

A call is "active" while a Teams process is capturing from the microphone. On
macOS 14.2+ CoreAudio exposes a per-process object list
(``kAudioHardwarePropertyProcessObjectList``) with each process's bundle id and
whether it is currently running *input*, so we can attribute the mic to Teams
exactly. On older systems we fall back to "Teams is running and the default
input device is in use by someone".

Everything is plain ``ctypes`` against CoreAudio/CoreFoundation/libproc (no
extra pyobjc framework packages) and PATH-independent, so it works when meetre
is launched by launchd.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import threading
import time
from typing import Callable, Optional

# Bundle-id prefix shared by new Teams (com.microsoft.teams2) and classic Teams
# (com.microsoft.teams), plus their helper processes.
TEAMS_BUNDLE_PREFIX = "com.microsoft.teams"
# Executable-path marker for helper processes without a bundle id.
TEAMS_PATH_MARKERS = ("/Microsoft Teams.app/", "/Microsoft Teams classic.app/",
                      "/Microsoft Teams (work or school).app/")


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode("ascii"))[0]


_SYSTEM_OBJECT = 1
_SCOPE_GLOBAL = _fourcc("glob")
_ELEMENT_MAIN = 0
_PROCESS_OBJECT_LIST = _fourcc("prs#")
_PROCESS_PID = _fourcc("ppid")
_PROCESS_BUNDLE_ID = _fourcc("pbid")
_PROCESS_RUNNING_INPUT = _fourcc("piri")
_DEFAULT_INPUT_DEVICE = _fourcc("dIn ")
_DEVICE_RUNNING_SOMEWHERE = _fourcc("gone")
_CF_UTF8 = 0x08000100


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


def _load(name: str, fallback: str):
    try:
        return ctypes.CDLL(ctypes.util.find_library(name) or fallback)
    except OSError:
        return None


_ca = _load("CoreAudio", "/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
_cf = _load("CoreFoundation",
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_libc = _load("c", "/usr/lib/libSystem.B.dylib")

if _ca is not None:
    _ca.AudioObjectGetPropertyDataSize.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32)]
    _ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    _ca.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    _ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
if _cf is not None:
    _cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                       ctypes.c_long, ctypes.c_uint32]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFRelease.argtypes = [ctypes.c_void_p]


def _get(obj: int, selector: int, ctype):
    """Read a fixed-size property; None on error."""
    addr = _Address(selector, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    val = ctype()
    size = ctypes.c_uint32(ctypes.sizeof(val))
    if _ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None,
                                      ctypes.byref(size), ctypes.byref(val)) != 0:
        return None
    return val.value


def _process_objects() -> Optional[list[int]]:
    """CoreAudio process object ids, or None when unsupported (< macOS 14.2)."""
    addr = _Address(_PROCESS_OBJECT_LIST, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    size = ctypes.c_uint32(0)
    if _ca.AudioObjectGetPropertyDataSize(_SYSTEM_OBJECT, ctypes.byref(addr), 0,
                                          None, ctypes.byref(size)) != 0:
        return None
    n = size.value // ctypes.sizeof(ctypes.c_uint32)
    if n == 0:
        return []
    buf = (ctypes.c_uint32 * n)()
    if _ca.AudioObjectGetPropertyData(_SYSTEM_OBJECT, ctypes.byref(addr), 0, None,
                                      ctypes.byref(size), buf) != 0:
        return None
    return list(buf[: size.value // ctypes.sizeof(ctypes.c_uint32)])


def _bundle_id(obj: int) -> str:
    if _cf is None:
        return ""
    addr = _Address(_PROCESS_BUNDLE_ID, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ref))
    if _ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None,
                                      ctypes.byref(size), ctypes.byref(ref)) != 0 \
            or not ref.value:
        return ""
    try:
        buf = ctypes.create_string_buffer(512)
        ok = _cf.CFStringGetCString(ref, buf, len(buf), _CF_UTF8)
        return buf.value.decode("utf-8", "replace") if ok else ""
    finally:
        _cf.CFRelease(ref)  # returned under the CF "copy" rule


def _pid_path(pid: int) -> str:
    if _libc is None or not pid:
        return ""
    buf = ctypes.create_string_buffer(4096)
    try:
        n = _libc.proc_pidpath(ctypes.c_int(pid), buf, ctypes.c_uint32(len(buf)))
    except AttributeError:
        return ""
    return buf.value.decode("utf-8", "replace") if n > 0 else ""


def _is_teams_process(obj: int) -> bool:
    if _bundle_id(obj).lower().startswith(TEAMS_BUNDLE_PREFIX):
        return True
    path = _pid_path(_get(obj, _PROCESS_PID, ctypes.c_int32) or 0)
    return any(m in path for m in TEAMS_PATH_MARKERS)


def _teams_running() -> bool:
    try:
        from AppKit import NSWorkspace

        return any((app.bundleIdentifier() or "").lower().startswith(TEAMS_BUNDLE_PREFIX)
                   for app in NSWorkspace.sharedWorkspace().runningApplications())
    except Exception:  # noqa: BLE001
        return False


def teams_call_active() -> bool:
    """True while Microsoft Teams is capturing from the microphone."""
    if _ca is None:
        return False
    procs = _process_objects()
    if procs is not None:
        return any(_get(p, _PROCESS_RUNNING_INPUT, ctypes.c_uint32)
                   and _is_teams_process(p) for p in procs)
    # Fallback (< macOS 14.2): Teams is open and *something* holds the mic.
    if not _teams_running():
        return False
    dev = _get(_SYSTEM_OBJECT, _DEFAULT_INPUT_DEVICE, ctypes.c_uint32)
    return bool(dev and _get(dev, _DEVICE_RUNNING_SOMEWHERE, ctypes.c_uint32))


class CallWatcher:
    """Poll in a background thread and fire ``on_start`` once per call.

    A call must be seen on ``confirm`` consecutive polls before it counts (so a
    quick mic test doesn't trigger), and must be gone for ``end_grace`` seconds
    before the next one can fire (so a brief mute/device switch doesn't re-ask).
    ``on_start`` runs on the watcher thread — hand off to the main thread.
    """

    def __init__(self, on_start: Callable[[], None], *, interval: float = 3.0,
                 confirm: int = 2, end_grace: float = 20.0):
        self.on_start = on_start
        self.interval = interval
        self.confirm = confirm
        self.end_grace = end_grace
        self.in_call = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self.in_call = False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        seen = 0
        last_seen = 0.0
        while not self._stop.wait(self.interval):
            try:
                active = teams_call_active()
            except Exception:  # noqa: BLE001
                active = False
            now = time.monotonic()
            if active:
                seen += 1
                last_seen = now
                if not self.in_call and seen >= self.confirm:
                    self.in_call = True
                    try:
                        self.on_start()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                seen = 0
                if self.in_call and now - last_seen >= self.end_grace:
                    self.in_call = False
