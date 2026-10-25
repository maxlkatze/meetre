"""Audio capture for macOS — microphone **and** system audio.

macOS does not let an application tap the system audio output directly; you
need a virtual loopback driver. The free, standard one is **BlackHole**:

    brew install blackhole-2ch

Then in *Audio MIDI Setup* create a **Multi-Output Device** (your speakers +
BlackHole) and play meeting audio through it, so you still hear it while
BlackHole receives a copy.

meetre records two sources at once and mixes them into a single track:

* **mic**    — what you say (default input device)
* **system** — what the other participants say (the BlackHole loopback)

Either source can be disabled. ``find_loopback_device()`` auto-detects
BlackHole/Loopback/Soundflower so the common case needs no configuration.
"""

from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16_000  # what whisper expects
CHANNELS = 1
_LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "aggregate", "multi-output")


def list_devices() -> List[dict]:
    """Return input-capable audio devices, flagging loopback candidates."""
    import sounddevice as sd

    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            name = dev["name"]
            devices.append({
                "index": idx,
                "name": name,
                "channels": dev["max_input_channels"],
                "default_samplerate": dev.get("default_samplerate"),
                "loopback": any(h in name.lower() for h in _LOOPBACK_HINTS),
            })
    return devices


def _to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz with an anti-alias filter (the mic now records at
    the device's own rate, usually 48 kHz); linear as a fallback."""
    from math import gcd

    try:
        from scipy.signal import resample_poly

        g = gcd(SAMPLE_RATE, int(sr))
        return resample_poly(data, SAMPLE_RATE // g, int(sr) // g).astype("float32")
    except ImportError:
        n = int(round(len(data) * SAMPLE_RATE / sr))
        return np.interp(np.linspace(0, len(data), n, endpoint=False),
                         np.arange(len(data)), data).astype("float32")


def save_mp3(wav_path: Path, mp3_path: Path) -> Path:
    """Encode a WAV file to MP3 (libsndfile-native, no ffmpeg needed)."""
    import soundfile as sf

    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    sf.write(str(mp3_path), data, sr, format="MP3")
    return mp3_path


def default_input_device() -> Optional[int]:
    import sounddevice as sd

    try:
        return sd.default.device[0]
    except Exception:
        return None


def find_loopback_device() -> Optional[int]:
    """Best-guess system-audio capture device (BlackHole et al.)."""
    for dev in list_devices():
        if dev["loopback"]:
            return dev["index"]
    return None


class WavTail:
    """Incrementally reads a WAV that another writer is still appending to.

    Used for live captions while recording: the mic writer (libsndfile) and the
    ScreenCaptureKit helper (AVAudioFile) both stream PCM to disk but only fix
    up the RIFF sizes on close, so we parse the header ourselves and read from
    the ``data`` chunk to the current end of file. :meth:`read` returns the new
    audio as mono float32 at 16 kHz (empty until the header is on disk).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._pos: Optional[int] = None  # byte offset of the next unread frame
        self._fmt = None                 # (dtype, channels, samplerate, block_align)

    def _parse_header(self, f) -> bool:
        import struct

        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return False
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return False
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"data":
                if fmt is None:
                    return False
                self._fmt, self._pos = fmt, f.tell()
                return True
            body = f.read(size + (size & 1))
            if len(body) < size:
                return False
            if cid == b"fmt ":
                code, ch, sr, _, align, bits = struct.unpack("<HHIIHH", body[:16])
                if code == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE
                    code = struct.unpack("<H", body[24:26])[0]
                dtype = {(1, 16): "<i2", (1, 32): "<i4", (3, 32): "<f4"}.get((code, bits))
                if dtype is None:
                    return False
                fmt = (dtype, ch, sr, align)

    def read(self) -> np.ndarray:
        empty = np.zeros(0, dtype="float32")
        if not self.path.exists():
            return empty
        with open(self.path, "rb") as f:
            if self._pos is None and not self._parse_header(f):
                return empty
            dtype, ch, sr, align = self._fmt
            f.seek(self._pos)
            raw = f.read()
        raw = raw[: len(raw) - len(raw) % align]  # only whole frames
        if not raw:
            return empty
        self._pos += len(raw)
        data = np.frombuffer(raw, dtype=dtype).astype("float32")
        if dtype == "<i2":
            data /= 32768.0
        elif dtype == "<i4":
            data /= 2147483648.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if sr != SAMPLE_RATE and len(data):
            n = int(round(len(data) * SAMPLE_RATE / sr))
            data = np.interp(
                np.linspace(0, len(data), n, endpoint=False),
                np.arange(len(data)), data,
            ).astype("float32")
        return data


class _Source:
    """Streams one input device to a temporary WAV at its native rate.

    Survives the device changing under it — typically a call app switching a
    Bluetooth headset into hands-free mode, which silently kills the stream:
    a watchdog reopens the input when callbacks stop arriving, and the new
    audio (resampled to the file's rate if needed) continues the same WAV.
    Stopping never blocks: PortAudio's stop() can hang forever on a dead
    device, so it runs on a helper thread with a deadline.
    """

    STALL_SECONDS = 2.0   # no audio callback for this long → reopen
    STOP_DEADLINE = 3.0   # give up on a hung PortAudio stop/close after this

    def __init__(self, device: int):
        self.device = device
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._watch: Optional[threading.Thread] = None
        self._stream = None
        self.path = Path(tempfile.mktemp(suffix=f"_meetre_{device}.wav"))
        self.samplerate = SAMPLE_RATE      # the WAV's rate (fixed once written)
        self._stream_rate = SAMPLE_RATE    # the live stream's rate (may change)
        self._last_cb = 0.0
        self.frames = 0
        self.reopened = 0
        self.overflows = 0     # PortAudio input overflows (dropped buffers)
        self.padded = 0.0      # s of silence inserted to stay on the clock
        self._t0: Optional[float] = None

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    # The stem must stay on the wall clock: the call stem (ScreenCaptureKit)
    # does, and the echo removal and the you/call matching line the two up
    # by time. Opened at 16 kHz, PortAudio's rate conversion delivered ~1 %
    # too few samples (25 min of mic came out 16 s short) and overflows were
    # never noticed. Now the device runs at its own rate (converted at stop)
    # and any shortfall against the clock beyond SYNC_SLACK is filled with
    # silence, so a drop or stall doesn't shift everything after it.
    SYNC_SLACK = 0.08     # s of callback jitter tolerated before padding

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        now = time.monotonic()
        self._last_cb = now
        if status and status.input_overflow:
            self.overflows += 1
        self._queue.put((self._stream_rate, now, indata.copy()))

    def _writer(self):
        import soundfile as sf

        with sf.SoundFile(
            str(self.path), mode="w", samplerate=self.samplerate,
            channels=CHANNELS, subtype="PCM_16",
        ) as f:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    rate, at, block = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._t0 is None:  # the clock starts with the first audio
                    self._t0 = at - len(block) / rate
                behind = (at - self._t0) * self.samplerate - (self.frames + len(block) * self.samplerate / rate)
                if behind > self.SYNC_SLACK * self.samplerate:
                    gap = int(behind)
                    f.write(np.zeros((gap, CHANNELS), dtype="float32"))
                    self.frames += gap
                    self.padded += gap / self.samplerate
                # Downmix any multi-channel input to mono.
                if block.ndim > 1 and block.shape[1] > 1:
                    block = block.mean(axis=1, keepdims=True)
                if rate != self.samplerate and len(block):
                    # A reopened device may run at a different rate.
                    n = max(1, int(round(len(block) * self.samplerate / rate)))
                    block = np.interp(np.linspace(0, len(block), n, endpoint=False),
                                      np.arange(len(block)), block[:, 0]
                                      ).astype("float32").reshape(-1, 1)
                f.write(block)
                self.frames += len(block)

    def _open(self, device):
        """Open (not start) an input stream on ``device``; returns (stream, rate)."""
        import sounddevice as sd

        # The device's own rate: asking PortAudio for 16 kHz made it convert
        # and lose ~1 % of the samples. _mix resamples to 16 kHz at the end.
        for rate in (None, SAMPLE_RATE):
            try:
                sr = rate or int(sd.query_devices(device)["default_samplerate"])
                stream = sd.InputStream(samplerate=sr, channels=CHANNELS,
                                        device=device, callback=self._callback)
                return stream, sr
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError(f"Could not open audio device {device}")

    def start(self):
        self._stream, self._stream_rate = self._open(self.device)
        self.samplerate = self._stream_rate
        self._last_cb = time.monotonic()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        self._stream.start()
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()

    def _watchdog(self):
        import sys

        while not self._stop.wait(0.5):
            if time.monotonic() - self._last_cb < self.STALL_SECONDS:
                continue
            # The stream died (device reconfigured / switched). Drop it
            # without waiting on it, then reopen: same device first, else the
            # current default input after PortAudio rescans the devices.
            old, self._stream = self._stream, None
            _close_quietly(old, self.STOP_DEADLINE)
            for attempt in ("same", "default"):
                try:
                    if attempt == "default":
                        import sounddevice as sd

                        sd._terminate()
                        sd._initialize()
                        dev = sd.default.device[0]
                    else:
                        dev = self.device
                    stream, rate = self._open(dev)
                    self._stream_rate = rate
                    self._last_cb = time.monotonic()
                    stream.start()
                    self._stream = stream
                    self.reopened += 1
                    print(f"meetre: mic stream stalled — reopened ({attempt} device {dev}, "
                          f"{rate} Hz)", file=sys.stderr)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"meetre: mic reopen ({attempt}) failed: {e}", file=sys.stderr)
            else:
                self._last_cb = time.monotonic()  # retry after another stall window

    def stop(self):
        # Finish the WAV first, from what's already queued — never wait on the
        # device: a dead stream's stop() can block forever.
        self._stop.set()
        old, self._stream = self._stream, None
        _close_quietly(old, self.STOP_DEADLINE)
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None


def _close_quietly(stream, deadline: float) -> None:
    """stop()+close() a PortAudio stream on a helper thread; abandon it if
    that hangs (dead device) instead of hanging the caller."""
    if stream is None:
        return
    import sys

    def run():
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=run, daemon=True, name="meetre-stream-close")
    t.start()
    t.join(timeout=deadline)
    if t.is_alive():
        print("meetre: audio stream didn't stop in time — abandoned it", file=sys.stderr)


class _SystemAudioSource:
    """Captures macOS system audio natively via the ScreenCaptureKit helper.

    Runs the compiled Swift ``syscap`` binary as a subprocess writing a WAV,
    then exposes the same ``path`` / ``seconds`` / ``start`` / ``stop`` surface
    as :class:`_Source` so :class:`Recorder` can treat both uniformly.
    """

    def __init__(self):
        self.path = Path(tempfile.mktemp(suffix="_meetre_system.wav"))
        self.samplerate = SAMPLE_RATE
        self._proc: Optional[subprocess.Popen] = None
        self._start_t: Optional[float] = None
        self._ready = threading.Event()
        self._log: List[str] = []

    @property
    def seconds(self) -> float:
        if self._start_t is None or not self._ready.is_set():
            return 0.0
        return max(0.0, time.monotonic() - self._start_t)

    def _watch(self):
        import sys

        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._log.append(line)
            if line.strip() == "READY":
                self._ready.set()
            elif line.strip():
                print(f"meetre: {self._binary_name}: {line.strip()}", file=sys.stderr, flush=True)

    _binary_name = "syscap"

    def _binary(self):
        from . import sysaudio

        return sysaudio.helper_path()  # compiles on first use

    def start(self):
        binary = self._binary()
        self._proc = subprocess.Popen(
            [str(binary), str(self.path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._start_t = time.monotonic()
        threading.Thread(target=self._watch, daemon=True).start()
        # Wait for the helper to confirm capture started. The first run may
        # stall on the Screen Recording permission prompt, so allow some slack.
        if not self._ready.wait(timeout=20) and self._proc.poll() is not None:
            raise RuntimeError(
                "".join(self._log).strip()
                or "system-audio helper exited before capturing (permission denied?)"
            )

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()  # SIGTERM → helper finalises the WAV
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None


class _VoiceMicSource(_SystemAudioSource):
    """The microphone through macOS Voice Processing (``miccap`` helper).

    Apple's echo canceller removes what the Mac plays from its speakers — the
    other side of a call — from the mic. Measured with speakers on: the raw
    mic re-transcribes the remote sentences word for word; through Voice
    Processing nothing of them is left. Writes 16 kHz mono and survives
    device switches (the helper rebuilds itself).
    """

    _binary_name = "miccap"

    def __init__(self):
        super().__init__()
        self.path = Path(tempfile.mktemp(suffix="_meetre_mic_vp.wav"))

    def _binary(self):
        from . import sysaudio

        return sysaudio.mic_helper_path()


class Recorder:
    """Records mic + system audio concurrently and mixes to one 16 kHz file.

    ``native_system=True`` captures system audio via ScreenCaptureKit (no
    loopback driver). Otherwise a ``system_device`` index (e.g. BlackHole) is
    used. The mic and system sources record simultaneously and are mixed into
    a single track on :meth:`stop`.
    """

    def __init__(
        self,
        mic_device: Optional[int] = None,
        system_device: Optional[int] = None,
        native_system: bool = False,
        echo_cancellation: bool = False,
    ):
        if mic_device is None and system_device is None and not native_system:
            raise ValueError("At least one audio source is required")
        self._sources: list = []
        # Mark which sources are essential — if a non-essential source (system
        # audio) fails to start, we keep recording the rest rather than abort.
        self._essential: list = []
        self._mic_device = mic_device
        if mic_device is not None:
            # Voice Processing always uses the system's default input, so it
            # only replaces the plain capture when that's the device in use.
            default = default_input_device()
            src = (_VoiceMicSource() if echo_cancellation and mic_device == default
                   else _Source(mic_device))
            src.role = "mic"
            self._sources.append(src)
            self._essential.append(True)
        if native_system:
            src = _SystemAudioSource()
            src.role = "system"
            self._sources.append(src)
            self._essential.append(False)
        elif system_device is not None:
            src = _Source(system_device)
            src.role = "system"
            self._sources.append(src)
            self._essential.append(False)
        # Per-source 16 kHz stems, populated by stop(): {role: Path}. Kept so the
        # transcriber can attribute speakers by source instead of a mono mix.
        self.stems: dict = {}
        # Populated by start(): non-fatal messages about sources that dropped.
        self.start_errors: List[str] = []

    @property
    def seconds(self) -> float:
        return max((s.seconds for s in self._sources), default=0.0)

    def start(self, path: Path) -> None:
        self._out = path
        path.parent.mkdir(parents=True, exist_ok=True)
        started: list = []
        for src, essential in zip(self._sources, self._essential):
            try:
                try:
                    src.start()
                except Exception as e:  # noqa: BLE001
                    if not isinstance(src, _VoiceMicSource):
                        raise
                    # Voice Processing unavailable: plain capture instead.
                    self.start_errors.append(f"echo cancellation unavailable ({e}); using the plain microphone")
                    src = _Source(self._mic_device)
                    src.role = "mic"
                    src.start()
                started.append(src)
            except Exception as e:  # noqa: BLE001
                if essential:
                    # Roll back anything already started before re-raising.
                    for s in started:
                        try:
                            s.stop()
                        except Exception:  # noqa: BLE001
                            pass
                    raise
                self.start_errors.append(str(e))
        self._sources = started
        if not self._sources:
            raise RuntimeError("No audio source could be started")

    def live_paths(self) -> dict:
        """{role: path} of the WAVs being written right now (for live captions)."""
        return {getattr(s, "role", "source"): s.path for s in self._sources}

    def stop(self) -> Path:
        # Stop EVERY source even if one fails: a mic that errors on stop must
        # not leave the system-audio helper capturing forever, nor throw away
        # the audio that was recorded.
        import sys

        errors = []
        for src in self._sources:
            role = getattr(src, "role", "source")
            t0 = time.monotonic()
            try:
                src.stop()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{role}: {e}")
            print(f"meetre: stopped {role} in {time.monotonic() - t0:.1f}s"
                  + (f" (mic reopened {src.reopened}×)" if getattr(src, "reopened", 0) else "")
                  + (f", {src.overflows} overflows" if getattr(src, "overflows", 0) else "")
                  + (f", {src.padded:.2f}s padded to stay in sync"
                     if getattr(src, "padded", 0) else ""),
                  file=sys.stderr, flush=True)
        if errors:
            print("meetre: error stopping audio source(s): " + "; ".join(errors), file=sys.stderr)
            self.start_errors.extend(errors)
        return self._mix()

    def _mix(self) -> Path:
        import soundfile as sf

        tracks = []
        self.stems = {}
        for src in self._sources:
            if not src.path.exists():
                continue
            data, sr = sf.read(str(src.path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE and len(data):
                data = _to_16k(data, sr)
            # Save a per-source stem for source-aware diarization.
            role = getattr(src, "role", "source")
            stem = Path(tempfile.mktemp(suffix=f"_meetre_stem_{role}.wav"))
            sf.write(str(stem), data, SAMPLE_RATE, subtype="PCM_16")
            self.stems[role] = stem
            tracks.append(data)
            src.path.unlink(missing_ok=True)

        if not tracks:
            raise RuntimeError("No audio was captured")

        length = max(len(t) for t in tracks)
        mix = np.zeros(length, dtype="float32")
        for t in tracks:
            mix[: len(t)] += t
        # Prevent clipping when summing sources.
        peak = float(np.max(np.abs(mix))) if length else 0.0
        if peak > 1.0:
            mix /= peak

        sf.write(str(self._out), mix, SAMPLE_RATE, subtype="PCM_16")
        return self._out
