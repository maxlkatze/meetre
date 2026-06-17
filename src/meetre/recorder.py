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


class _Source:
    """Streams one input device to a temporary WAV at its native rate."""

    def __init__(self, device: int):
        self.device = device
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self.path = Path(tempfile.mktemp(suffix=f"_meetre_{device}.wav"))
        self.samplerate = SAMPLE_RATE
        self.frames = 0

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        self._queue.put(indata.copy())

    def _writer(self):
        import soundfile as sf

        with sf.SoundFile(
            str(self.path), mode="w", samplerate=self.samplerate,
            channels=CHANNELS, subtype="PCM_16",
        ) as f:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    block = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                # Downmix any multi-channel input to mono.
                if block.ndim > 1 and block.shape[1] > 1:
                    block = block.mean(axis=1, keepdims=True)
                f.write(block)
                self.frames += len(block)

    def start(self):
        import sounddevice as sd

        # Prefer 16 kHz; fall back to the device default rate if rejected.
        for rate in (SAMPLE_RATE, None):
            try:
                sr = rate or int(sd.query_devices(self.device)["default_samplerate"])
                self._stream = sd.InputStream(
                    samplerate=sr, channels=CHANNELS,
                    device=self.device, callback=self._callback,
                )
                self.samplerate = sr
                break
            except Exception:
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"Could not open audio device {self.device}")

        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


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
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._log.append(line)
            if line.strip() == "READY":
                self._ready.set()

    def start(self):
        from . import sysaudio

        binary = sysaudio.helper_path()  # compiles on first use
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
    ):
        if mic_device is None and system_device is None and not native_system:
            raise ValueError("At least one audio source is required")
        self._sources: list = []
        # Mark which sources are essential — if a non-essential source (system
        # audio) fails to start, we keep recording the rest rather than abort.
        self._essential: list = []
        if mic_device is not None:
            src = _Source(mic_device)
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

    def stop(self) -> Path:
        for src in self._sources:
            src.stop()
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
                # Linear resample to 16 kHz.
                n = int(round(len(data) * SAMPLE_RATE / sr))
                data = np.interp(
                    np.linspace(0, len(data), n, endpoint=False),
                    np.arange(len(data)), data,
                ).astype("float32")
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
