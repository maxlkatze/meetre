"""Speech-to-text plus optional speaker diarization.

Backends, tried in order of preference:

1. **mlx-whisper** — Apple's MLX framework, Metal-accelerated. The fastest
   option on Apple Silicon (M-series) and the default on this Mac.
2. **faster-whisper** — CTranslate2 CPU/GPU backend. Portable fallback for
   Intel Macs or when MLX isn't installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# pyannote.audio 4.x ships OpenTelemetry usage metrics that phone home. meetre is
# local-only, so disable them before pyannote is imported (setdefault keeps it
# overridable for anyone who explicitly opts in).
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

# Map friendly model sizes to the HuggingFace repos MLX uses.
# large-v3-turbo is the recommended default on Apple Silicon: near-large-v3
# accuracy (~13.4% vs 13.2% WER) at 4–6× the speed (14–18× real-time on M5).
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


# Seeding the decoder with a well-punctuated German sentence nudges Whisper
# toward correct German noun capitalisation, umlauts and punctuation.
_INITIAL_PROMPTS = {
    "de": "Willkommen zum Meeting. Hier ist das Wortprotokoll der Besprechung "
          "mit korrekter Groß- und Kleinschreibung, Umlauten und Satzzeichen.",
}


def _initial_prompt(language: Optional[str]) -> Optional[str]:
    return _INITIAL_PROMPTS.get(language) if language else None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


# Alternative backend: NVIDIA Parakeet TDT v3 on MLX (multilingual, very fast,
# supports streaming). Selectable as the "parakeet-tdt-v3" model.
PARAKEET_MODELS = {"parakeet-tdt-v3": "mlx-community/parakeet-tdt-0.6b-v3"}


def is_parakeet(model: Optional[str]) -> bool:
    return bool(model) and (model in PARAKEET_MODELS or model.startswith("parakeet"))


def mlx_repo(model: str) -> str:
    """The HuggingFace repo for a given model (whisper size or parakeet)."""
    if model in PARAKEET_MODELS:
        return PARAKEET_MODELS[model]
    return MLX_REPOS.get(model, MLX_REPOS["base"])


def available_backend() -> Optional[str]:
    """Return the best transcription backend present, or None."""
    try:
        import mlx_whisper  # noqa: F401

        return "mlx-whisper"
    except ImportError:
        pass
    try:
        import faster_whisper  # noqa: F401

        return "faster-whisper"
    except ImportError:
        return None


def transcribe(
    audio_path: Path,
    model: str = "base",
    language: Optional[str] = None,
    compute_type: str = "int8",
    progress=None,
) -> Tuple[List[Segment], str]:
    """Transcribe ``audio_path`` into timestamped segments.

    Returns ``(segments, backend_name)``. ``progress`` is an optional
    callable(seconds_done, total_seconds) for UI updates.
    """
    if is_parakeet(model):
        return _transcribe_parakeet(audio_path, model, progress), "parakeet-tdt-v3"
    backend = available_backend()
    if backend == "mlx-whisper":
        return _transcribe_mlx(audio_path, model, language, progress), backend
    if backend == "faster-whisper":
        return _transcribe_faster(audio_path, model, language, compute_type, progress), backend
    raise RuntimeError(
        "No transcription backend installed. Install one:\n"
        "  pip install mlx-whisper      # Apple Silicon (recommended)\n"
        "  pip install faster-whisper   # Intel / fallback"
    )


# Even out loudness before transcription so a soft-spoken or distant participant
# is brought up to the level of louder speakers. This is applied only to the
# audio fed to Whisper — the saved recording and the per-stem energy used for
# speaker attribution stay untouched.
NORMALIZE_FOR_ASR = True


def normalize_for_asr(data, sr=16_000):
    """Gated automatic gain control: boost quiet speech and tame loud peaks,
    without amplifying silence/noise (which makes Whisper hallucinate).

    Compresses the dynamic range frame by frame with a smoothed, clamped gain
    curve, then peak-normalises into headroom. Returns float32 in [-1, 1].
    """
    import numpy as np

    n = len(data)
    if n == 0:
        return np.asarray(data, dtype="float32")
    data = (np.asarray(data, dtype="float32") - float(np.mean(data)))
    if float(np.max(np.abs(data))) < 1e-5:
        return data  # essentially silence — leave it alone

    frame = max(1, int(sr * 0.02))                 # 20 ms frames
    pad = (-n) % frame
    padded = np.concatenate([data, np.zeros(pad, dtype="float32")]) if pad else data
    frames = padded.reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    # Noise floor: just above the quietest frames so hiss isn't pumped up.
    floor = max(float(np.percentile(rms, 15)) * 1.5, 10 ** (-45 / 20))
    target, max_gain, min_gain = 0.12, 8.0, 0.4    # ~-18 dBFS, +18 dB / -8 dB

    gain = np.clip(target / np.maximum(rms, 1e-6), min_gain, max_gain)
    quiet = rms <= floor
    gain[quiet] = np.minimum(gain[quiet], 1.0)      # never boost the noise floor

    # Smooth the gain curve (~150 ms) so it doesn't pump between frames.
    win = max(1, int(0.15 / 0.02))
    if win > 1:
        kernel = np.ones(win, dtype="float32") / win
        gain = np.convolve(gain, kernel, mode="same")

    out = data * np.repeat(gain, frame)[:n]
    out_peak = float(np.max(np.abs(out)))
    if out_peak > 0:
        out *= 0.97 / out_peak                      # use headroom, avoid clipping
    return np.clip(out, -1.0, 1.0).astype("float32")


def _load_audio_16k(audio_path):
    """Read a WAV into a mono float32 array at 16 kHz, without needing ffmpeg."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16_000 and len(data):
        n = int(round(len(data) * 16_000 / sr))
        data = np.interp(
            np.linspace(0, len(data), n, endpoint=False),
            np.arange(len(data)), data,
        ).astype("float32")
    if NORMALIZE_FOR_ASR:
        data = normalize_for_asr(data, 16_000)
    return np.ascontiguousarray(data, dtype="float32")


def _transcribe_mlx(audio_path, model, language, progress) -> List[Segment]:
    import mlx_whisper

    repo = MLX_REPOS.get(model, MLX_REPOS["base"])
    # Decode the audio ourselves so mlx-whisper never invokes ffmpeg.
    audio = _load_audio_16k(audio_path)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        language=language,
        initial_prompt=_initial_prompt(language),
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    segs = result.get("segments", [])
    total = segs[-1]["end"] if segs else 0.0
    out: List[Segment] = []
    for s in segs:
        out.append(Segment(start=s["start"], end=s["end"], text=s["text"].strip()))
        if progress is not None:
            progress(s["end"], total)
    return out


def _parakeet_load_audio(filename, sampling_rate, dtype=None):
    """Replacement for parakeet-mlx's ffmpeg-only loader: read via soundfile and
    resample to the model's rate, returning a normalised mono mx float32 array."""
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(filename), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sampling_rate and len(data):
        n = int(round(len(data) * sampling_rate / sr))
        data = np.interp(
            np.linspace(0, len(data), n, endpoint=False), np.arange(len(data)), data
        ).astype("float32")
    return mx.array(np.ascontiguousarray(data, dtype="float32"))


_parakeet_cache: dict = {}


def load_parakeet(repo: str):
    """Load (and cache) a Parakeet model, patching out its ffmpeg dependency."""
    try:
        import parakeet_mlx
        from parakeet_mlx import from_pretrained
    except ImportError as e:  # pragma: no cover - optional extra
        raise RuntimeError(
            "Parakeet needs the 'parakeet' extra: pip install 'meetre[parakeet]'"
        ) from e
    # parakeet-mlx decodes audio with ffmpeg; meetre ships none, so route its
    # file loading through soundfile instead.
    parakeet_mlx.parakeet.load_audio = _parakeet_load_audio
    if repo not in _parakeet_cache:
        _parakeet_cache[repo] = from_pretrained(repo)
    return _parakeet_cache[repo]


def _transcribe_parakeet(audio_path, model, progress=None) -> List[Segment]:
    repo = PARAKEET_MODELS.get(model, model)
    pk = load_parakeet(repo)

    cb = None
    if progress is not None:
        def cb(done, total):  # parakeet reports (samples_done, total_samples)
            progress(done, total)

    # chunk_duration enables progress + bounds memory on long meetings.
    result = pk.transcribe(str(audio_path), chunk_duration=120.0, chunk_callback=cb)
    out: List[Segment] = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in result.sentences
    ]
    if not out and result.text.strip():
        out.append(Segment(start=0.0, end=0.0, text=result.text.strip()))
    return out


def _transcribe_faster(audio_path, model, language, compute_type, progress) -> List[Segment]:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device="cpu", compute_type=compute_type)
    seg_iter, info = whisper.transcribe(
        str(audio_path), language=language, vad_filter=True,
        initial_prompt=_initial_prompt(language),
    )
    total = info.duration or 0.0
    out: List[Segment] = []
    for s in seg_iter:
        out.append(Segment(start=s.start, end=s.end, text=s.text.strip()))
        if progress is not None:
            progress(s.end, total)
    return out


def _labels(language: Optional[str]) -> dict:
    """Speaker label vocabulary for a language.

    ``local``/``remote`` are used when a side has a single speaker;
    ``local_multi``/``remote_multi`` are numbered when a side has several.
    """
    if language == "de":
        return {"local": "Ich", "local_multi": "Vor Ort",
                "remote": "Sprecher", "remote_multi": "Sprecher"}
    return {"local": "Me", "local_multi": "Local",
            "remote": "Speaker", "remote_multi": "Speaker"}


def _rms(arr) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def transcribe_attributed(
    mix_path: Path,
    stems: dict,
    model: str = "base",
    language: Optional[str] = None,
    compute_type: str = "int8",
    detect_speakers: bool = False,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress=None,
    diar_progress=None,
) -> Tuple[List[Segment], str]:
    """Transcribe the mixed audio once, then attribute speakers from the stems.

    Timing comes from a single transcription of the mix (one clean timeline).
    Each segment is attributed to a side by comparing energy in the mic stem
    (you) vs the system stem (remote) over the segment window — the stems are
    sample-aligned with the mix. With ``detect_speakers``, pyannote runs on each
    stem separately so multiple local AND multiple remote speakers are split
    (e.g. "Ich"/"Vor Ort 2" locally, "Sprecher 1/2" remotely).

    ``progress`` is ``callable(seconds_done, total_seconds)`` for transcription;
    ``diar_progress`` is ``callable(label, fraction)`` for the pyannote steps.
    """
    import soundfile as sf

    segments, backend = transcribe(mix_path, model, language, compute_type, progress=progress)
    if not segments:
        return segments, backend

    lbl = _labels(language)

    def _load(p):
        if not p or not Path(p).exists():
            return None
        data, _ = sf.read(str(p), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data  # stems are already 16 kHz mono

    mic = _load(stems.get("mic"))
    system = _load(stems.get("system"))

    def _energy(arr, start, end):
        if arr is None:
            return 0.0
        a, b = max(0, int(start * 16_000)), min(len(arr), int(end * 16_000))
        return _rms(arr[a:b]) if b > a else 0.0

    # 1) Attribute each segment to a side (you vs. remote) by which stem is
    #    louder over the segment window.
    mic_segs, sys_segs = [], []
    for seg in segments:
        if _energy(mic, seg.start, seg.end) >= _energy(system, seg.start, seg.end):
            mic_segs.append(seg)
        else:
            sys_segs.append(seg)

    # 2) Either split each side into individual people (pyannote per stem —
    #    handles multiple local AND multiple remote speakers), or just label
    #    the side.
    if detect_speakers:
        try:
            mic_turns = diarize_turns(Path(stems["mic"]), hf_token, progress=diar_progress) \
                if stems.get("mic") and mic is not None else []
            _assign_side(mic_segs, mic_turns, lbl["local"], lbl["local_multi"])
        except RuntimeError:
            for s in mic_segs:
                s.speaker = lbl["local"]
        try:
            sys_turns = diarize_turns(Path(stems["system"]), hf_token,
                                      num_speakers, min_speakers, max_speakers,
                                      progress=diar_progress) \
                if stems.get("system") and system is not None else []
            _assign_side(sys_segs, sys_turns, lbl["remote"], lbl["remote_multi"])
        except RuntimeError:
            for s in sys_segs:
                s.speaker = lbl["remote"]
    else:
        for s in mic_segs:
            s.speaker = lbl["local"]
        for s in sys_segs:
            s.speaker = lbl["remote"]

    return segments, backend


def diarization_ready(hf_token: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Return (ready, reason_if_not) for speaker detection."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False, "pyannote not installed (pip install 'meetre[persons]')"
    if not hf_token:
        return False, "no HuggingFace token set (meetre config hf_token <token>)"
    return True, None


# pyannote diarization pipeline. "community-1" (ships with pyannote.audio 4.x)
# is markedly more accurate than the older 3.1 pipeline and runs fully on-device.
# We deliberately do NOT use the cloud "precision-2" variant (it uploads audio).
_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Friendlier labels for pyannote's internal step names (shown on the progress bar).
_DIAR_STEPS = {
    "segmentation": "Speakers: segmenting",
    "speaker_counting": "Speakers: counting",
    "embeddings": "Speakers: analyzing voices",
    "discrete_diarization": "Speakers: assigning",
}


def _pyannote_hook(progress):
    """A pyannote-compatible hook that forwards step progress to ``progress``.

    pyannote calls ``hook(step_name, step_artifact=None, file=None, total=None,
    completed=None)`` as each internal step runs (same protocol as the library's
    ProgressHook, but routed to the menu bar instead of the console). We turn it
    into ``progress(label, fraction)`` — ``fraction`` is None when a step has no
    determinate total. Returns None if ``progress`` is None (so the pipeline
    just runs without a hook).
    """
    if progress is None:
        return None

    def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
        label = _DIAR_STEPS.get(step_name) or "Detecting speakers"
        if total:
            progress(label, max(0.0, min(1.0, (completed or 0) / total)))
        else:
            progress(label, None)

    return hook


def diarize_turns(
    audio_path: Path,
    hf_token: Optional[str],
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress=None,
) -> List[tuple]:
    """Run pyannote and return raw speaker turns ``[(start, end, label), …]``.

    Requires the ``persons`` extra and a HuggingFace token. ``progress`` is an
    optional ``callable(label, fraction)`` fed live by pyannote's step hook.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Person detection needs the 'persons' extra: pip install 'meetre[persons]'"
        ) from e

    # Belt-and-suspenders: also disable telemetry via the API in case pyannote
    # was imported before our env var took effect.
    try:
        from pyannote.audio.telemetry.metrics import set_telemetry_metrics

        set_telemetry_metrics(False)
    except Exception:  # noqa: BLE001
        pass

    if not hf_token:
        raise RuntimeError(
            "Person detection needs a HuggingFace token. Set it via "
            "`meetre config` (hf_token) after accepting the pyannote model terms."
        )

    # torch>=2.6 defaults torch.load to weights_only=True, which rejects
    # pyannote's checkpoints. They come from a trusted source, so load them
    # with weights_only=False for the duration of pipeline construction.
    import torch

    _orig_load = torch.load

    def _trusting_load(*a, **k):
        k["weights_only"] = False  # Lightning passes weights_only=True explicitly
        return _orig_load(*a, **k)

    torch.load = _trusting_load
    try:
        # pyannote.audio 4.x takes ``token=``; 3.x takes ``use_auth_token=``.
        try:
            pipeline = Pipeline.from_pretrained(_DIARIZATION_MODEL, token=hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(_DIARIZATION_MODEL, use_auth_token=hf_token)
    finally:
        torch.load = _orig_load

    # Exact count wins; otherwise pass whichever range bounds are set.
    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
    hook = _pyannote_hook(progress)
    # Feed pyannote a pre-loaded waveform instead of a path: pyannote.audio 4.x
    # decodes files via torchcodec (needs FFmpeg), which meetre doesn't ship.
    # We already read audio with soundfile, so hand it the tensor directly.
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(data, dtype="float32")).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)
    # pyannote.audio 4.x (community-1) returns a result object exposing the
    # Annotation under `.speaker_diarization`; 3.x returned the Annotation
    # directly. Handle both.
    annotation = getattr(output, "speaker_diarization", output)
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _best_turn(turns: List[tuple], start: float, end: float) -> Optional[str]:
    best, best_overlap = None, 0.0
    for ts, te, spk in turns:
        overlap = min(end, te) - max(start, ts)
        if overlap > best_overlap:
            best_overlap, best = overlap, spk
    return best


def _assign_side(side_segs: List[Segment], turns: List[tuple],
                 single_label: str, multi_label: str) -> None:
    """Label segments on one side; use a numbered scheme only if >1 speaker."""
    raws = [_best_turn(turns, s.start, s.end) for s in side_segs]
    uniq = list(dict.fromkeys(r for r in raws if r is not None))
    if len(uniq) <= 1:
        for s in side_segs:
            s.speaker = single_label
    else:
        order = {u: i + 1 for i, u in enumerate(uniq)}
        for s, r in zip(side_segs, raws):
            s.speaker = f"{multi_label} {order[r]}" if r in order else f"{multi_label} 1"


def diarize(
    audio_path: Path,
    segments: List[Segment],
    hf_token: Optional[str],
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress=None,
) -> List[Segment]:
    """Assign a "Speaker N" label to each segment using pyannote on one file."""
    turns = diarize_turns(audio_path, hf_token, num_speakers, min_speakers,
                          max_speakers, progress=progress)
    label_map: dict = {}
    for seg in segments:
        raw = _best_turn(turns, seg.start, seg.end)
        if raw is not None:
            if raw not in label_map:
                label_map[raw] = f"Speaker {len(label_map) + 1}"
            seg.speaker = label_map[raw]
    return segments
