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

from .mlxthread import on_mlx_thread

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
    # Optional per-word timing from phoneme alignment: [{"word","start","end"}, …]
    words: Optional[list] = None


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
    vad: bool = False,
    word_align: bool = False,
) -> Tuple[List[Segment], str]:
    """Transcribe ``audio_path`` into timestamped segments.

    Returns ``(segments, backend_name)``. ``progress`` is an optional
    callable(seconds_done, total_seconds) for UI updates. ``vad`` drops
    silence-only (hallucinated) segments; ``word_align`` attaches per-word
    timestamps via phoneme forced alignment. Both apply to the Whisper path.
    """
    if is_parakeet(model):
        return _transcribe_parakeet(audio_path, model, progress), "parakeet-tdt-v3"
    backend = available_backend()
    if backend == "mlx-whisper":
        segs = _transcribe_mlx(audio_path, model, language, progress)
    elif backend == "faster-whisper":
        # faster-whisper already VAD-filters internally (vad_filter=True).
        segs = _transcribe_faster(audio_path, model, language, compute_type, progress)
    else:
        raise RuntimeError(
            "No transcription backend installed. Install one:\n"
            "  pip install mlx-whisper      # Apple Silicon (recommended)\n"
            "  pip install faster-whisper   # Intel / fallback"
        )
    segs = _post_process(segs, audio_path, vad and backend == "mlx-whisper", word_align)
    return segs, backend


def _post_process(segments, audio_path, vad: bool, word_align: bool):
    """Optional VAD silence filtering + phoneme word-alignment (best-effort)."""
    if not segments or not (vad or word_align):
        return segments
    from . import align

    if vad:
        try:
            segments = align.filter_silence(segments, audio_path)
        except Exception:  # noqa: BLE001
            pass
    if word_align:
        try:
            segments = align.align_words(segments, audio_path)
        except Exception:  # noqa: BLE001
            pass
    return segments


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


@on_mlx_thread
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


def live_supported(model: Optional[str]) -> bool:
    """Live (while-recording) transcription needs an MLX backend."""
    return is_parakeet(model) or available_backend() == "mlx-whisper"


@on_mlx_thread
def transcribe_live(audio, model: str, language: Optional[str] = None,
                    prompt: Optional[str] = None) -> List[Segment]:
    """Transcribe a short in-memory 16 kHz mono window (live captions).

    Tuned for latency: greedy decoding only (no temperature fallback), no
    conditioning across windows — ``prompt`` (the recent committed text) gives
    continuity instead. Timestamps are relative to the window start.
    """
    import numpy as np

    audio = np.ascontiguousarray(audio, dtype="float32")
    if NORMALIZE_FOR_ASR:
        audio = normalize_for_asr(audio, 16_000)
    if is_parakeet(model):
        import tempfile

        import soundfile as sf

        pk = load_parakeet(PARAKEET_MODELS.get(model, model))
        with tempfile.NamedTemporaryFile(suffix="_meetre_live.wav") as tmp:
            sf.write(tmp.name, audio, 16_000, subtype="FLOAT")
            result = pk.transcribe(tmp.name)
        return [Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in result.sentences if s.text.strip()]

    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MLX_REPOS.get(model, MLX_REPOS["base"]),
        language=language,
        initial_prompt=prompt or _initial_prompt(language),
        condition_on_previous_text=False,
        temperature=0.0,
        word_timestamps=False,
    )
    return [Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result.get("segments", []) if s["text"].strip()]


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
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "parakeet-mlx is missing — reinstall meetre to pick it up: "
            "pip install -e . (or rerun install.sh)"
        ) from e
    # parakeet-mlx decodes audio with ffmpeg; meetre ships none, so route its
    # file loading through soundfile instead.
    parakeet_mlx.parakeet.load_audio = _parakeet_load_audio
    if repo not in _parakeet_cache:
        _parakeet_cache[repo] = from_pretrained(repo)
    return _parakeet_cache[repo]


@on_mlx_thread
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
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip(),
                words=_parakeet_words(s.tokens))
        for s in result.sentences
    ]
    if not out and result.text.strip():
        out.append(Segment(start=0.0, end=0.0, text=result.text.strip()))
    return out


def _parakeet_words(tokens) -> Optional[list]:
    """Merge Parakeet's sub-word tokens into timed words (a token that starts
    with a space begins a new word; the space is kept for re-joining)."""
    words: list = []
    for t in tokens or []:
        text = str(t.text)
        if words and not text.startswith(" "):
            words[-1]["word"] += text
            words[-1]["end"] = float(t.end)
        else:
            words.append({"word": text, "start": float(t.start), "end": float(t.end)})
    return words or None


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
    vad: bool = False,
    word_align: bool = False,
    merged_analysis: bool = False,
    on_transcribed=None,
) -> Tuple[List[Segment], str]:
    """Transcribe each source on its own, then split each side into people.

    The mic stem (you) and the system stem (the call) are transcribed
    separately — like the live captions — so every segment's side is simply
    its source. Comparing stem energies on a single transcription of the mix
    did not work: the call stem is digital silence between words, so its
    "noise floor" is ~0 and it out-shouts the mic on nearly every frame. Mic
    segments are kept only where VAD hears speech, and mic segments that repeat
    what the call said at about the same time (speakers → mic echo) are dropped.
    The stems are sample-aligned with the mix, so their timings share one
    timeline.

    With ``detect_speakers`` and ``merged_analysis=False``, pyannote runs on each
    stem separately so multiple local AND multiple remote speakers are split
    (e.g. "Ich"/"Vor Ort 2" locally, "Sprecher 1/2" remotely). With
    ``merged_analysis=True`` (the default in config) pyannote runs once on the
    mixed audio and the same turns are reused for both sides — one analysis
    instead of two.

    Either way, a source that never exceeds a low decibel level (essentially
    silent throughout) is skipped: it gets no segments and no analysis pass.

    ``progress`` is ``callable(seconds_done, total_seconds)`` for transcription;
    ``diar_progress`` is ``callable(label, fraction)`` for the pyannote steps.
    ``on_transcribed(segments)`` is called once segments carry side labels
    (you / remote), *before* the slow per-person analysis, so callers can show
    the transcript early; the same segments are then relabelled in place.
    """
    import numpy as np
    import soundfile as sf

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

    # Drop sources that never exceed a low decibel level (peak below ~-45 dBFS):
    # they captured nothing useful, so they get no segments and no analysis pass.
    silence_floor = 10 ** (-45 / 20)

    def _silent(arr):
        return arr is None or float(np.max(np.abs(arr))) < silence_floor

    if _silent(mic):
        mic = None
    if _silent(system):
        system = None

    # 1) Who spoke: each source is transcribed as its own stream, so a
    #    segment's side is its source. With one side silent, the mix is
    #    transcribed and everything falls to the side that recorded.
    if mic is not None and system is not None:
        from .align import speech_regions_array
        from .attribution import drop_echo

        def _part(k):  # progress of source k (of 2) as one overall bar
            if progress is None:
                return None
            return lambda done, total: progress(k * total + done, 2 * total)

        sys_segs, backend = transcribe(Path(stems["system"]), model, language, compute_type,
                                       progress=_part(0), vad=vad, word_align=word_align)
        mic_segs, backend = transcribe(Path(stems["mic"]), model, language, compute_type,
                                       progress=_part(1), vad=vad, word_align=word_align)
        # The mic has faint residual sound that ASR happily "hears" words in;
        # keep only segments where VAD finds speech (as the live captions do).
        regions = speech_regions_array(mic)
        if regions is not None:
            mic_segs = [g for g in mic_segs
                        if any(min(g.end, e) - max(g.start, b) > 0.1 for b, e in regions)]
        mic_segs = drop_echo(mic_segs, sys_segs)
        for s in sys_segs:
            s.speaker = lbl["remote"]
        for s in mic_segs:
            s.speaker = lbl["local"]
        segments = sorted(sys_segs + mic_segs, key=lambda s: (s.start, s.end))
    else:
        segments, backend = transcribe(mix_path, model, language, compute_type,
                                       progress=progress, vad=vad, word_align=word_align)
        label = lbl["remote"] if mic is None else lbl["local"]
        for s in segments:
            s.speaker = label
        mic_segs = segments if mic is not None else []
        sys_segs = segments if mic is None else []
    if not segments:
        return segments, backend
    if on_transcribed is not None:
        on_transcribed(segments)

    # 2) Split each side into individual people (side labels stay otherwise).
    if detect_speakers and merged_analysis:
        # One analysis on the mixed audio; reuse the same turns for both sides.
        try:
            turns = diarize_turns(mix_path, hf_token, num_speakers, min_speakers,
                                  max_speakers, progress=diar_progress)
        except RuntimeError:
            turns = []
        if turns:
            _assign_side(mic_segs, turns, lbl["local"], lbl["local_multi"])
            _assign_side(sys_segs, turns, lbl["remote"], lbl["remote_multi"])
    elif detect_speakers:
        # Per-stem analysis (pyannote on each source separately).
        try:
            mic_turns = diarize_turns(Path(stems["mic"]), hf_token, progress=diar_progress) \
                if mic is not None else []
            _assign_side(mic_segs, mic_turns, lbl["local"], lbl["local_multi"])
        except RuntimeError:
            pass
        try:
            sys_turns = diarize_turns(Path(stems["system"]), hf_token,
                                      num_speakers, min_speakers, max_speakers,
                                      progress=diar_progress) \
                if system is not None else []
            _assign_side(sys_segs, sys_turns, lbl["remote"], lbl["remote_multi"])
        except RuntimeError:
            pass

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


_pipeline_cache: dict = {}


def _load_pipeline(Pipeline, hf_token: str):
    """Load the diarization pipeline once per process, on the GPU when possible.

    pyannote defaults to the CPU; on Apple Silicon moving it to MPS (Metal) is
    ~12x faster (10 min of audio: ~275 s on CPU vs ~23 s on MPS, M5 Pro).
    """
    import torch

    cached = _pipeline_cache.get(hf_token)
    if cached is not None:
        return cached

    # torch>=2.6 defaults torch.load to weights_only=True, which rejects
    # pyannote's checkpoints. They come from a trusted source, so load them
    # with weights_only=False for the duration of pipeline construction.
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

    try:
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except Exception:  # noqa: BLE001
        pass
    _pipeline_cache[hf_token] = pipeline
    return pipeline


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

    import torch

    pipeline = _load_pipeline(Pipeline, hf_token)

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
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)
    except Exception:  # noqa: BLE001
        if str(getattr(pipeline, "device", "cpu")) == "cpu":
            raise
        # An op the GPU (MPS) backend can't run: retry once on the CPU.
        pipeline.to(torch.device("cpu"))
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


# A diarization cluster only counts as its own person on a side when it holds
# a real share of that side's speech; otherwise its few segments are boundary
# spill-over (e.g. the other side's voice cluster leaking in around turn
# changes when analysing the mix) and get folded into the side's main voice.
_MIN_CLUSTER_SECONDS = 4.0
_MIN_CLUSTER_SHARE = 0.10


def _assign_side(side_segs: List[Segment], turns: List[tuple],
                 single_label: str, multi_label: str) -> None:
    """Label segments on one side; use a numbered scheme only if >1 speaker."""
    raws = [_best_turn(turns, s.start, s.end) for s in side_segs]
    talk: dict = {}
    for seg, r in zip(side_segs, raws):
        if r is not None:
            talk[r] = talk.get(r, 0.0) + max(seg.end - seg.start, 0.0)
    total = sum(talk.values())
    real = [r for r, t in talk.items()
            if t >= _MIN_CLUSTER_SECONDS and t >= _MIN_CLUSTER_SHARE * total]
    if len(real) <= 1:
        for s in side_segs:
            s.speaker = single_label
        return
    # Number the real voices in order of first appearance; fold spill-over
    # clusters (and unmatched segments) into the nearest real voice in time.
    first = {}
    for seg, r in zip(side_segs, raws):
        if r in real and r not in first:
            first[r] = seg.start
    order = {r: i + 1 for i, r in enumerate(sorted(real, key=first.get))}
    prev = None
    labels = []
    for r in raws:
        labels.append(r if r in order else prev)
        prev = labels[-1] or prev
    nxt = None
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] is None:
            labels[i] = nxt
        nxt = labels[i] or nxt
    for seg, r in zip(side_segs, labels):
        seg.speaker = f"{multi_label} {order.get(r, 1)}"


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
    # Split sentences at speaker changes using word timings, so a sentence
    # that spans two people isn't given entirely to one of them.
    from .attribution import split_by_turns

    segments = split_by_turns(segments, turns)
    label_map: dict = {}
    for seg in segments:
        raw = _best_turn(turns, seg.start, seg.end)
        if raw is not None:
            if raw not in label_map:
                label_map[raw] = f"Speaker {len(label_map) + 1}"
            seg.speaker = label_map[raw]
    return segments
