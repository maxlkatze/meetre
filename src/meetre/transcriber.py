"""Speech-to-text plus optional speaker diarization.

Backends, tried in order of preference:

1. **mlx-whisper** — Apple's MLX framework, Metal-accelerated. The fastest
   option on Apple Silicon (M-series) and the default on this Mac.
2. **faster-whisper** — CTranslate2 CPU/GPU backend. Portable fallback for
   Intel Macs or when MLX isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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


def mlx_repo(model: str) -> str:
    """The HuggingFace repo MLX uses for a given whisper model size."""
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


def diarization_ready(hf_token: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Return (ready, reason_if_not) for speaker detection."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False, "pyannote not installed (pip install 'meetre[persons]')"
    if not hf_token:
        return False, "no HuggingFace token set (meetre config hf_token <token>)"
    return True, None


def diarize(
    audio_path: Path,
    segments: List[Segment],
    hf_token: Optional[str],
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Segment]:
    """Assign a speaker label to each segment using pyannote.

    Requires the ``persons`` extra (``pip install 'meetre[persons]'``) and a
    HuggingFace token with access to ``pyannote/speaker-diarization-3.1``.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Person detection needs the 'persons' extra: pip install 'meetre[persons]'"
        ) from e

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
        # Force it off — Lightning passes weights_only=True explicitly.
        k["weights_only"] = False
        return _orig_load(*a, **k)

    torch.load = _trusting_load
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
    finally:
        torch.load = _orig_load

    # Exact count wins; otherwise pass whichever range bounds are set so
    # pyannote estimates the speaker count within them.
    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
    diarization = pipeline(str(audio_path), **kwargs)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    def best_speaker(seg: Segment) -> Optional[str]:
        best, best_overlap = None, 0.0
        for ts, te, spk in turns:
            overlap = min(seg.end, te) - max(seg.start, ts)
            if overlap > best_overlap:
                best_overlap, best = overlap, spk
        return best

    # Normalise pyannote's SPEAKER_00 → "Speaker 1" for readability.
    label_map: dict = {}
    for seg in segments:
        raw = best_speaker(seg)
        if raw is not None:
            if raw not in label_map:
                label_map[raw] = f"Speaker {len(label_map) + 1}"
            seg.speaker = label_map[raw]
    return segments
