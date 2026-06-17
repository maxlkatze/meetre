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
) -> Tuple[List[Segment], str]:
    """Transcribe the mixed audio once, then attribute speakers from the stems.

    Timing comes from a single transcription of the mix (one clean timeline).
    Each segment is attributed to a side by comparing energy in the mic stem
    (you) vs the system stem (remote) over the segment window — the stems are
    sample-aligned with the mix. With ``detect_speakers``, pyannote runs on each
    stem separately so multiple local AND multiple remote speakers are split
    (e.g. "Ich"/"Vor Ort 2" locally, "Sprecher 1/2" remotely).
    """
    import soundfile as sf

    segments, backend = transcribe(mix_path, model, language, compute_type)
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
            mic_turns = diarize_turns(Path(stems["mic"]), hf_token) if stems.get("mic") and mic is not None else []
            _assign_side(mic_segs, mic_turns, lbl["local"], lbl["local_multi"])
        except RuntimeError:
            for s in mic_segs:
                s.speaker = lbl["local"]
        try:
            sys_turns = diarize_turns(Path(stems["system"]), hf_token,
                                      num_speakers, min_speakers, max_speakers) \
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


def diarize_turns(
    audio_path: Path,
    hf_token: Optional[str],
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[tuple]:
    """Run pyannote and return raw speaker turns ``[(start, end, label), …]``.

    Requires the ``persons`` extra and a HuggingFace token.
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
        k["weights_only"] = False  # Lightning passes weights_only=True explicitly
        return _orig_load(*a, **k)

    torch.load = _trusting_load
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
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
    diarization = pipeline(str(audio_path), **kwargs)
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
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
) -> List[Segment]:
    """Assign a "Speaker N" label to each segment using pyannote on one file."""
    turns = diarize_turns(audio_path, hf_token, num_speakers, min_speakers, max_speakers)
    label_map: dict = {}
    for seg in segments:
        raw = _best_turn(turns, seg.start, seg.end)
        if raw is not None:
            if raw not in label_map:
                label_map[raw] = f"Speaker {len(label_map) + 1}"
            seg.speaker = label_map[raw]
    return segments
