"""Voice-activity detection and phoneme forced-alignment for the Whisper path.

Two optional post-processing passes over Whisper's segments, both best-effort
(any failure leaves the segments untouched) and run fully on-device:

* **VAD** (Silero) — drops segments that fall entirely in silence, the usual
  source of Whisper "hallucinations" during quiet stretches.
* **Phoneme alignment** (torchaudio MMS_FA forced alignment) — attaches exact
  per-word start/end times to each segment (``Segment.words``).

These need torch/torchaudio + silero-vad (the ``align`` extra). Models download
once on first use and are cached for the process.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

_SR = 16_000
_vad_model = None
_fa = None  # (model, tokenizer, aligner)


def _load_audio(audio_path) -> "object":
    """Raw mono float32 at 16 kHz (no AGC — VAD/alignment want the real levels)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != _SR and len(data):
        n = int(round(len(data) * _SR / sr))
        data = np.interp(
            np.linspace(0, len(data), n, endpoint=False), np.arange(len(data)), data
        ).astype("float32")
    return np.ascontiguousarray(data, dtype="float32")


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def speech_regions(audio_path) -> List[tuple]:
    """Speech intervals ``[(start_s, end_s), …]`` via Silero VAD ([] on failure)."""
    global _vad_model
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if _vad_model is None:
            _vad_model = load_silero_vad()
        audio = _load_audio(audio_path)
        ts = get_speech_timestamps(
            torch.from_numpy(audio), _vad_model, sampling_rate=_SR, return_seconds=True
        )
        return [(float(t["start"]), float(t["end"])) for t in ts]
    except Exception:  # noqa: BLE001
        return []


def filter_silence(segments: list, audio_path) -> list:
    """Drop segments that don't overlap any detected speech region."""
    regions = speech_regions(audio_path)
    if not regions:
        return segments  # VAD unavailable / found nothing usable → keep all
    kept = []
    for s in segments:
        if any(min(s.end, e) - max(s.start, st) > 0.0 for st, e in regions):
            kept.append(s)
    # Never return an empty transcript just because VAD disagreed.
    return kept or segments


# ---------------------------------------------------------------------------
# Phoneme forced alignment (per-word timestamps)
# ---------------------------------------------------------------------------

def _normalize_words(text: str) -> List[str]:
    # MMS_FA expects lowercase letter tokens; strip punctuation, keep letters.
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return [w for w in cleaned.split() if w]


def _load_fa():
    global _fa
    if _fa is None:
        from torchaudio.pipelines import MMS_FA as bundle

        _fa = (bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner())
    return _fa


def align_words(segments: list, audio_path) -> list:
    """Attach per-word timestamps to each segment via MMS_FA forced alignment.

    Best-effort and per-segment: a segment that fails to align is left as-is.
    """
    try:
        import torch

        model, tokenizer, aligner = _load_fa()
        audio = _load_audio(audio_path)
    except Exception:  # noqa: BLE001
        return segments

    for seg in segments:
        words = _normalize_words(seg.text)
        a, b = int(seg.start * _SR), int(seg.end * _SR)
        if not words or b - a < int(0.1 * _SR):
            continue
        try:
            import numpy as np

            wav = torch.from_numpy(np.ascontiguousarray(audio[a:b])).unsqueeze(0)
            with torch.inference_mode():
                emission, _ = model(wav)
            spans = aligner(emission[0], tokenizer(words))
            ratio = wav.size(1) / emission.size(1) / _SR
            seg.words = [
                {
                    "word": w,
                    "start": round(seg.start + sp[0].start * ratio, 3),
                    "end": round(seg.start + sp[-1].end * ratio, 3),
                }
                for w, sp in zip(words, spans)
                if sp
            ]
        except Exception:  # noqa: BLE001
            continue
    return segments
