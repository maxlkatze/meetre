"""Who is who within one side — voiceprints of the transcribed segments.

The side (you vs. the call) is already known from the stems, so this only has
to tell voices apart *within* a side. Every segment of ≥ MIN_SECONDS gets a
voiceprint (NeMo TitaNet-small via sherpa-onnx, CPU, ~11 ms per second of
audio), the prints are clustered, and short segments inherit the voice next to
them. No segmentation model, no torch, no HuggingFace token: a 25-minute
meeting takes a few seconds instead of pyannote's ~45 s (and pyannote is an
optional extra that often isn't installed at all).
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

SR = 16_000
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/nemo_en_titanet_small.onnx")
MODEL_PATH = Path.home() / ".cache" / "meetre" / "speaker" / "nemo_en_titanet_small.onnx"

# Shorter segments only inherit a neighbour's voice. 0.8 s still separates
# well (TitaNet EER 6.4 % at 0.8 s); 1.5 s left a 48 s two-person chat of
# short lines with too few prints to find the second voice.
MIN_SECONDS = 0.8
WINDOW = 2.0        # a print is the mean of up to WINDOWS windows this long
WINDOWS = 4
# Measured on real calls (2026-10-25): the same person over a call / laptop mic
# scores only 0.4-0.7 between segments, different people 0.0-0.35 — far lower
# than clean audio. 0.5 split one person into 2-3 "voices" on 21 of 34 sides.
SAME_VOICE = 0.4    # average-linkage cosine similarity to merge two clusters
MIN_CLUSTER_SECONDS = 10.0  # a voice needs this much speech …
MIN_CLUSTER_SHARE = 0.15    # … and this share of the side to count as a person
ECHO_SIMILAR = 0.55 # mic print this close to a call voice (and overlapping it) = echo

_extractor = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _extractor
    with _lock:
        if _extractor is None:
            import sherpa_onnx

            if not MODEL_PATH.exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = MODEL_PATH.with_suffix(".part")
                urllib.request.urlretrieve(MODEL_URL, tmp)
                tmp.rename(MODEL_PATH)
            _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(MODEL_PATH), num_threads=4))
        return _extractor


def voiceprint(audio: np.ndarray) -> np.ndarray:
    """Unit-length voiceprint of a 16 kHz clip: mean of ≤ WINDOWS spread windows."""
    ex = _load()
    n, w = len(audio), int(WINDOW * SR)
    starts = [0] if n <= w else np.linspace(0, n - w, min(WINDOWS, n // w)).astype(int)
    vecs = []
    for s in starts:
        st = ex.create_stream()
        st.accept_waveform(SR, np.ascontiguousarray(audio[s:s + w], dtype="float32"))
        st.input_finished()
        vecs.append(np.asarray(ex.compute(st), dtype="float32"))
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def _cluster(prints: np.ndarray, n: Optional[int]) -> List[int]:
    """Average-linkage agglomerative clustering on cosine similarity.

    Merges while the closest pair is ≥ SAME_VOICE, or down to ``n`` clusters
    when the count is known."""
    # Cluster-to-cluster mean similarities, updated in place on each merge
    # (Lance-Williams), so 400 segments cluster in milliseconds.
    sim = (prints @ prints.T).astype("float64")
    np.fill_diagonal(sim, -np.inf)
    size = np.ones(len(prints))
    alive = np.ones(len(prints), bool)
    owner = np.arange(len(prints))
    while alive.sum() > 1:
        a, b = np.unravel_index(np.argmax(sim), sim.shape)
        if n is not None and alive.sum() <= n:
            break
        if n is None and sim[a, b] < SAME_VOICE:
            break
        row = (size[a] * sim[a] + size[b] * sim[b]) / (size[a] + size[b])
        sim[a], sim[:, a] = row, row
        sim[a, a] = -np.inf
        sim[b], sim[:, b] = -np.inf, -np.inf
        size[a] += size[b]
        alive[b] = False
        owner[owner == b] = a
    ids = {k: i for i, k in enumerate(dict.fromkeys(owner.tolist()))}
    return [ids[k] for k in owner.tolist()]


def _prints(audio: np.ndarray, segs: list) -> tuple:
    long = [i for i, s in enumerate(segs) if s.end - s.start >= MIN_SECONDS]
    if not long:
        return long, np.zeros((0, 192), dtype="float32")
    return long, np.stack([voiceprint(audio[int(segs[i].start * SR):int(segs[i].end * SR)])
                           for i in long])


def _voices(segs: list, long: list, prints: np.ndarray, n: Optional[int]) -> dict:
    """``{segment index: voice}`` for the long segments; spill-over clusters
    (too little speech to be a person) are folded into the closest real one."""
    if len(long) < 2 or n == 1:
        return {i: 0 for i in long}
    raw = dict(zip(long, _cluster(prints, n)))
    talk: dict = {}
    for i, k in raw.items():
        talk[k] = talk.get(k, 0.0) + segs[i].end - segs[i].start
    total = sum(talk.values())
    if n is not None:  # the count was given: trust it, keep every cluster
        real = list(talk)
    else:
        real = [k for k, t in talk.items()
                if t >= MIN_CLUSTER_SECONDS and t >= MIN_CLUSTER_SHARE * total] \
            or [max(talk, key=talk.get)]
    def centroid(k):
        c = prints[[j for j, i in enumerate(long) if raw[i] == k]].mean(0)
        return c / (np.linalg.norm(c) + 1e-9)

    centroids = {k: centroid(k) for k in real}
    for j, i in enumerate(long):
        if raw[i] not in real:
            raw[i] = max(real, key=lambda k: float(prints[j] @ centroids[k]))
    return raw


TRACK_WINDOW = 1.0  # voice track: a print every TRACK_HOP s over TRACK_WINDOW s
TRACK_HOP = 0.5
TRACK_MARGIN = 0.05 # a window counts only if one voice wins by this much


CLEAR_MARGIN = 0.15  # a short segment whose own print wins by this much
CLEAR_SECONDS = 4.0  # (and ≤ this long) is one voice: no track needed


def _track(audio: np.ndarray, segs: list, cents: np.ndarray,
           own: Optional[dict] = None) -> List[tuple]:
    """``[(start, end, voice)]`` second by second through the segments: people
    talk over each other and Whisper's segments (and invented timings) don't
    follow speaker changes, so a segment's own print can be a blend."""
    turns = []
    for i, s in enumerate(segs):
        if own and i in own and s.end - s.start <= CLEAR_SECONDS:
            sims = cents @ own[i]
            top = np.argsort(sims)[::-1]
            if len(top) < 2 or sims[top[0]] - sims[top[1]] >= CLEAR_MARGIN:
                turns.append((s.start, s.end, int(top[0])))
                continue
        start, end = s.start, s.end
        if end - start < TRACK_WINDOW:  # one word: a window centred on it
            mid = (start + end) / 2
            start, end = max(0.0, mid - TRACK_WINDOW / 2), mid + TRACK_WINDOW / 2
        t = start
        while t < end - 0.5 or t == start:
            w = audio[int(t * SR):int(min(t + TRACK_WINDOW, end) * SR)]
            if len(w) >= SR // 2:
                sims = cents @ voiceprint(w)
                top = np.argsort(sims)[::-1]
                if len(top) < 2 or sims[top[0]] - sims[top[1]] >= TRACK_MARGIN:
                    c = t + min(TRACK_WINDOW, end - t) / 2  # credit the window's middle
                    turns.append((c - TRACK_HOP / 2, c + TRACK_HOP / 2, int(top[0])))
            t += TRACK_HOP
    return turns


def _label(audio: np.ndarray, segs: list, long: list, prints: np.ndarray, raw: dict,
           single_label: str, multi_label: str) -> list:
    """Name the voices; with several, follow the voice track, splitting
    segments (with word timings) where the speaker changes. Returns the
    (possibly split) segments."""
    voices = sorted(set(raw.values()))
    if len(voices) <= 1:
        for s in segs:
            s.speaker = single_label
        return segs
    from .attribution import split_by_turns

    cents = []
    for k in voices:
        c = prints[[j for j, i in enumerate(long) if raw[i] == k]].mean(0)
        cents.append(c / (np.linalg.norm(c) + 1e-9))
    turns = _track(audio, segs, np.stack(cents), {i: prints[j] for j, i in enumerate(long)})
    segs = split_by_turns(segs, turns)
    labels: List[Optional[int]] = []
    for s in segs:
        votes = np.zeros(len(voices))
        for b, e, k in turns:
            votes[k] += max(0.0, min(s.end, e) - max(s.start, b))
        labels.append(int(votes.argmax()) if votes.any() else None)
    known = [i for i, k in enumerate(labels) if k is not None]
    for i, s in enumerate(segs):  # no clear window: the nearest decided segment
        if labels[i] is None:
            labels[i] = labels[min(known, key=lambda j: abs(segs[j].start - s.start))] \
                if known else 0
    order: dict = {}
    for i in sorted(range(len(segs)), key=lambda i: segs[i].start):
        order.setdefault(labels[i], len(order) + 1)
    out: list = []
    for s, k in zip(segs, labels):
        s.speaker = f"{multi_label} {order[k]}"
        prev = out[-1] if out else None
        if prev is not None and prev.speaker == s.speaker and prev.words and s.words \
                and s.start - prev.end < 0.05:  # split pieces that ended up one voice
            prev.text = f"{prev.text} {s.text}"
            prev.end, prev.words = s.end, prev.words + s.words
        else:
            out.append(s)
    return out


def split_side(audio: np.ndarray, segs: list, single_label: str, multi_label: str,
               n: Optional[int] = None) -> list:
    """Label ``segs`` (one side, timestamps into ``audio``):
    ``single_label`` if one voice, else ``"{multi_label} 1"``, ``… 2`` in order
    of first appearance. ``n`` = known number of voices on this side. Returns
    the segments, split where the speaker changes mid-segment."""
    long, prints = _prints(audio, segs)
    return _label(audio, segs, long, prints, _voices(segs, long, prints, n),
                  single_label, multi_label)


def split_sides(mic: np.ndarray, mic_segs: list, system: np.ndarray, sys_segs: list,
                labels: dict, n_total: Optional[int] = None) -> tuple:
    """Both sides of a call: drop mic segments that are the call's voice heard
    through the speakers (sounds like a call voice *and* overlaps call speech),
    then split each side into voices. Returns ``(mic_segs, sys_segs)``: the
    kept mic segments, both sides split where the speaker changes.
    ``n_total`` = everyone in the meeting: the call's voices are found first,
    the mic is split into the rest.

    Complements the text-based check (:func:`attribution.drop_echo`, run
    first), which misses echo the mic transcribed differently; on real calls
    the echo forms its own mic "voice" at 0.6-0.85 similarity to the call vs.
    ≤ 0.35 for your own voice."""
    s_long, s_prints = _prints(system, sys_segs)
    s_raw = _voices(sys_segs, s_long, s_prints, None)
    n_local = None
    if n_total:  # everyone not heard on the call is in the room
        n_local = max(1, n_total - (len(set(s_raw.values())) if s_raw else 0))
    m_long, m_prints = _prints(mic, mic_segs)
    echo = set()
    if len(s_long) and len(m_long):
        cents = [s_prints[[j for j, i in enumerate(s_long) if s_raw[i] == k]].mean(0)
                 for k in set(s_raw.values())]
        cents = np.stack([c / (np.linalg.norm(c) + 1e-9) for c in cents])
        for j, i in enumerate(m_long):
            g = mic_segs[i]
            if float((cents @ m_prints[j]).max()) >= ECHO_SIMILAR and any(
                    min(g.end, s.end) - max(g.start, s.start) > 0 for s in sys_segs):
                echo.add(i)
    keep = [i for i in range(len(mic_segs)) if i not in echo]
    kept = [mic_segs[i] for i in keep]
    k_long = [keep.index(i) for i in m_long if i not in echo]
    k_prints = m_prints[[j for j, i in enumerate(m_long) if i not in echo]] if k_long else m_prints[:0]
    kept = _label(mic, kept, k_long, k_prints, _voices(kept, k_long, k_prints, n_local),
                  labels["local"], labels["local_multi"])
    sys_segs = _label(system, sys_segs, s_long, s_prints, s_raw,
                      labels["remote"], labels["remote_multi"])
    return kept, sys_segs
