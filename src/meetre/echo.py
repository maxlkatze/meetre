"""Remove the call's voice from the mic recording (speaker echo), offline.

Without headphones the mic hears the call through the Mac's speakers, so the
other person shows up a second time in "your" stem: as doubled lines, wrong
speaker labels and a doubled voice in the MP3. Live echo cancellation (macOS
Voice Processing) can't be used during calls: it makes every other app hear
the mic ~35 dB quieter. But meetre records the call audio itself as its own
stem — the exact signal that came out of the speakers — so afterwards the
echo can be predicted and subtracted, without touching the call:

1. **Delay**: the mic hears the call 0.3-0.6 s late (output buffers +
   capture); GCC-PHAT finds it every 10 s and the call is re-aligned to it
   (constant on synced stems; it drifts on recordings made before the mic
   clock fix).
2. **Adaptive filter**: per frequency bin, a few STFT frames of taps learn
   the room/speaker response (NLMS, normalised by the call's power); it
   adapts only while the call is talking and you aren't (double talk would
   teach it to cancel you).
3. **Residual suppression**: what the filter misses is attenuated per bin
   where the predicted echo dominates (spectral gain with a floor, smoothed).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SR = 16_000
FRAME = 512             # 32 ms STFT frames …
HOP = 256               # … at 50 % overlap
TAPS = 8                # filter length in frames (~130 ms of room response)
MU = 0.3                # NLMS step size
CONVERGE = 300          # frames (~5 s of call speech) before double talk is guarded
MAX_DELAY = 1.0         # s: search range for the bulk delay
SUPPRESS = 1.5          # residual echo over-subtraction factor
FLOOR = 0.08            # never attenuate a bin below this gain (-22 dB)


def bulk_delay(mic: np.ndarray, ref: np.ndarray, seconds: float = 120.0) -> int:
    """Samples by which ``mic`` lags ``ref`` (GCC-PHAT on the first
    ``seconds``); 0 if there's no clear peak (no echo to speak of)."""
    n = min(len(mic), len(ref), int(seconds * SR))
    if n < SR:
        return 0
    size = 1 << (2 * n - 1).bit_length()
    x = np.fft.rfft(mic[:n], size) * np.conj(np.fft.rfft(ref[:n], size))
    cc = np.fft.irfft(x / (np.abs(x) + 1e-9), size)
    span = int(MAX_DELAY * SR)
    lags = cc[:span]  # mic can only lag
    d = int(np.argmax(lags))
    noise = float(np.median(np.abs(lags))) + 1e-12
    return d if lags[d] > 8 * noise else 0


TRACK_WINDOW = 10.0     # s per delay measurement
TRACK_CONFIDENT = 15.0  # GCC-PHAT peak / median for a measurement to count


def delay_track(mic: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """The mic's lag behind ``ref`` in samples for every mic sample, measured
    every TRACK_WINDOW s and interpolated; None if the echo never shows.

    Recordings before the clock fix drift (the mic lost ~1 % of its samples,
    unevenly): the lag slid from +0.6 s to -1.6 s within minutes, which no
    fixed alignment can follow. With synced stems this is just a constant."""
    w = int(TRACK_WINDOW * SR)
    span = int(3 * MAX_DELAY * SR)
    pts = []
    for a in range(0, min(len(mic), len(ref)) - w + 1, w):
        x, r = mic[a:a + w], ref[a:a + w]
        if float(np.sqrt(np.mean(r * r))) < 0.005:
            continue  # the call is silent here: nothing to measure
        size = 1 << (2 * w - 1).bit_length()
        X = np.fft.rfft(x, size) * np.conj(np.fft.rfft(r, size))
        cc = np.fft.irfft(X / (np.abs(X) + 1e-9), size)
        lags = np.concatenate([cc[-span:], cc[:span]])
        k = int(np.argmax(lags))
        if lags[k] > TRACK_CONFIDENT * (float(np.median(np.abs(lags))) + 1e-12):
            pts.append((a + w / 2, k - span))
    if not pts:
        return None
    t, d = np.array(pts, dtype="float64").T
    if len(d) >= 3:  # drop outliers: a lag far from both neighbours' median
        med = np.array([np.median(d[max(0, i - 2):i + 3]) for i in range(len(d))])
        keep = np.abs(d - med) < 0.15 * SR
        t, d = t[keep], d[keep]
    return np.interp(np.arange(len(mic)), t, d) if len(t) else None


def _stft(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FRAME + 1)[:-1].astype("float32")
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def _istft(X: np.ndarray, length: int) -> np.ndarray:
    win = np.hanning(FRAME + 1)[:-1].astype("float32")
    frames = np.fft.irfft(X, FRAME, axis=1) * win
    out = np.zeros(length + FRAME, dtype="float64")
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + FRAME] += f
        norm[i * HOP:i * HOP + FRAME] += win * win
    return (out / np.maximum(norm, 1e-6))[:length].astype("float32")


def cancel(mic: np.ndarray, ref: np.ndarray, delay: Optional[int] = None) -> np.ndarray:
    """``mic`` with the echo of ``ref`` (the call audio, same clock and
    sample rate) removed. Returns a new float32 array, same length."""
    mic = np.asarray(mic, dtype="float32")
    ref = np.asarray(ref, dtype="float32")
    if len(ref) < len(mic):
        ref = np.pad(ref, (0, len(mic) - len(ref)))
    ref = ref[:len(mic)]
    # Line the call up with its echo (a little lead keeps the filter causal
    # if a delay estimate is a few ms late).
    lead = HOP
    if delay is not None:
        lag = np.full(len(mic), float(delay))
    else:
        lag = delay_track(mic, ref)
        if lag is None:
            return mic.copy()  # no echo found
    idx = np.arange(len(mic)) - (lag - lead)
    ok = (idx >= 0) & (idx < len(ref) - 1)
    aligned = np.zeros(len(mic), dtype="float32")
    aligned[ok] = np.interp(idx[ok], np.arange(len(ref)), ref)
    ref = aligned

    M, R = _stft(mic), _stft(ref)
    frames, bins = M.shape
    W = np.zeros((TAPS, bins), dtype=np.complex128)
    hist = np.zeros((TAPS, bins), dtype=np.complex128)
    E = np.empty_like(M)               # echo estimate
    Y = np.empty_like(M)               # filter output (mic - echo)
    ref_pow = np.abs(R) ** 2
    far_floor = np.percentile(ref_pow.sum(1), 30) * 4 + 1e-9
    # NLMS regulariser: keeps quiet bins from blowing the filter up.
    delta = 0.01 * TAPS * ref_pow.mean(0) + 1e-10
    learned = 0  # frames adapted so far: the double-talk guard needs an estimate first
    for t in range(frames):
        hist = np.roll(hist, 1, axis=0)
        hist[0] = R[t]
        e = (W * hist).sum(0)
        y = M[t] - e
        E[t], Y[t] = e, y
        far = ref_pow[t].sum() > far_floor
        # Double talk: the mic holds much more than the predicted echo.
        near = learned > CONVERGE and np.sum(np.abs(y) ** 2) > 4 * np.sum(np.abs(e) ** 2)
        if far and not near:
            W += MU * hist.conj() * y / ((np.abs(hist) ** 2).sum(0) + delta)
            learned += 1
    # Residual suppression: attenuate bins dominated by (predicted) echo.
    echo_pow = np.abs(E) ** 2
    out_pow = np.abs(Y) ** 2 + 1e-12
    gain = np.clip(1 - SUPPRESS * echo_pow / (out_pow + echo_pow), FLOOR, 1.0)
    k = np.ones(3) / 3  # smooth over time so it doesn't warble
    gain = np.apply_along_axis(lambda g: np.convolve(g, k, mode="same"), 0, gain)
    return _istft(Y * gain, len(mic))


STABLE = 0.05  # s: lag spread allowed for a recording to count as synced


def clean_stem(mic: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """The mic with the call's echo removed, or None when there's no echo or
    the stems drift (recordings before the mic clock fix: cancelling those
    gained under 1 dB, so they are left as they are)."""
    lag = delay_track(mic, ref)
    if lag is None or float(lag.max() - lag.min()) > STABLE * SR:
        return None
    return cancel(mic, ref, delay=int(np.median(lag)))
