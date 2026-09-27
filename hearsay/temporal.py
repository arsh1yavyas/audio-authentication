"""Ordered-window features for speech dynamics across a clip.

Unlike the clip-level baseline, this view keeps coarse time order. The features
are intentionally lightweight so they can run in the challenge Docker image.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .features import EPSILON, SAMPLE_RATE, decode_audio

WINDOWS = 8
N_FFT = 512
HOP_LENGTH = 160
FRAME_FEATURES = ("rms", "zcr", "centroid", "flatness", "flux",
                  "low_energy", "mid_energy", "high_energy")
TEMPORAL_FEATURE_NAMES = tuple(
    [f"bin_{window:02d}_{name}_{stat}" for window in range(WINDOWS)
     for name in FRAME_FEATURES for stat in ("mean", "std")]
    + [f"trend_{name}" for name in FRAME_FEATURES]
)


def extract_temporal_features(path: Path, audio: np.ndarray | None = None) -> np.ndarray:
    """Return fixed-size ordered-window summaries and clip-wide feature trends."""
    signal = decode_audio(path) if audio is None else np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or signal.size < SAMPLE_RATE // 2:
        raise ValueError(f"Audio is too short or not mono: {path}")
    peak = float(np.max(np.abs(signal)))
    if not np.isfinite(peak) or peak < 1e-6 or not np.all(np.isfinite(signal)):
        raise ValueError(f"Audio is silent or invalid: {path}")
    signal = (signal / peak).astype(np.float32)

    spectrum = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = spectrum.astype(np.float64) ** 2
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    total = np.maximum(power.sum(axis=0), EPSILON)
    normalized = spectrum / np.maximum(spectrum.sum(axis=0, keepdims=True), EPSILON)
    flux = np.r_[0.0, np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0))]
    frames = {
        "rms": librosa.feature.rms(S=spectrum, frame_length=N_FFT)[0],
        "zcr": librosa.feature.zero_crossing_rate(signal, frame_length=N_FFT,
                                                    hop_length=HOP_LENGTH)[0],
        "centroid": librosa.feature.spectral_centroid(S=spectrum, sr=SAMPLE_RATE)[0] / (SAMPLE_RATE / 2),
        "flatness": librosa.feature.spectral_flatness(S=spectrum)[0],
        "flux": flux,
    }
    for name, lo, hi in (("low_energy", 0, 500), ("mid_energy", 500, 2_000),
                         ("high_energy", 2_000, SAMPLE_RATE // 2 + 1)):
        mask = (frequencies >= lo) & (frequencies < hi)
        frames[name] = power[mask].sum(axis=0) / total

    frame_count = len(next(iter(frames.values())))
    edges = np.linspace(0, frame_count, WINDOWS + 1, dtype=int)
    values: list[float] = []
    bin_means = []
    for window in range(WINDOWS):
        start, end = int(edges[window]), int(edges[window + 1])
        end = max(end, start + 1)
        means = []
        for name in FRAME_FEATURES:
            part = frames[name][start:min(end, frame_count)]
            if part.size == 0:
                part = frames[name][-1:]
            mean, std = float(np.mean(part)), float(np.std(part))
            values.extend((mean, std))
            means.append(mean)
        bin_means.append(means)
    # A normalized linear trend captures direction of change without speech content.
    time_axis = np.linspace(-1.0, 1.0, WINDOWS)
    slopes = np.polyfit(time_axis, np.asarray(bin_means), 1)[0]
    values.extend(float(value) for value in slopes)
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(TEMPORAL_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Temporal feature extraction failed for {path}")
    return vector
