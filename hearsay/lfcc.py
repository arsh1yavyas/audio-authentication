"""Linear-frequency cepstral statistics for a transparent spoof-cue view.

LFCCs retain equal-width frequency detail that Mel-spaced MFCC filters can
compress. They are an additional evidence source, not proof of synthesis.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.fft import dct

from .features import EPSILON, SAMPLE_RATE, decode_audio, validate_audio

N_FFT = 1024
HOP_LENGTH = 256
N_FILTERS = 64
N_COEFFICIENTS = 20
COMPONENTS = ("static", "delta", "delta2")
LFCC_FEATURE_NAMES = tuple(
    f"lfcc_{component}_{coefficient:02d}_{stat}"
    for component in COMPONENTS
    for coefficient in range(1, N_COEFFICIENTS + 1)
    for stat in ("mean", "std")
)


def _linear_filterbank() -> np.ndarray:
    """Build triangular equal-Hz filters over the full supported band."""
    frequencies = np.linspace(0.0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    edges = np.linspace(0.0, SAMPLE_RATE / 2, N_FILTERS + 2)
    bank = np.zeros((N_FILTERS, len(frequencies)), dtype=np.float64)
    for index in range(N_FILTERS):
        left, center, right = edges[index:index + 3]
        bank[index] = np.maximum(
            0.0, np.minimum((frequencies - left) / max(center - left, EPSILON),
                            (right - frequencies) / max(right - center, EPSILON)))
    # Normalize discrete filter weights so bin alignment does not change
    # one filter's response relative to another.
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), EPSILON)
    return bank


_FILTERBANK = _linear_filterbank()


def extract_lfcc_features(path: Path, audio: np.ndarray | None = None) -> np.ndarray:
    """Return mean/std of static LFCCs and first/second temporal deltas."""
    signal = decode_audio(path) if audio is None else validate_audio(audio, path)
    peak = float(np.max(np.abs(signal)))
    signal = (signal / peak).astype(np.float32)
    magnitude = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    log_energy = np.log(np.maximum(_FILTERBANK @ (magnitude.astype(np.float64) ** 2), EPSILON))
    # Omit C0, which mostly encodes overall level; keep twenty spectral-shape
    # coefficients and summarize both level and movement over time.
    static = dct(log_energy, type=2, norm="ortho", axis=0)[1:N_COEFFICIENTS + 1]
    delta = librosa.feature.delta(static, order=1, axis=-1)
    delta2 = librosa.feature.delta(static, order=2, axis=-1)
    values = []
    for component in (static, delta, delta2):
        for row in component:
            values.extend((float(np.mean(row)), float(np.std(row))))
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(LFCC_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"LFCC feature extraction failed for {path}")
    return vector
