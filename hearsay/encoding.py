"""Container and encoding forensics as a separate experimental feature view.

Missing metadata is represented as unknown. These clues are not treated as
proof of synthetic speech: re-encoding and transcoding can affect real speech.
"""

from __future__ import annotations

import wave
from pathlib import Path

import librosa
import numpy as np

from .features import EPSILON, SAMPLE_RATE, decode_audio

ENCODING_FEATURE_NAMES = (
    "is_wav", "pcm_format", "channels", "source_sample_rate", "bits_per_sample",
    "source_bytes_per_second", "log_file_bytes_per_second", "clip_fraction",
    "zero_fraction", "repeat_sample_fraction", "unique_level_fraction",
    "high_frequency_fraction", "spectral_cutoff_fraction", "spectral_flatness",
)
ENCODING_HEADER_FEATURE_NAMES = ENCODING_FEATURE_NAMES[:7]
ENCODING_SIGNAL_FEATURE_NAMES = ENCODING_FEATURE_NAMES[7:]


def _wav_header(path: Path) -> tuple[float, ...]:
    """Read standard PCM WAV header fields; return zeros when not available."""
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            rate = stream.getframerate()
            width = stream.getsampwidth()
            frames = stream.getnframes()
            compression = stream.getcomptype()
        duration = frames / max(rate, 1)
        return (1.0, float(compression == "NONE"), float(channels),
                float(rate), float(width * 8),
                float(rate * channels * width),
                float(np.log1p(path.stat().st_size / max(duration, 1e-6))))
    except (wave.Error, OSError, EOFError):
        return (float(path.suffix.lower() == ".wav"), 0.0, 0.0, 0.0, 0.0,
                0.0, float(np.log1p(path.stat().st_size)))


def extract_encoding_features(path: Path, audio: np.ndarray | None = None) -> np.ndarray:
    """Extract basic format facts and decoded-signal encoding traces."""
    signal = decode_audio(path) if audio is None else np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or signal.size < SAMPLE_RATE // 2:
        raise ValueError(f"Audio is too short or not mono: {path}")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Audio contains invalid samples: {path}")
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-6:
        raise ValueError(f"Audio is silent: {path}")
    spectrum = np.abs(librosa.stft(signal.astype(np.float32), n_fft=1024, hop_length=512))
    power = spectrum.astype(np.float64) ** 2
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)
    total = max(float(power.sum()), EPSILON)
    high_fraction = float(power[frequencies >= 5_000].sum() / total)
    energy_by_frequency = power.sum(axis=1)
    cumulative = np.cumsum(energy_by_frequency) / max(float(energy_by_frequency.sum()), EPSILON)
    cutoff_index = min(int(np.searchsorted(cumulative, 0.99)), len(frequencies) - 1)
    cutoff_fraction = float(frequencies[cutoff_index] / (SAMPLE_RATE / 2))

    # Quantize to 16-bit units to reveal repeated levels after decoding while
    # avoiding a large set operation on long recordings.
    quantized = np.clip(np.rint(signal * 32768), -32768, 32767).astype(np.int32)
    sample = quantized[::max(1, len(quantized) // 200_000)]
    unique_fraction = float(len(np.unique(sample)) / max(len(sample), 1))
    repeated = float(np.mean(np.diff(sample) == 0)) if len(sample) > 1 else 0.0
    duration = len(signal) / SAMPLE_RATE
    header = _wav_header(path)
    values = np.asarray((
        *header,
        float(np.mean(np.abs(signal) >= 0.999)),
        float(np.mean(np.abs(signal) < 1e-7)),
        repeated,
        unique_fraction,
        high_fraction,
        cutoff_fraction,
        float(np.mean(librosa.feature.spectral_flatness(S=spectrum))),
    ), dtype=np.float32)
    if values.shape != (len(ENCODING_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError(f"Encoding feature extraction failed for {path}")
    return values


def extract_signal_encoding_features(path: Path, audio: np.ndarray | None = None) -> np.ndarray:
    """Return signal-level traces without container, sample-rate, or file-size clues."""
    return extract_encoding_features(path, audio)[len(ENCODING_HEADER_FEATURE_NAMES):]
