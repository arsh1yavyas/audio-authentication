"""Lightweight audio and container evidence used for conditional routing.

These measurements describe recording conditions. They are not authenticity
decisions: noisy or re-encoded audio may still be bona fide speech.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .features import SAMPLE_RATE, decode_audio

QUALITY_FEATURE_NAMES = (
    "duration_seconds", "peak_level", "rms_level", "clipped_fraction",
    "active_fraction", "spectral_flatness", "high_frequency_fraction",
    "low_frequency_fraction", "spectral_flux", "bytes_per_second",
    "has_metadata", "has_encoder_tag",
)


def _metadata_flags(path: Path) -> tuple[float, float]:
    """Read optional container tags without making missing tags suspicious."""
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(path)
        tags = getattr(audio, "tags", None) if audio is not None else None
        if not tags:
            return 0.0, 0.0
        rendered = " ".join(f"{key} {value}" for key, value in tags.items()).lower()
        encoder_terms = ("elevenlabs", "eleven labs", "tts", "text to speech", "synthesized",
                         "synthetic", "amazon polly", "google", "openai", "qwen")
        return 1.0, float(any(term in rendered for term in encoder_terms))
    except Exception:
        # Metadata parsing is opportunistic; malformed tags must not block scoring.
        return 0.0, 0.0


def extract_quality_features(path: Path, audio: np.ndarray | None = None) -> np.ndarray:
    """Extract a compact, cheap-to-compute quality vector for routing."""
    audio = (decode_audio(path) if audio is None else audio).astype(np.float32, copy=False)
    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    active = float(np.mean(np.abs(audio) > max(peak * 0.015, 1e-5)))

    # A single small STFT is enough for broad condition features; it is not a
    # denoising step and never replaces the original signal sent to the models.
    spectrum = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    power = spectrum.astype(np.float64) ** 2
    total = max(float(power.sum()), 1e-12)
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=spectrum)))
    high_fraction = float(power[frequencies >= 5_000].sum() / total)
    low_fraction = float(power[frequencies <= 120].sum() / total)
    normalized = spectrum / np.maximum(spectrum.sum(axis=0, keepdims=True), 1e-10)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0)))) if spectrum.shape[1] > 1 else 0.0
    has_metadata, has_encoder = _metadata_flags(path)

    values = np.asarray((duration, peak, rms, clipped, active, flatness,
                         high_fraction, low_fraction, flux,
                         path.stat().st_size / max(duration, 1e-6),
                         has_metadata, has_encoder), dtype=np.float64)
    # Compress broad scale differences while preserving zero-valued indicators.
    values[0] = np.log1p(values[0])
    values[9] = np.log1p(max(values[9], 0.0))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Quality feature extraction failed for {path}")
    return values
