"""Decode audio and turn the Research.md signal measurements into a fixed vector."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 120
N_FFT = 1024
HOP_LENGTH = 256
EPSILON = 1e-10

STATS = ("mean", "std")
SERIES = (
    "zcr",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff_85",
    "spectral_flatness",
    "spectral_flux",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
)
FEATURE_NAMES = tuple(
    [f"{name}_{stat}" for name in SERIES for stat in STATS]
    + [f"mfcc_{i:02d}_{stat}" for i in range(1, 21) for stat in STATS]
    + [f"chroma_{i:02d}_{stat}" for i in range(12) for stat in STATS]
)


def decode_audio(path: Path) -> np.ndarray:
    """Decode the first MAX_SECONDS of any format supported by bundled FFmpeg."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-t", str(MAX_SECONDS), "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise ValueError(f"Cannot start FFmpeg for {path}: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Cannot decode {path}: {message}")
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if audio.size < SAMPLE_RATE // 2:
        raise ValueError(f"Audio is shorter than 0.5 seconds: {path}")
    if not np.all(np.isfinite(audio)) or np.max(np.abs(audio)) < 1e-6:
        raise ValueError(f"Audio is silent or invalid: {path}")
    return audio


def _summarize(values: np.ndarray) -> list[float]:
    return [float(np.mean(values)), float(np.std(values))]


def extract_features(path: Path) -> np.ndarray:
    """Return STFT, MFCC, chroma, and six statistical feature families.

    The STFT is represented by spectral flux and three relative frequency bands.
    Mean and standard deviation retain some time variation in a fixed-size vector.
    """
    audio = decode_audio(path)
    # Peak normalization removes arbitrary recording gain while preserving dynamics.
    audio = audio / np.max(np.abs(audio))
    signal = audio.astype(np.float32)
    spectrum = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = spectrum**2
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    total_power = np.maximum(power.sum(axis=0), EPSILON)
    normalized_spectrum = spectrum / np.maximum(spectrum.sum(axis=0), EPSILON)
    flux = np.r_[0.0, np.sqrt(np.sum(np.diff(normalized_spectrum, axis=1) ** 2, axis=0))]

    measurements = {
        "zcr": librosa.feature.zero_crossing_rate(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)[0],
        "rms": librosa.feature.rms(S=spectrum, frame_length=N_FFT)[0],
        "spectral_centroid": librosa.feature.spectral_centroid(S=spectrum, sr=SAMPLE_RATE)[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=spectrum, sr=SAMPLE_RATE)[0],
        "spectral_rolloff_85": librosa.feature.spectral_rolloff(S=spectrum, sr=SAMPLE_RATE, roll_percent=0.85)[0],
        "spectral_flatness": librosa.feature.spectral_flatness(S=spectrum)[0],
        "spectral_flux": flux,
    }
    for name, lower, upper in (
        ("low_band_energy", 0, 500),
        ("mid_band_energy", 500, 2_000),
        ("high_band_energy", 2_000, SAMPLE_RATE // 2 + 1),
    ):
        mask = (frequencies >= lower) & (frequencies < upper)
        measurements[name] = power[mask].sum(axis=0) / total_power

    values: list[float] = []
    for name in SERIES:
        values.extend(_summarize(measurements[name]))

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=20,
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(S=power, sr=SAMPLE_RATE,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    for matrix in (mfcc, chroma):
        for row in matrix:
            values.extend(_summarize(row))
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Feature extraction failed for {path}")
    return vector
