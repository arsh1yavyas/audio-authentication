"""Unit checks for optional temporal and encoding feature views."""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from hearsay.encoding import ENCODING_FEATURE_NAMES, extract_encoding_features
from hearsay.features import SAMPLE_RATE
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features


def write_signal(path: Path, signal: np.ndarray, width: int = 2) -> None:
    maximum = 2 ** (8 * width - 1) - 1
    samples = (np.clip(signal, -1, 1) * maximum).astype("<i2" if width == 2 else "<i4")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(width)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())


class FeatureViewsTest(unittest.TestCase):
    def test_feature_shapes_are_fixed_and_finite(self) -> None:
        times = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
        signal = 0.5 * np.sin(2 * math.pi * 440 * times)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            write_signal(path, signal)
            temporal = extract_temporal_features(path)
            encoding = extract_encoding_features(path)
            lfcc = extract_lfcc_features(path)
            self.assertEqual(temporal.shape, (len(TEMPORAL_FEATURE_NAMES),))
            self.assertEqual(encoding.shape, (len(ENCODING_FEATURE_NAMES),))
            self.assertEqual(lfcc.shape, (len(LFCC_FEATURE_NAMES),))
            self.assertTrue(np.isfinite(temporal).all())
            self.assertTrue(np.isfinite(encoding).all())
            self.assertTrue(np.isfinite(lfcc).all())
            self.assertEqual(encoding[0], 1.0)
            self.assertEqual(encoding[1], 1.0)
            self.assertEqual(encoding[2], 1.0)
            self.assertEqual(encoding[4], 16.0)

    def test_temporal_view_retains_coarse_order(self) -> None:
        times = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
        low = 0.5 * np.sin(2 * math.pi * 180 * times[:SAMPLE_RATE * 2])
        high = 0.5 * np.sin(2 * math.pi * 2_800 * times[:SAMPLE_RATE * 2])
        first = np.r_[low, high]
        reversed_order = np.r_[high, low]
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.wav"
            reversed_path = Path(temporary) / "reversed.wav"
            write_signal(first_path, first)
            write_signal(reversed_path, reversed_order)
            first_features = extract_temporal_features(first_path)
            reverse_features = extract_temporal_features(reversed_path)
            self.assertGreater(float(np.linalg.norm(first_features - reverse_features)), 0.1)
            # Each bin has 8 feature means followed by their standard deviations;
            # first-bin low-band energy therefore starts at offset 10.
            self.assertGreater(float(first_features[10]), float(reverse_features[10]))


if __name__ == "__main__":
    unittest.main()
