"""Regression checks for the optional MindCF experiment."""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hearsay.features import FEATURE_NAMES
from hearsay.model import fit_mindcf
from scripts.train_mindcf import verified_existing_features


class MindcfTest(unittest.TestCase):
    def test_fitter_returns_finite_two_class_scores(self) -> None:
        rng = np.random.default_rng(42)
        features = np.r_[rng.normal(-1, 0.2, (12, 4)),
                         rng.normal(1, 0.2, (12, 4))]
        labels = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
            model = fit_mindcf(features, labels)
        scores = model.predict_proba(features)
        self.assertEqual(scores.shape, (24, 2))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.0)

    def test_existing_cache_checks_sample_rows_and_source_ages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"{index}.wav" for index in range(7)]
            for path in files:
                path.write_bytes(b"test placeholder")
            expected = np.vstack([np.full(len(FEATURE_NAMES), index, dtype=float)
                                  for index in range(len(files))])
            cache = root / "features.npz"
            np.savez_compressed(cache, features=expected)
            with patch("scripts.train_mindcf.extract_features",
                       side_effect=lambda path: expected[int(path.stem)].copy()) as extractor:
                actual, checked = verified_existing_features(files, cache)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(checked, [0, 1, 3, 4, 6])
            self.assertEqual(extractor.call_count, 5)

            os.utime(files[0], ns=(cache.stat().st_mtime_ns + 1_000_000_000,)*2)
            with self.assertRaisesRegex(ValueError, "Audio has changed"):
                verified_existing_features(files, cache)

    def test_existing_cache_rejects_mismatched_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            source.write_bytes(b"test placeholder")
            cache = root / "features.npz"
            np.savez_compressed(cache, features=np.zeros((1, len(FEATURE_NAMES))))
            with patch("scripts.train_mindcf.extract_features",
                       return_value=np.ones(len(FEATURE_NAMES))):
                with self.assertRaisesRegex(ValueError, "differs at row 0"):
                    verified_existing_features([source], cache)


if __name__ == "__main__":
    unittest.main()
