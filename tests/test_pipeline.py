"""Small software smoke test; generated tones are not an authenticity benchmark."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import imageio_ffmpeg

from hearsay.__main__ import main
from hearsay.features import FEATURE_NAMES, extract_features
from hearsay.metadata import METADATA_REAL_WEIGHT, adjust_score, analyze_metadata
from hearsay.model import explain, load_model, train
from hearsay.utils import read_manifest, split_data, write_split_manifests


def write_wave(path: Path, frequency: float, noise: float, seed: int) -> None:
    sample_rate = 16_000
    times = np.arange(sample_rate * 2) / sample_rate
    rng = np.random.default_rng(seed)
    signal = 0.5 * np.sin(2 * math.pi * frequency * times) + noise * rng.standard_normal(len(times))
    samples = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


class PipelineTest(unittest.TestCase):
    def test_metadata_check_and_real_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            valid = folder / "valid.wav"
            write_wave(valid, 220, 0.02, 7)
            analysis = analyze_metadata(valid)
            self.assertEqual(analysis.status, "consistent")
            self.assertAlmostEqual(adjust_score(0.6, analysis), 0.6 * (1 - METADATA_REAL_WEIGHT))
            broken = folder / "broken.wav"
            contents = bytearray(valid.read_bytes())
            contents[4:8] = (len(contents) + 100).to_bytes(4, "little")
            broken.write_bytes(contents)
            broken_analysis = analyze_metadata(broken)
            self.assertEqual(broken_analysis.status, "inconsistent")
            self.assertEqual(adjust_score(0.6, broken_analysis), 0.6)
            mislabeled = folder / "mislabeled.wav"
            mislabeled.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00audio")
            self.assertEqual(analyze_metadata(mislabeled).status, "inconsistent")

    def test_grouped_split_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / "all.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("filename", "label", "group"))
                for label in ("real", "synthetic"):
                    for group in range(4):
                        for clip in range(2):
                            name = f"{label}_{group}_{clip}.wav"
                            (folder / name).touch()
                            writer.writerow((name, label, f"{label}_{group}"))
            train_path, test_path = write_split_manifests(manifest, folder / "split")
            train_files, train_labels, train_groups = read_manifest(train_path)
            test_files, test_labels, test_groups = read_manifest(test_path)
            self.assertEqual(len(train_files) + len(test_files), 16)
            self.assertFalse(set(train_files) & set(test_files))
            self.assertFalse(set(train_groups) & set(test_groups))
            self.assertEqual(set(train_labels), {0, 1})
            self.assertEqual(set(test_labels), {0, 1})
            self.assertEqual(int(sum(test_labels == 0)), int(sum(test_labels == 1)))
            before = (train_path.read_bytes(), test_path.read_bytes())
            write_split_manifests(manifest, folder / "split")
            self.assertEqual(before, (train_path.read_bytes(), test_path.read_bytes()))
            features = np.random.default_rng(0).normal(size=(16, len(FEATURE_NAMES)))
            with patch("hearsay.model.feature_matrix", return_value=features):
                report = train(train_path, folder / "split_model.joblib", test_manifest=test_path)
            self.assertEqual(report["training_samples"], len(train_files))
            self.assertEqual(report["validation"]["test_samples"], len(test_files))
            self.assertEqual(report["validation"]["method"], "provided test manifest")
            with self.assertRaisesRegex(ValueError, "overlapping audio files"):
                train(train_path, folder / "invalid.joblib", test_manifest=train_path)

    def test_clip_split_keeps_both_labels(self) -> None:
        labels = np.asarray([0] * 6 + [1] * 6)
        train_idx, test_idx = split_data(labels, seed=7)
        self.assertEqual(set(train_idx) & set(test_idx), set())
        self.assertEqual(set(labels[train_idx]), {0, 1})
        self.assertEqual(set(labels[test_idx]), {0, 1})

    def test_mp3_and_m4a_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source.wav"
            write_wave(source, 220, 0.02, 7)
            for suffix in (".mp3", ".m4a"):
                converted = folder / f"converted{suffix}"
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
                                "-i", str(source), str(converted)], check=True)
                self.assertEqual(extract_features(converted).shape, (len(FEATURE_NAMES),))
                self.assertEqual(analyze_metadata(converted).status, "consistent")

    def test_train_explain_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = []
            for index in range(12):
                path = folder / f"clip_{index}.wav"
                write_wave(path, 170 + index * 13, 0.01 if index < 6 else 0.25, index)
                paths.append(path)
            manifest = folder / "train.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("filename", "label"))
                writer.writerows((path.name, "real" if i < 6 else "synthetic")
                                 for i, path in enumerate(paths))
            model_path = folder / "forest.joblib"
            report = train(manifest, model_path)
            self.assertEqual(report["feature_count"], len(FEATURE_NAMES))
            self.assertIsNotNone(report["validation"])
            self.assertEqual(report["training_samples"] + report["validation"]["test_samples"], 12)
            vector = extract_features(paths[0])
            self.assertTrue(np.isfinite(vector).all())
            model = load_model(model_path)
            result = explain(model, vector, analyze_metadata(paths[0]))
            self.assertGreaterEqual(result["cm-score"], 0.0)
            self.assertLessEqual(result["cm-score"], 1.0)
            self.assertAlmostEqual(result["audio_score"],
                                   float(model.predict_proba(vector.reshape(1, -1))[0, 1]), places=6)
            self.assertAlmostEqual(result["cm-score"],
                                   result["audio_score"] * (1 - METADATA_REAL_WEIGHT), places=6)
            self.assertAlmostEqual(result["audio_score"],
                                   result["baseline_probability"] + result["total_contribution"],
                                   places=6)
            template = folder / "template.csv"
            with template.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("filename", "cm-score"))
                writer.writerows((path.name, "0.006") for path in reversed(paths))
            output = folder / "predictions.tsv"
            self.assertEqual(main(["predict", "--model", str(model_path), "--input", str(folder),
                                   "--template", str(template), "--output", str(output)]), 0)
            with output.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["filename"] for row in rows], [p.name for p in reversed(paths)])
            self.assertTrue(all(0 <= float(row["cm-score"]) <= 1 for row in rows))
            explained_output = folder / "explained.tsv"
            explanations = folder / "explanations.jsonl"
            self.assertEqual(main(["predict", "--model", str(model_path), "--input", str(folder),
                                   "--template", str(template), "--output", str(explained_output),
                                   "--explanations", str(explanations)]), 0)
            self.assertEqual(output.read_text(), explained_output.read_text())
            self.assertEqual(len([json.loads(line) for line in explanations.read_text().splitlines()]),
                             len(paths))


if __name__ == "__main__":
    unittest.main()
