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
import joblib
from sklearn.ensemble import ExtraTreesClassifier

from hearsay.__main__ import main
from hearsay.features import FEATURE_NAMES, extract_features
from hearsay.metadata import (METADATA_REAL_WEIGHT, MetadataAnalysis, adjust_score,
                              analyze_metadata)
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.model import explain, load_model, train
from hearsay.routing import min_dcf
from hearsay.temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features
from hearsay.utils import read_manifest, split_data, write_split_manifests
from scripts.train_optimized import summarize as summarize_optimized


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
    def test_julia_optimizer_reports_hearsay_costs(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.2, 0.7, 0.9])
        report = summarize_optimized(labels, scores)
        expected, threshold = min_dcf(labels, scores, p_spoof=0.3, c_miss=1.0, c_fa=4.0)
        self.assertAlmostEqual(report["min_dcf"], expected)
        self.assertEqual(report["min_dcf_threshold"], threshold)
        self.assertEqual(report["min_dcf_config"],
                         {"p_spoof": 0.3, "c_miss": 1.0, "c_fa": 4.0})

    def test_saved_optimized_model_score_explanation(self) -> None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "hearsay-optimized.joblib"
        model = load_model(model_path)
        vector = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        result = explain(model, vector)
        expected = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
        self.assertAlmostEqual(result["audio_score"], expected)
        self.assertAlmostEqual(result["cm-score"], expected)
        self.assertEqual(len(result["component_scores"]), 2)
        self.assertEqual(result["feature_evidence"], [])

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
    def test_metadata_adjustment_is_small_and_missing_is_neutral(self) -> None:
        self.assertAlmostEqual(adjust_score(0.8, MetadataAnalysis("consistent")), 0.76)
        self.assertAlmostEqual(adjust_score(0.8, MetadataAnalysis("inconsistent")), 0.8)
        self.assertAlmostEqual(adjust_score(0.8, MetadataAnalysis("unknown")), 0.8)
        with self.assertRaises(ValueError):
            adjust_score(1.1, MetadataAnalysis("consistent"))

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

            temporal_features = np.vstack([extract_temporal_features(path) for path in paths])
            temporal_labels = np.asarray([0 if i < 6 else 1 for i in range(len(paths))])
            temporal_model = ExtraTreesClassifier(n_estimators=10, random_state=3).fit(
                temporal_features, temporal_labels)
            temporal_path = folder / "temporal.joblib"
            joblib.dump({"model": temporal_model, "feature_names": TEMPORAL_FEATURE_NAMES,
                         "version": 1, "kind": "experimental-temporal"}, temporal_path)
            policy = folder / "fusion_policy.json"
            policy.write_text(json.dumps({"temporal_weight": 0.5,
                                          "metadata_real_weight": 0.05}), encoding="utf-8")
            fused_output = folder / "fused.tsv"
            self.assertEqual(main([
                "predict-fused", "--model", str(model_path),
                "--temporal-model", str(temporal_path), "--policy", str(policy),
                "--input", str(folder), "--template", str(template),
                "--output", str(fused_output),
            ]), 0)
            with fused_output.open("r", newline="", encoding="utf-8") as stream:
                fused_rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["filename"] for row in fused_rows],
                             [path.name for path in reversed(paths)])
            self.assertTrue(all(0.0 <= float(row["cm-score"]) <= 1.0
                                for row in fused_rows))

            combined_features = np.vstack([
                np.concatenate((extract_features(path), extract_lfcc_features(path)))
                for path in paths
            ])
            lfcc_model = ExtraTreesClassifier(n_estimators=10, random_state=4).fit(
                combined_features, temporal_labels)
            lfcc_path = folder / "baseline_lfcc.joblib"
            joblib.dump({"model": lfcc_model,
                         "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                         "version": 1, "kind": "experimental-baseline-plus-lfcc"}, lfcc_path)
            lfcc_output = folder / "lfcc.tsv"
            self.assertEqual(main([
                "predict-lfcc", "--model", str(lfcc_path), "--input", str(folder),
                "--template", str(template), "--output", str(lfcc_output),
            ]), 0)
            with lfcc_output.open("r", newline="", encoding="utf-8") as stream:
                lfcc_rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["filename"] for row in lfcc_rows],
                             [path.name for path in reversed(paths)])
            self.assertTrue(all(0.0 <= float(row["cm-score"]) <= 1.0
                                for row in lfcc_rows))
            lowpass_lfcc_path = folder / "baseline_lfcc_lowpass.joblib"
            joblib.dump({"model": lfcc_model,
                         "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                         "version": 1, "kind": "experimental-baseline-plus-lfcc",
                         "lowpass_hz": 7_000.0}, lowpass_lfcc_path)
            lowpass_lfcc_output = folder / "lfcc_lowpass.tsv"
            self.assertEqual(main([
                "predict-lfcc", "--model", str(lowpass_lfcc_path), "--input", str(folder),
                "--template", str(template), "--output", str(lowpass_lfcc_output),
            ]), 0)
            with lowpass_lfcc_output.open("r", newline="", encoding="utf-8") as stream:
                lowpass_rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["filename"] for row in lowpass_rows],
                             [path.name for path in reversed(paths)])


if __name__ == "__main__":
    unittest.main()
