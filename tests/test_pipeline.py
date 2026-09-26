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

import numpy as np
import imageio_ffmpeg

from hearsay.__main__ import main
from hearsay.features import FEATURE_NAMES, extract_features
from hearsay.model import explain, load_model, train


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
            vector = extract_features(paths[0])
            self.assertTrue(np.isfinite(vector).all())
            model = load_model(model_path)
            result = explain(model, vector)
            self.assertGreaterEqual(result["cm-score"], 0.0)
            self.assertLessEqual(result["cm-score"], 1.0)
            self.assertAlmostEqual(result["cm-score"],
                                   float(model.predict_proba(vector.reshape(1, -1))[0, 1]), places=6)
            self.assertAlmostEqual(result["cm-score"],
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
