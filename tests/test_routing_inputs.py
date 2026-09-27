"""Score-table validation for optional condition-routing experiments."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_routing import read_rows


class RoutingInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "real.wav").write_bytes(b"audio")
        (self.root / "spoof.wav").write_bytes(b"audio")
        self.score_file = self.root / "scores.tsv"

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        with self.score_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "filename", "label", "group", "arshiya_score", "julia_score",
            ), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_rows_are_aligned_and_groups_trimmed(self) -> None:
        self.write_rows([
            {"filename": "real.wav", "label": "real", "group": " speaker_1 ",
             "arshiya_score": "0.1", "julia_score": "0.2"},
            {"filename": "spoof.wav", "label": "spoof", "group": "generator_1",
             "arshiya_score": "0.9", "julia_score": "0.8"},
        ])
        rows, files, labels = read_rows(self.score_file, "arshiya_score", "julia_score")
        self.assertEqual([row["group"] for row in rows], ["speaker_1", "generator_1"])
        self.assertEqual(files, [self.root / "real.wav", self.root / "spoof.wav"])
        self.assertEqual(labels.tolist(), [0, 1])

    def test_rejects_invalid_scores_duplicate_files_and_missing_groups(self) -> None:
        valid = {"filename": "real.wav", "label": "real", "group": "speaker_1",
                 "arshiya_score": "0.1", "julia_score": "0.2"}
        for changes in ({"arshiya_score": "nan"}, {"julia_score": "1.2"},
                        {"group": "  "}):
            with self.subTest(changes=changes):
                self.write_rows([{**valid, **changes}])
                with self.assertRaises(ValueError):
                    read_rows(self.score_file, "arshiya_score", "julia_score")
        self.write_rows([valid, valid])
        with self.assertRaisesRegex(ValueError, "Duplicate filename"):
            read_rows(self.score_file, "arshiya_score", "julia_score")


if __name__ == "__main__":
    unittest.main()
