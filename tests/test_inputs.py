"""Input validation for routed scores and challenge archive preparation."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hearsay.__main__ import _read_external_scores
from scripts.prepare_test_data import main as prepare_test_data
from scripts.prepare_training_data import prepare_real, prepare_synthetic


def add_tar_file(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    tar.addfile(member, io.BytesIO(content))


class InputValidationTest(unittest.TestCase):
    def test_external_scores_reject_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scores = Path(temporary) / "scores.tsv"
            for value in ("nan", "inf", "-0.01", "1.01", ""):
                with self.subTest(value=value):
                    scores.write_text(f"filename\tjulia-score\nclip.wav\t{value}\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        _read_external_scores(scores, "julia-score")
            scores.write_text("filename\tjulia-score\nclip.wav\t0.75\n", encoding="utf-8")
            self.assertEqual(_read_external_scores(scores, "julia-score"), {"clip.wav": 0.75})

    def test_test_archive_rejects_windows_path_separators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive = folder / "test.tar"
            with tarfile.open(archive, "w") as tar:
                add_tar_file(tar, "HackGTHearsayTesting/HGT\\..\\..\\outside.wav", b"bad")
            with patch("sys.argv", ["prepare_test_data.py", "--archive", str(archive),
                                    "--output", str(folder / "data")]):
                with self.assertRaisesRegex(ValueError, "Unexpected TAR file"):
                    prepare_test_data()
            self.assertFalse((folder / "outside.wav").exists())

    def test_test_archive_detects_same_size_existing_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive = folder / "test.tar"
            with tarfile.open(archive, "w") as tar:
                add_tar_file(tar, "HackGTHearsayTesting/HGT123.wav", b"audio")
                add_tar_file(tar, "HackGTHearsayTesting/HGT_Hearsay_score_template.csv",
                             b"filename\tcm-score\nHGT123.wav\t0.006\n")
            output = folder / "data"
            args = ["prepare_test_data.py", "--archive", str(archive), "--output", str(output)]
            with patch("sys.argv", args):
                prepare_test_data()
            (output / "test" / "HGT123.wav").write_bytes(b"wrong")
            with patch("sys.argv", args):
                with self.assertRaisesRegex(ValueError, "differs from archive"):
                    prepare_test_data()

    def test_training_zip_wrapped_tar_and_existing_content_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tar_bytes = io.BytesIO()
            with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
                add_tar_file(tar, "resampled/LJ001-0001.wav", b"audio")
            archive = folder / "real.zip"
            with zipfile.ZipFile(archive, "w") as outer:
                outer.writestr("LJRealResampled.tar", tar_bytes.getvalue())
            output = folder / "data" / "train"
            self.assertEqual(len(prepare_real(archive, output)), 1)
            (output / "real" / "LJ001-0001.wav").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "differs from archive"):
                prepare_real(archive, output)

    def test_duplicate_synthetic_archive_member_cannot_enter_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive = folder / "fake.tar"
            name = "DiffSSD/generated_speech/generator/speaker/clip.wav"
            with tarfile.open(archive, "w") as tar:
                add_tar_file(tar, name, b"audio")
                add_tar_file(tar, name, b"audio")
            with self.assertRaisesRegex(ValueError, "Duplicate synthetic filenames"):
                prepare_synthetic(archive, folder / "data" / "train", 2, 2)


if __name__ == "__main__":
    unittest.main()
