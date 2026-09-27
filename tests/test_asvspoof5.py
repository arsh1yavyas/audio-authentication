"""ASVspoof 5 protocol-to-manifest safety and grouping tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hearsay.utils import read_manifest
from scripts.evaluate_asvspoof5 import _balanced_group_sample, _fold_ids
from scripts.prepare_asvspoof5 import build_manifest


class ASVspoofManifestTests(unittest.TestCase):
    def test_balanced_sampling_keeps_each_generator_and_speaker_group(self) -> None:
        labels = [0] * 12 + [1] * 24
        groups = ([f"speaker-{i}" for i in range(12)]
                  + [f"attack-{i % 4}" for i in range(24)])
        selected = _balanced_group_sample(np.asarray(labels), groups, 8, 5)
        self.assertEqual(int(np.sum(np.asarray(labels)[selected] == 0)), 8)
        self.assertEqual(int(np.sum(np.asarray(labels)[selected] == 1)), 8)
        self.assertEqual(len({groups[i] for i in selected if labels[i] == 0}), 8)
        self.assertEqual({groups[i] for i in selected if labels[i] == 1},
                         {f"attack-{i}" for i in range(4)})

    def test_grouped_folds_never_split_speakers_or_attack_families(self) -> None:
        labels = np.asarray([0] * 12 + [1] * 24)
        groups = ([f"speaker-{i // 2}" for i in range(12)]
                  + [f"attack-{i // 6}" for i in range(24)])
        fold_ids = _fold_ids(labels, groups, folds=3)
        for group in set(groups):
            self.assertEqual(len(set(fold_ids[np.asarray(groups) == group])), 1)

    def test_manifest_filters_missing_audio_and_groups_by_speaker_or_attack(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio" / "train"
            audio.mkdir(parents=True)
            (audio / "T_0000000001.flac").write_bytes(b"real")
            (audio / "T_0000000002.flac").write_bytes(b"spoof")
            protocol = root / "ASVspoof5.train.tsv"
            protocol.write_text(
                "T_0100 T_0000000001 F - - - AC1 bonafide bonafide -\n"
                "T_0100 T_0000000002 F - - - AC2 A07 spoof -\n"
                "T_0101 T_0000000003 M - - - AC1 A08 spoof -\n",
                encoding="utf-8",
            )
            output = root / "manifests" / "train.tsv"

            summary = build_manifest(protocol, audio, output)
            files, labels, groups = read_manifest(output)

            self.assertEqual(summary["clips"], 2)
            self.assertEqual(summary["real"], 1)
            self.assertEqual(summary["synthetic"], 1)
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(groups, ["asvspoof5:real:T_0100", "asvspoof5:spoof:A07"])
            self.assertEqual([path.name for path in files],
                             ["T_0000000001.flac", "T_0000000002.flac"])

    def test_manifest_rejects_path_like_utterance_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "audio"
            audio.mkdir()
            (audio / "T_0000000001.flac").write_bytes(b"placeholder")
            protocol = root / "protocol.tsv"
            protocol.write_text("T_0000 ../escape F - - - AC1 A01 spoof -\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe utterance"):
                build_manifest(protocol, audio, root / "manifest.tsv")


if __name__ == "__main__":
    unittest.main()
