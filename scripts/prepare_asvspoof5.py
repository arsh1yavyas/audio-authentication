"""Build safe grouped manifests from official ASVspoof 5 protocol files.

The manifests deliberately use only Track 1 train or development rows. A real
speaker is one validation group; a spoof attack family is another. This keeps
the group-held-out validation from learning the same speaker or generator on
both sides of a split.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import os
from pathlib import Path
import re


PROTOCOLS = {
    "train": "ASVspoof5.train.tsv",
    "dev": "ASVspoof5.dev.track_1.tsv",
}
SAFE_UTTERANCE = re.compile(r"^[TDE]_[0-9]{10}$")


def _audio_index(audio_root: Path) -> dict[str, Path]:
    """Index audio basenames once and reject ambiguous duplicate names."""
    index: dict[str, Path] = {}
    for path in audio_root.rglob("*.flac"):
        if path.name in index:
            raise ValueError(f"Duplicate FLAC basename under {audio_root}: {path.name}")
        index[path.name] = path.resolve()
    if not index:
        raise FileNotFoundError(f"No FLAC files found under {audio_root}")
    return index


def build_manifest(protocol: Path, audio_root: Path, output: Path) -> dict:
    """Convert a protocol into a manifest containing only audio present locally."""
    available = _audio_index(audio_root)
    written = []
    seen = set()
    counts: Counter[str] = Counter()
    groups: set[str] = set()

    with protocol.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 10:
                raise ValueError(f"Expected 10 protocol fields on line {line_number}")
            speaker, utterance, _gender, _codec, _quality, _seed, _condition, attack, key, _tmp = fields
            if not SAFE_UTTERANCE.fullmatch(utterance):
                raise ValueError(f"Unsafe utterance id on protocol line {line_number}")
            if key not in {"bonafide", "spoof"}:
                raise ValueError(f"Unrecognized key on protocol line {line_number}: {key}")
            filename = f"{utterance}.flac"
            audio = available.get(filename)
            if audio is None:
                continue
            if audio in seen:
                raise ValueError(f"Duplicate audio reference on protocol line {line_number}: {filename}")
            seen.add(audio)
            label = "real" if key == "bonafide" else "synthetic"
            group = (f"asvspoof5:real:{speaker}" if key == "bonafide"
                     else f"asvspoof5:spoof:{attack}")
            written.append((audio, label, group))
            counts[label] += 1
            groups.add(group)

    if not written or set(counts) != {"real", "synthetic"}:
        raise ValueError("The selected audio and protocol must contain both bona fide and spoof clips")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("filename", "label", "group"))
        for audio, label, group in written:
            writer.writerow((os.path.relpath(audio, output.parent.resolve()), label, group))

    return {
        "protocol": str(protocol),
        "audio_root": str(audio_root),
        "manifest": str(output),
        "clips": len(written),
        "real": counts["real"],
        "synthetic": counts["synthetic"],
        "groups": len(groups),
        "spoof_attack_families": sorted({group.rsplit(":", 1)[-1]
                                          for group in groups if ":spoof:" in group}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/external/asvspoof5"))
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Extracted official FLAC shard directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_manifest(args.protocol_dir / PROTOCOLS[args.partition],
                            args.audio_root, args.output)
    print(report)


if __name__ == "__main__":
    main()
