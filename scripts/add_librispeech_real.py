"""Add diverse real speakers and rebalance an existing LJ/DiffSSD sample.

Source: OpenSLR SLR31 Mini LibriSpeech dev-clean-2, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path


def priority(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


def add_librispeech(archive: Path, data_dir: Path, per_speaker: int) -> list[tuple[str, str, str]]:
    heaps: dict[str, list[tuple[int, str]]] = {}
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            parts = member.name.split("/")
            if (len(parts) != 5 or parts[:2] != ["LibriSpeech", "dev-clean-2"]
                    or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)):
                raise ValueError(f"Unexpected LibriSpeech audio path: {member.name}")
            speaker = parts[2]
            rank = priority(member.name)
            heap = heaps.setdefault(speaker, [])
            if len(heap) >= per_speaker and rank >= -heap[0][0]:
                continue
            destination = data_dir / "train" / "real_librispeech" / speaker / parts[-1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read {member.name}")
                with destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            if destination.stat().st_size != member.size:
                raise ValueError(f"Existing file differs from archive: {destination}")
            entry = (-rank, str(destination))
            if len(heap) < per_speaker:
                heapq.heappush(heap, entry)
            else:
                heapq.heapreplace(heap, entry)
    rows = []
    for speaker, heap in sorted(heaps.items()):
        for _, name in heap:
            path = Path(name)
            rows.append((path.relative_to(data_dir).as_posix(), "real", f"real:libri:{speaker}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--per-speaker", type=int, default=20)
    args = parser.parse_args()
    if args.per_speaker < 1:
        parser.error("--per-speaker must be positive")
    lj_files = sorted((args.data / "train" / "real").glob("LJ*.wav"))
    if not lj_files:
        raise ValueError("Prepare LJ/DiffSSD training data first")
    lj_rows = [(path.relative_to(args.data).as_posix(), "real", f"real:{path.name.split('-')[0]}")
               for path in lj_files]
    libri_rows = add_librispeech(args.archive, args.data, args.per_speaker)
    real_rows = lj_rows + libri_rows

    fake_root = args.data / "train" / "synthetic"
    candidates = {}
    for generator_dir in fake_root.iterdir():
        if generator_dir.is_dir():
            files = [path for path in generator_dir.rglob("*")
                     if path.is_file() and path.suffix.lower() in (".wav", ".mp3")]
            candidates[generator_dir.name] = sorted(files, key=lambda p: priority(p.relative_to(args.data).as_posix()))
    if not candidates:
        raise ValueError("No prepared DiffSSD candidates found")
    fake_rows = []
    positions = Counter()
    while len(fake_rows) < len(real_rows):
        advanced = False
        for generator in sorted(candidates):
            index = positions[generator]
            if index < len(candidates[generator]):
                path = candidates[generator][index]
                fake_rows.append((path.relative_to(args.data).as_posix(), "synthetic",
                                  f"synthetic:{generator}"))
                positions[generator] += 1
                advanced = True
                if len(fake_rows) == len(real_rows):
                    break
        if not advanced:
            break
    manifest = args.data / "train.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "label", "group"))
        writer.writerows(real_rows + fake_rows)
    summary = {
        "lj_real": len(lj_rows), "librispeech_real": len(libri_rows),
        "librispeech_speakers": len({row[2] for row in libri_rows}),
        "synthetic": len(fake_rows),
        "synthetic_per_generator": dict(Counter(row[2].split(":", 1)[1] for row in fake_rows)),
        "real_source": "OpenSLR SLR31 Mini LibriSpeech dev-clean-2; CC BY 4.0",
    }
    (args.data / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
