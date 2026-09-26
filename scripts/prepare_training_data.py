"""Create a balanced training manifest from LJRealResampled and DiffSSD ZIPs.

The DiffSSD archive is much larger than the real archive. This streams it once,
keeping a deterministic hash sample of WAV files from each generator family.
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
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

SAFE_PART = re.compile(r"[A-Za-z0-9_.-]+\Z")
MAX_AUDIO_SIZE = 30_000_000


def nested_tar(zip_path: Path, expected_name: str):
    """Return streaming context managers to avoid unpacking the whole TAR."""
    outer = zipfile.ZipFile(zip_path)
    names = outer.namelist()
    if names != [expected_name]:
        outer.close()
        raise ValueError(f"Expected only {expected_name} inside {zip_path}")
    return outer


def safe_parts(name: str) -> tuple[str, ...]:
    parts = PurePosixPath(name).parts
    if not parts or any(part in (".", "..") or not SAFE_PART.fullmatch(part) for part in parts):
        raise ValueError(f"Unsafe archive path: {name}")
    return parts


def copy_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    if member.size > MAX_AUDIO_SIZE:
        raise ValueError(f"Unusually large audio file: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != member.size:
            raise ValueError(f"Existing file differs from archive: {destination}")
        return
    source = tar.extractfile(member)
    if source is None:
        raise ValueError(f"Cannot read {member.name}")
    with destination.open("xb") as target:
        shutil.copyfileobj(source, target)


def prepare_real(archive: Path, output: Path) -> list[tuple[str, str, str]]:
    rows = []
    with nested_tar(archive, "LJRealResampled.tar") as outer:
        with outer.open("LJRealResampled.tar") as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = safe_parts(member.name)
                    if len(parts) != 2 or parts[0] != "resampled" or not re.fullmatch(r"LJ\d{3}-\d{4}\.wav", parts[1]):
                        raise ValueError(f"Unexpected LJ file: {member.name}")
                    destination = output / "real" / parts[1]
                    copy_member(tar, member, destination)
                    rows.append((str(destination.relative_to(output.parent)).replace("\\", "/"),
                                 "real", f"real:{parts[1].split('-')[0]}"))
    if len(rows) != len({row[0] for row in rows}):
        raise ValueError("Duplicate real filenames")
    return rows


def prepare_synthetic(archive: Path, output: Path, target_count: int,
                      candidates_per_generator: int) -> tuple[list[tuple[str, str, str]], dict]:
    # A max heap for each generator: the worst selected hash is at index zero.
    heaps: dict[str, list[tuple[int, str]]] = {}
    counts: Counter[str] = Counter()
    ignored = 0
    with nested_tar(archive, "DiffSSD.tar") as outer:
        with outer.open("DiffSSD.tar") as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parts = safe_parts(member.name)
                    if (len(parts) not in (4, 5) or parts[:2] != ("DiffSSD", "generated_speech")
                            or Path(parts[-1]).suffix.lower() not in (".wav", ".mp3")):
                        ignored += 1
                        continue
                    if len(parts) == 5:
                        generator, speaker, filename = parts[2:]
                    else:
                        generator, filename = parts[2:]
                        speaker = "_no_speaker"
                    counts[generator] += 1
                    total = sum(counts.values())
                    if total % 5000 == 0:
                        print(f"Scanned {total} synthetic audio files across {len(counts)} generators", flush=True)
                    priority = int.from_bytes(hashlib.blake2b(member.name.encode(), digest_size=8).digest(), "big")
                    heap = heaps.setdefault(generator, [])
                    if len(heap) >= candidates_per_generator and priority >= -heap[0][0]:
                        continue
                    destination = output / "synthetic" / generator / speaker / filename
                    copy_member(tar, member, destination)
                    entry = (-priority, str(destination))
                    if len(heap) < candidates_per_generator:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heapreplace(heap, entry)
    if not heaps:
        raise ValueError("No synthetic audio files found")

    ranked = {
        gen: sorted(((-priority, Path(path)) for priority, path in heap), key=lambda item: item[0])
        for gen, heap in heaps.items()
    }
    # Round-robin allocation keeps generator families represented even when
    # their original archive sizes differ substantially.
    selected: list[tuple[str, Path]] = []
    positions = Counter()
    while len(selected) < target_count:
        advanced = False
        for generator in sorted(ranked):
            pos = positions[generator]
            if pos < len(ranked[generator]):
                selected.append((generator, ranked[generator][pos][1]))
                positions[generator] += 1
                advanced = True
                if len(selected) == target_count:
                    break
        if not advanced:
            break
    rows = [(str(path.relative_to(output.parent)).replace("\\", "/"),
             "synthetic", f"synthetic:{generator}") for generator, path in selected]
    summary = {"archive_audio_per_generator": dict(counts),
               "selected_per_generator": dict(Counter(gen for gen, _ in selected)),
               "ignored_other_files": ignored}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-archive", type=Path, required=True)
    parser.add_argument("--synthetic-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--synthetic-ratio", type=int, default=2)
    parser.add_argument("--candidates-per-generator", type=int, default=500)
    args = parser.parse_args()
    if args.synthetic_ratio < 1 or args.candidates_per_generator < 1:
        parser.error("Ratio and candidates-per-generator must be positive")
    train_dir = args.output / "train"
    real_rows = prepare_real(args.real_archive, train_dir)
    print(f"Prepared {len(real_rows)} real WAVs", flush=True)
    fake_rows, summary = prepare_synthetic(
        args.synthetic_archive, train_dir,
        target_count=len(real_rows) * args.synthetic_ratio,
        candidates_per_generator=args.candidates_per_generator,
    )
    manifest = args.output / "train.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "label", "group"))
        writer.writerows(real_rows + fake_rows)
    summary.update({"real_selected": len(real_rows), "synthetic_selected": len(fake_rows),
                    "manifest": str(manifest)})
    (args.output / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
