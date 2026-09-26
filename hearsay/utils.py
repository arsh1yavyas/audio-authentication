"""Read labeled audio manifests and make reproducible train/test splits."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


LABELS = {"0": 0, "real": 0, "bonafide": 0, "bona fide": 0,
          "1": 1, "synthetic": 1, "spoof": 1, "fake": 1}


def read_manifest(path: Path) -> tuple[list[Path], np.ndarray, list[str] | None]:
    """Read filename,label[,group] with audio paths relative to the manifest."""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        if not reader.fieldnames or not {"filename", "label"}.issubset(reader.fieldnames):
            raise ValueError("Manifest needs filename and label columns")
        has_group = "group" in reader.fieldnames
        files, labels, groups = [], [], []
        for line, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not filename or label not in LABELS:
                raise ValueError(f"Invalid filename or label on manifest line {line}")
            audio_path = (path.parent / filename).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio on manifest line {line}: {audio_path}")
            files.append(audio_path)
            labels.append(LABELS[label])
            if has_group:
                group = (row.get("group") or "").strip()
                if not group:
                    raise ValueError(f"Missing group on manifest line {line}")
                groups.append(group)
    if len(files) != len(set(files)):
        raise ValueError("Manifest contains duplicate audio paths")
    if len(set(labels)) != 2:
        raise ValueError("Manifest must contain both real and synthetic examples")
    return files, np.asarray(labels, dtype=np.int8), groups if has_group else None


def split_data(labels: np.ndarray, groups: list[str] | None = None, *,
               test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray] | None:
    """Return disjoint train/test indices with both labels in each partition.

    Related clips stay together when groups are supplied. Return None when a
    useful two-class holdout cannot be made from the available data.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    labels = np.asarray(labels)
    if groups is not None and len(groups) != len(labels):
        raise ValueError("groups must have one entry per label")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (real) or 1 (synthetic)")
    if len(labels) < 10 or np.bincount(labels, minlength=2).min() < 3:
        return None
    indices = np.arange(len(labels))
    if groups is None:
        n_test = max(2, math.ceil(len(labels) * test_size))
        if n_test >= len(labels) or n_test < 2 or len(labels) - n_test < 2:
            return None
        try:
            train, test = train_test_split(indices, test_size=n_test, stratify=labels,
                                           random_state=seed)
        except ValueError:
            return None
        return np.sort(train), np.sort(test)

    unique_groups = set(groups)
    if len(unique_groups) < 4:
        return None
    n_test_groups = max(2, math.ceil(len(unique_groups) * test_size))
    if n_test_groups > len(unique_groups) - 2:
        return None
    class_counts = np.bincount(labels, minlength=2)
    best = None
    best_score = float("inf")
    for attempt in range(100):
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=n_test_groups,
                                            random_state=seed + attempt).split(indices, labels, groups))
        if len(set(labels[train])) == 2 and len(set(labels[test])) == 2:
            test_fraction_by_class = np.bincount(labels[test], minlength=2) / class_counts
            score = float(np.abs(test_fraction_by_class - test_size).sum())
            if score < best_score:
                best = np.sort(train), np.sort(test)
                best_score = score
    return best


def write_split_manifests(manifest: Path, output: Path, *, test_size: float = 0.2,
                          seed: int = 42) -> tuple[Path, Path]:
    """Write train.csv and test.csv referencing the original audio files."""
    files, labels, groups = read_manifest(manifest)
    split = split_data(labels, groups, test_size=test_size, seed=seed)
    if split is None:
        raise ValueError("Not enough examples or groups for a two-class train/test split")
    train_path, test_path = output / "train.csv", output / "test.csv"
    if manifest.resolve() in (train_path.resolve(), test_path.resolve()):
        raise ValueError("Output directory would overwrite the source manifest")
    output.mkdir(parents=True, exist_ok=True)
    columns = ("filename", "label", "group") if groups is not None else ("filename", "label")
    for destination, indices in zip((train_path, test_path), split):
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for index in indices:
                relative = Path(os.path.relpath(files[index], destination.parent)).as_posix()
                row = [relative, "synthetic" if labels[index] else "real"]
                if groups is not None:
                    row.append(groups[index])
                writer.writerow(row)
    return train_path, test_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a labeled audio manifest into train/test manifests")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for train.csv and test.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        train_path, test_path = write_split_manifests(
            args.manifest, args.output, test_size=args.test_size, seed=args.seed)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Training manifest: {train_path}")
    print(f"Test manifest: {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
