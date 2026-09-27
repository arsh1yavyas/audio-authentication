"""Create group-held-out Arshiya scores for fusion with Julia's OOF scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.model import _fit_ensemble, feature_matrix, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/arshiya_oof.tsv"))
    parser.add_argument("--cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("OOF scoring requires a group column to prevent speaker/generator leakage")
    matrix = feature_matrix(files, args.cache)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = np.full((len(files), 3), np.nan, dtype=float)
    fold_ids = np.full(len(files), -1, dtype=int)
    for fold, (train, test) in enumerate(splitter.split(matrix, labels, groups), start=1):
        if len(np.unique(labels[train])) != 2:
            raise ValueError(f"Training side of fold {fold} does not contain both classes")
        model = _fit_ensemble(matrix[train], labels[train], args.seed + fold)
        scores[test, 0] = model.predict_proba(matrix[test])[:, 1]
        # Save the two internal experts too, enabling a measured route-vs-blend
        # comparison before Julia's independent scorer is available.
        for expert_index, (forest, columns, _weight) in enumerate(model.components, start=1):
            scores[test, expert_index] = forest.predict_proba(matrix[test][:, columns])[:, 1]
        fold_ids[test] = fold
        print(f"Scored held-out fold {fold}: {len(test)} clips", flush=True)
    if not np.all(np.isfinite(scores)) or np.any(fold_ids < 1):
        raise RuntimeError("Not every clip received an out-of-fold score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("filename", "label", "group", "fold", "arshiya_score",
                         "arshiya_full_score", "arshiya_spectral_score"))
        for path, label, group, fold, score, full, spectral in zip(
                files, labels, groups, fold_ids, scores[:, 0], scores[:, 1], scores[:, 2]):
            relative = path.relative_to(args.manifest.parent.resolve()).as_posix()
            writer.writerow((relative, int(label), group, int(fold), f"{score:.8f}",
                             f"{full:.8f}", f"{spectral:.8f}"))
    print(f"Wrote {len(files)} group-held-out scores to {args.output}")


if __name__ == "__main__":
    main()
