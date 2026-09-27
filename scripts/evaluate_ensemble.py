"""Compare forest weights on held-out generator families without saving a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from hearsay.features import FEATURE_NAMES
from hearsay.model import feature_matrix
from hearsay.utils import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42", help="Comma-separated real-speaker holdout seeds")
    parser.add_argument("--generator", help="Evaluate one synthetic group, such as synthetic:grad_tts")
    parser.add_argument("--output", type=Path, default=Path("models/ensemble_holdout.json"))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    files, labels, groups = read_manifest(Path("data/train.csv"))
    matrix = feature_matrix(files, Path("data/train_features.npz"))
    groups = np.asarray(groups)
    real = np.flatnonzero(labels == 0)
    fake = np.flatnonzero(labels == 1)
    spectral = [i for i, name in enumerate(FEATURE_NAMES) if not name.startswith(("mfcc_", "chroma_"))]
    generators = sorted(set(groups[fake]))
    if args.generator:
        if args.generator not in generators:
            raise ValueError(f"Unknown generator group: {args.generator}")
        generators = [args.generator]
    by_weight: dict[str, dict[str, float]] = {str(w): {} for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
    for seed in seeds:
        # Hold out LibriSpeech speakers, LJ chapters, and each synthetic
        # generator as groups. AUC does not depend on a classification cutoff.
        train_real, test_real = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                       random_state=seed).split(real, labels[real], groups[real]))
        train_real, test_real = real[train_real], real[test_real]
        for generator in generators:
            train_index = np.r_[train_real, fake[groups[fake] != generator]]
            test_index = np.r_[test_real, fake[groups[fake] == generator]]
            predictions = []
            for columns in (list(range(len(FEATURE_NAMES))), spectral):
                model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               max_features="sqrt", class_weight="balanced_subsample",
                                               random_state=42, n_jobs=-1)
                model.fit(matrix[train_index][:, columns], labels[train_index])
                predictions.append(model.predict_proba(matrix[test_index][:, columns])[:, 1])
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                score = weight * predictions[0] + (1 - weight) * predictions[1]
                by_weight[str(weight)][f"{seed}:{generator}"] = round(float(roc_auc_score(labels[test_index], score)), 4)
            print(f"Evaluated seed {seed}, {generator}", flush=True)
    summary = {}
    for weight, folds in by_weight.items():
        values = list(folds.values())
        summary[weight] = {"mean_auc": round(float(np.mean(values)), 4),
                           "worst_auc": round(float(np.min(values)), 4),
                           "by_generator": folds}
        print(f"full-feature weight {weight}: mean={summary[weight]['mean_auc']}, "
              f"worst={summary[weight]['worst_auc']}")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
