"""Paired grouped evaluation of the training corpus before/after LibriSpeech.

The test clips and held-out groups are identical for both systems. The baseline
is trained on the original LJ/DiffSSD manifest; the expanded system can also
use the additional real LibriSpeech speakers available in each training fold.
Neither the Hearsay answer key nor unlabeled challenge test files are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import FEATURE_NAMES
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import adjust_score, analyze_metadata
from hearsay.model import feature_matrix, fit_optimized
from hearsay.routing import min_dcf
from hearsay.utils import read_manifest


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, threshold = min_dcf(labels, scores, p_spoof=0.3, c_miss=1.0, c_fa=4.0)
    return {
        "normalized_minDCF": float(cost),
        "minDCF_threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _metadata_scores(scores: np.ndarray, files: list[Path]) -> np.ndarray:
    return np.asarray([
        adjust_score(score, analyze_metadata(path))
        for score, path in zip(scores, files)
    ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"),
                        help="Expanded LJ + LibriSpeech + DiffSSD manifest")
    parser.add_argument("--base-manifest", type=Path,
                        default=Path("data/train_pre_librispeech.csv"),
                        help="Saved original LJ + DiffSSD manifest")
    parser.add_argument("--feature-cache", type=Path,
                        default=Path("data/librispeech_features.npz"))
    parser.add_argument("--base-feature-cache", type=Path,
                        default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/arshiya_librispeech_expansion_comparison.json"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    full_files, full_labels, full_groups = read_manifest(args.manifest)
    base_files, base_labels, base_groups = read_manifest(args.base_manifest)
    if full_groups is None or base_groups is None:
        raise ValueError("Both manifests must contain source/speaker/generator groups")
    if args.folds < 2:
        raise ValueError("At least two folds are required")

    full_groups_array = np.asarray(full_groups)
    base_groups_array = np.asarray(base_groups)
    full_baseline = feature_matrix(full_files, args.feature_cache)
    base_baseline = feature_matrix(base_files, args.base_feature_cache)
    full_lfcc = np.vstack([extract_lfcc_features(path) for path in full_files])
    base_lfcc = np.vstack([extract_lfcc_features(path) for path in base_files])
    full_matrix = np.hstack((full_baseline, full_lfcc))
    base_matrix = np.hstack((base_baseline, base_lfcc))

    old_scores = np.full(len(full_labels), np.nan, dtype=float)
    libri_only_scores = np.full(len(full_labels), np.nan, dtype=float)
    expanded_scores = np.full(len(full_labels), np.nan, dtype=float)
    fold_details = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(full_matrix, full_labels, full_groups_array), start=1):
        held_groups = set(full_groups_array[valid_idx])
        old_train_idx = np.flatnonzero(~np.isin(base_groups_array, list(held_groups)))
        if len(np.unique(base_labels[old_train_idx])) != 2:
            raise ValueError(f"The original-corpus training fold {fold} contains one class")

        libri_train_idx = np.asarray([
            index for index, group in enumerate(full_groups_array)
            if group.startswith("real:libri:") and group not in held_groups
        ], dtype=int)
        libri_augmented_features = np.vstack((base_matrix[old_train_idx],
                                               full_matrix[libri_train_idx]))
        libri_augmented_labels = np.concatenate((base_labels[old_train_idx],
                                                  full_labels[libri_train_idx]))

        old_model = fit_optimized(base_matrix[old_train_idx], base_labels[old_train_idx],
                                  seed=61_000 + fold)
        libri_only_model = fit_optimized(libri_augmented_features,
                                         libri_augmented_labels, seed=61_000 + fold)
        expanded_model = fit_optimized(full_matrix[train_idx], full_labels[train_idx],
                                       seed=61_000 + fold)
        valid_files = [full_files[index] for index in valid_idx]
        old_scores[valid_idx] = _metadata_scores(
            old_model.predict_proba(full_matrix[valid_idx])[:, 1], valid_files)
        libri_only_scores[valid_idx] = _metadata_scores(
            libri_only_model.predict_proba(full_matrix[valid_idx])[:, 1], valid_files)
        expanded_scores[valid_idx] = _metadata_scores(
            expanded_model.predict_proba(full_matrix[valid_idx])[:, 1], valid_files)

        fold_details.append({
            "fold": fold,
            "held_out_groups": sorted(held_groups),
            "held_out_samples": int(len(valid_idx)),
            "held_out_real": int(np.sum(full_labels[valid_idx] == 0)),
            "held_out_synthetic": int(np.sum(full_labels[valid_idx] == 1)),
            "original_training_samples": int(len(old_train_idx)),
            "librispeech_added_training_samples": int(len(libri_train_idx)),
            "librispeech_augmented_training_samples": int(len(libri_augmented_labels)),
            "expanded_training_samples": int(len(train_idx)),
            "original_corpus": _metrics(full_labels[valid_idx], old_scores[valid_idx]),
            "librispeech_added_only": _metrics(full_labels[valid_idx],
                                                libri_only_scores[valid_idx]),
            "expanded_corpus": _metrics(full_labels[valid_idx], expanded_scores[valid_idx]),
        })
        print(f"Completed paired fold {fold}/{args.folds}", file=sys.stderr, flush=True)

    if (not np.isfinite(old_scores).all() or not np.isfinite(libri_only_scores).all()
            or not np.isfinite(expanded_scores).all()):
        raise AssertionError("Every clip must receive exactly one out-of-fold score")

    report = {
        "dataset": {
            "original_samples": len(base_files),
            "original_real": int(np.sum(base_labels == 0)),
            "original_synthetic": int(np.sum(base_labels == 1)),
            "expanded_samples": len(full_files),
            "expanded_real": int(np.sum(full_labels == 0)),
            "expanded_synthetic": int(np.sum(full_labels == 1)),
            "expanded_real_group_counts": {
                "lj_chapters": len({group for group, label in zip(full_groups, full_labels)
                                     if label == 0 and group.startswith("real:LJ")}),
                "librispeech_speakers": len({group for group, label in zip(full_groups, full_labels)
                                              if label == 0 and group.startswith("real:libri:")}),
            },
            "expanded_group_count": len(set(full_groups)),
        },
        "validation": (
            f"Paired {args.folds}-fold StratifiedGroupKFold, random_state=2026; "
            "same held-out groups and clips for both training corpora"
        ),
        "metric": "normalized minDCF; Pspoof=.3, Cmiss=1, Cfa=4",
        "score_direction": "higher means synthetic",
        "model": "Julia 75% standardized RBF SVM / 25% standardized logistic regression on 84 baseline + 120 LFCC features",
        "out_of_fold_results": {
            "original_lj_diffssd_training": _metrics(full_labels, old_scores),
            "librispeech_added_same_lj_diffssd_training": _metrics(full_labels,
                                                                    libri_only_scores),
            "expanded_lj_librispeech_diffssd_training": _metrics(full_labels, expanded_scores),
        },
        "fold_details": fold_details,
        "caveat": (
            "Mini LibriSpeech adds real speakers only; the synthetic examples remain from DiffSSD. "
            "This validates real-source diversity and generator-held-out performance, not transfer "
            "to unseen synthesis toolkits. The challenge test set and answer key were not used."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["out_of_fold_results"], indent=2))
    print(f"Saved paired comparison to {args.report}")


if __name__ == "__main__":
    main()
