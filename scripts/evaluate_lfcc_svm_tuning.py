"""Nested group validation for the LFCC SVM on expanded LJ/Libri/DiffSSD data.

The outer grouped folds estimate performance. SVM regularization is selected
only from inner out-of-fold scores, avoiding tuning on the outer test fold.
The resulting single SVM is directly usable by the existing predict-lfcc CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import FEATURE_NAMES
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import METADATA_REAL_WEIGHT, adjust_score, analyze_metadata
from hearsay.model import feature_matrix, fit_optimized
from hearsay.routing import min_dcf
from hearsay.utils import read_manifest


def _classifier(c_value: float, seed: int) -> object:
    return make_pipeline(StandardScaler(), SVC(
        C=c_value, gamma="scale", class_weight="balanced", probability=True,
        random_state=seed,
    ))


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, threshold = min_dcf(labels, scores, p_spoof=0.3, c_miss=1.0, c_fa=4.0)
    return {
        "normalized_minDCF": float(cost),
        "minDCF_threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _score(model: object, features: np.ndarray, files: list[Path]) -> np.ndarray:
    raw = model.predict_proba(features)[:, 1]
    return np.asarray([
        adjust_score(score, analyze_metadata(path))
        for score, path in zip(raw, files)
    ], dtype=float)


def _select_c(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
              files: list[Path], candidates: list[float], folds: int,
              seed: int) -> tuple[float, dict[float, float]]:
    scores = {candidate: np.full(len(labels), np.nan, dtype=float)
              for candidate in candidates}
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(features, labels, groups), start=1):
        if len(np.unique(labels[train_idx])) != 2 or len(np.unique(labels[valid_idx])) != 2:
            raise ValueError("An inner grouped fold does not contain both classes")
        for candidate in candidates:
            model = _classifier(candidate, seed + inner_fold)
            model.fit(features[train_idx], labels[train_idx])
            scores[candidate][valid_idx] = _score(
                model, features[valid_idx], [files[index] for index in valid_idx])
    costs = {}
    for candidate, candidate_scores in scores.items():
        if not np.isfinite(candidate_scores).all():
            raise AssertionError("Inner folds did not score every training clip")
        costs[candidate] = float(min_dcf(labels, candidate_scores)[0])
    return min(costs, key=costs.get), costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--feature-cache", type=Path,
                        default=Path("data/librispeech_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/arshiya_lfcc_svm_tuning.json"))
    parser.add_argument("--model", type=Path,
                        default=Path("models/arshiya_librispeech_lfcc_svm.joblib"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--c-values", type=float, nargs="+",
                        default=(0.1, 0.3, 1.0, 3.0, 10.0))
    args = parser.parse_args()

    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Nested tuning requires a group column")
    if args.folds < 2 or args.inner_folds < 2:
        raise ValueError("Outer and inner validation each need at least two folds")
    candidates = sorted(set(args.c_values))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("All --c-values must be positive")

    baseline = feature_matrix(files, args.feature_cache)
    lfcc = np.vstack([extract_lfcc_features(path) for path in files])
    features = np.hstack((baseline, lfcc))
    group_array = np.asarray(groups)
    tuned_scores = np.full(len(labels), np.nan, dtype=float)
    fixed_scores = np.full(len(labels), np.nan, dtype=float)
    folds = []
    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (train_idx, valid_idx) in enumerate(
            outer.split(features, labels, group_array), start=1):
        chosen_c, inner_costs = _select_c(
            features[train_idx], labels[train_idx], group_array[train_idx],
            [files[index] for index in train_idx], candidates, args.inner_folds,
            seed=72_000 + fold,
        )
        tuned = _classifier(chosen_c, seed=73_000 + fold)
        tuned.fit(features[train_idx], labels[train_idx])
        tuned_scores[valid_idx] = _score(
            tuned, features[valid_idx], [files[index] for index in valid_idx])

        fixed = fit_optimized(features[train_idx], labels[train_idx], seed=73_000 + fold)
        fixed_scores[valid_idx] = _score(
            fixed, features[valid_idx], [files[index] for index in valid_idx])
        folds.append({
            "fold": fold,
            "held_out_groups": sorted(set(group_array[valid_idx])),
            "selected_c": chosen_c,
            "inner_grouped_minDCF_by_c": {str(key): value
                                          for key, value in inner_costs.items()},
            "held_out_samples": int(len(valid_idx)),
            "tuned_svm": _metrics(labels[valid_idx], tuned_scores[valid_idx]),
            "fixed_julia_svm_logistic": _metrics(labels[valid_idx], fixed_scores[valid_idx]),
        })
        print(f"Completed nested outer fold {fold}/{args.folds}; selected C={chosen_c:g}",
              file=sys.stderr, flush=True)

    if not np.isfinite(tuned_scores).all() or not np.isfinite(fixed_scores).all():
        raise AssertionError("Outer folds did not score every clip")

    # Select the deployable full-data C using group-disjoint OOF scores over
    # the training corpus, then refit on every labeled training row.
    final_c, full_inner_costs = _select_c(
        features, labels, group_array, files, candidates, args.folds, seed=74_000)
    final_model = _classifier(final_c, seed=75_000).fit(features, labels)
    bundle = {
        "model": final_model,
        "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": "Single standardized RBF SVM; C selected by grouped OOF minDCF",
        "lowpass_hz": None,
        "metadata_real_weight": METADATA_REAL_WEIGHT,
        "selected_c": final_c,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    report = {
        "samples": len(files),
        "real": int(np.sum(labels == 0)),
        "synthetic": int(np.sum(labels == 1)),
        "groups": len(set(groups)),
        "feature_count": len(FEATURE_NAMES) + len(LFCC_FEATURE_NAMES),
        "metric": "normalized minDCF; Pspoof=.3, Cmiss=1, Cfa=4",
        "validation": (
            f"Nested {args.folds}-fold outer / {args.inner_folds}-fold inner "
            "StratifiedGroupKFold; random_state=2026 for outer folds"
        ),
        "model_search": {
            "candidates": candidates,
            "gamma": "scale",
            "class_weight": "balanced",
            "selection_metric": "pooled inner OOF normalized minDCF",
        },
        "out_of_fold_results": {
            "nested_tuned_single_svm": _metrics(labels, tuned_scores),
            "fixed_julia_75_25_svm_logistic": _metrics(labels, fixed_scores),
        },
        "fold_details": folds,
        "final_model": {
            "selected_c_from_full_grouped_oof": final_c,
            "full_grouped_oof_minDCF_by_c": {str(key): value
                                              for key, value in full_inner_costs.items()},
            "artifact": str(args.model),
        },
        "caveat": (
            "The external LibriSpeech source contributes bona fide speech only; spoof clips "
            "remain from DiffSSD. OOF estimates are for model selection and may not predict "
            "the separate Hearsay challenge set."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["out_of_fold_results"], indent=2))
    print(f"Final grouped-OOF C: {final_c:g}; saved {args.model}")


if __name__ == "__main__":
    main()
