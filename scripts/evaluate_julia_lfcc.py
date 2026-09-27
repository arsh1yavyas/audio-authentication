"""Compare Julia's classifier with and without Arshiya LFCC features.

Every score is out-of-fold under a speaker/generator-grouped split. Model
selection and the final saved candidates use labeled training data only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import (FEATURE_NAMES, decode_audio, extract_features,
                              lowpass_audio)
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import METADATA_REAL_WEIGHT, adjust_score, analyze_metadata
from hearsay.model import _fit_ensemble, feature_matrix, fit_optimized, read_manifest
from hearsay.routing import min_dcf


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, threshold = min_dcf(labels, scores, p_spoof=0.3, c_miss=1.0, c_fa=4.0)
    return {
        "normalized_minDCF": cost,
        "minDCF_threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _extra_trees(seed: int, class_weight="balanced") -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        class_weight=class_weight, n_jobs=-1, random_state=seed,
    )


def _apply_metadata(scores: np.ndarray, files: list[Path]) -> np.ndarray:
    return np.asarray([
        adjust_score(score, analyze_metadata(path))
        for score, path in zip(scores, files)
    ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/arshiya_julia_lfcc_validation.json"))
    parser.add_argument("--raw-model", type=Path,
                        default=Path("models/arshiya_julia_lfcc_raw.joblib"))
    parser.add_argument("--lowpass-model", type=Path,
                        default=Path("models/arshiya_julia_lfcc_lowpass.joblib"))
    parser.add_argument("--extra-trees-model", type=Path,
                        default=Path("models/arshiya_lfcc_lowpass_candidate.joblib"))
    parser.add_argument("--cost-weighted-model", type=Path,
                        default=Path("models/arshiya_lfcc_lowpass_costweighted.joblib"))
    parser.add_argument("--lowpass-hz", type=float, default=7_000.0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Grouped Julia/LFCC evaluation requires a group column")
    group_array = np.asarray(groups)
    baseline = feature_matrix(files, args.feature_cache)
    lfcc = np.vstack([extract_lfcc_features(path) for path in files])
    combined = np.hstack((baseline, lfcc))

    commonband_baseline, commonband_lfcc = [], []
    for index, path in enumerate(files, start=1):
        filtered = lowpass_audio(decode_audio(path), args.lowpass_hz)
        commonband_baseline.append(extract_features(path, filtered))
        commonband_lfcc.append(extract_lfcc_features(path, filtered))
        if index % 100 == 0 or index == len(files):
            print(f"Prepared common-band features {index}/{len(files)}",
                  file=sys.stderr, flush=True)
    commonband = np.hstack((np.vstack(commonband_baseline),
                            np.vstack(commonband_lfcc)))

    names = ("julia_optimized_baseline", "julia_optimized_plus_lfcc",
             "julia_optimized_commonband_plus_lfcc", "arshiya_extra_trees_plus_lfcc",
             "arshiya_extra_trees_commonband_plus_lfcc",
             "cost_sensitive_extra_trees_commonband_plus_lfcc")
    out_of_fold = {name: np.full(len(labels), np.nan) for name in names}
    folds = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(baseline, labels, group_array), start=1):
        seed = 51_000 + fold
        for name, matrix in ((names[0], baseline), (names[1], combined),
                             (names[2], commonband)):
            classifier = fit_optimized(matrix[train_idx], labels[train_idx], seed)
            scores = classifier.predict_proba(matrix[valid_idx])[:, 1]
            out_of_fold[name][valid_idx] = _apply_metadata(
                scores, [files[index] for index in valid_idx])
        raw_extra = _extra_trees(seed).fit(combined[train_idx], labels[train_idx])
        out_of_fold[names[3]][valid_idx] = _apply_metadata(
            raw_extra.predict_proba(combined[valid_idx])[:, 1],
            [files[index] for index in valid_idx])
        lowpass_extra = _extra_trees(seed).fit(commonband[train_idx], labels[train_idx])
        out_of_fold[names[4]][valid_idx] = _apply_metadata(
            lowpass_extra.predict_proba(commonband[valid_idx])[:, 1],
            [files[index] for index in valid_idx])
        real_cost_per_clip = 4.0 * 0.7 / int(np.sum(labels[train_idx] == 0))
        synthetic_cost_per_clip = 1.0 * 0.3 / int(np.sum(labels[train_idx] == 1))
        cost_weighted = _extra_trees(seed, class_weight={
            0: real_cost_per_clip / synthetic_cost_per_clip, 1: 1.0,
        }).fit(commonband[train_idx], labels[train_idx])
        out_of_fold[names[5]][valid_idx] = _apply_metadata(
            cost_weighted.predict_proba(commonband[valid_idx])[:, 1],
            [files[index] for index in valid_idx])
        folds.append({
            "fold": fold,
            "held_groups": sorted(set(group_array[valid_idx])),
            "scores": {name: _metrics(labels[valid_idx], out_of_fold[name][valid_idx])
                       for name in names},
        })
        print(f"Completed grouped fold {fold}/{args.folds}", flush=True)

    if any(not np.isfinite(values).all() for values in out_of_fold.values()):
        raise AssertionError("Every sample must receive exactly one out-of-fold score")

    raw_bundle = {
        "model": fit_optimized(combined, labels, seed=52_000),
        "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": "Julia 75% RBF SVM plus 25% logistic regression",
        "lowpass_hz": None,
        "metadata_real_weight": METADATA_REAL_WEIGHT,
    }
    lowpass_bundle = {
        "model": fit_optimized(commonband, labels, seed=52_001),
        "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": "Julia 75% RBF SVM plus 25% logistic regression",
        "lowpass_hz": args.lowpass_hz,
        "metadata_real_weight": METADATA_REAL_WEIGHT,
    }
    extra_trees_bundle = {
        "model": _extra_trees(52_002).fit(commonband, labels),
        "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": "Arshiya ExtraTrees trained on common-band features",
        "lowpass_hz": args.lowpass_hz,
        "metadata_real_weight": METADATA_REAL_WEIGHT,
    }
    real_cost_per_clip = 4.0 * 0.7 / int(np.sum(labels == 0))
    synthetic_cost_per_clip = 1.0 * 0.3 / int(np.sum(labels == 1))
    cost_weighted_bundle = {
        "model": _extra_trees(52_003, class_weight={
            0: real_cost_per_clip / synthetic_cost_per_clip, 1: 1.0,
        }).fit(commonband, labels),
        "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": "Cost-sensitive ExtraTrees trained on common-band features",
        "lowpass_hz": args.lowpass_hz,
        "metadata_real_weight": METADATA_REAL_WEIGHT,
        "class_weights": {"real": real_cost_per_clip / synthetic_cost_per_clip,
                          "synthetic": 1.0},
    }
    for destination, bundle in ((args.raw_model, raw_bundle),
                                (args.lowpass_model, lowpass_bundle),
                                (args.extra_trees_model, extra_trees_bundle),
                                (args.cost_weighted_model, cost_weighted_bundle)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, destination)

    counts = {"real": int(np.sum(labels == 0)),
              "synthetic": int(np.sum(labels == 1))}
    metadata_counts = {
        status: sum(analyze_metadata(path).status == status for path in files)
        for status in ("consistent", "inconsistent", "unknown")
    }
    report = {
        "samples": len(files),
        "class_counts": counts,
        "groups": len(set(groups)),
        "validation": f"{args.folds}-fold StratifiedGroupKFold; random_state=2026",
        "metric": "normalized minDCF with Hearsay settings Pspoof=.3, Cmiss=1, Cfa=4",
        "models": {
            names[0]: "Julia 75% standardized RBF SVM / 25% standardized logistic regression on baseline features",
            names[1]: "same Julia model on concatenated baseline + LFCC features",
            names[2]: "same Julia model on 7 kHz common-band baseline + LFCC features",
            names[3]: "Arshiya ExtraTrees on concatenated baseline + LFCC features",
            names[4]: "Arshiya ExtraTrees on 7 kHz common-band baseline + LFCC features",
            names[5]: "Common-band ExtraTrees with per-clip class weights proportional to Hearsay error costs",
        },
        "metadata_status_counts": metadata_counts,
        "out_of_fold_results": {
            name: _metrics(labels, scores) for name, scores in out_of_fold.items()
        },
        "fold_details": folds,
        "artifacts": {
            "julia_raw_lfcc": str(args.raw_model),
            "julia_common_band_lfcc": str(args.lowpass_model),
            "best_extra_trees_common_band_lfcc": str(args.extra_trees_model),
            "cost_sensitive_common_band_lfcc": str(args.cost_weighted_model),
        },
        "common_band_cutoff_hz": args.lowpass_hz,
        "caveat": (
            "These results use only the local labeled manifest and grouped folds. "
            "The provided challenge test set has no labels here. LJ/DiffSSD source "
            "and bandwidth mismatch can inflate internal validation performance; "
            "treat this as candidate selection, not a challenge score guarantee."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["out_of_fold_results"], indent=2))
    print(f"Saved report: {args.report}")
    print(f"Saved raw LFCC model: {args.raw_model}")
    print(f"Saved common-band LFCC model: {args.lowpass_model}")
    print(f"Saved ExtraTrees candidate: {args.extra_trees_model}")
    print(f"Saved cost-weighted ExtraTrees candidate: {args.cost_weighted_model}")


if __name__ == "__main__":
    main()
