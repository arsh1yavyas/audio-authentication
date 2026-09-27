"""Compare merged Julia and Arshiya models on Julia's recorded group holdout.

The local manifest may not contain every validation source from Julia's run.
This evaluator uses only matching labeled rows from data/train.csv, reports any
missing validation groups, and trains local candidates with those groups held
out of fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import (FEATURE_NAMES, decode_audio, extract_features,
                              lowpass_audio)
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import adjust_score, analyze_metadata
from hearsay.model import _fit_ensemble, feature_matrix, fit_optimized, load_model
from hearsay.routing import min_dcf
from hearsay.utils import read_manifest


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, threshold = min_dcf(labels, scores, p_spoof=0.3, c_miss=1.0, c_fa=4.0)
    return {
        "normalized_minDCF": cost,
        "minDCF_threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _metadata_adjust(scores: np.ndarray, files: list[Path]) -> np.ndarray:
    return np.asarray([adjust_score(score, analyze_metadata(path))
                       for score, path in zip(scores, files)])


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--julia-report", type=Path,
                        default=Path("models/optimized-comparison.json"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/arshiya_merged_holdout_validation.json"))
    parser.add_argument("--seed", type=int, default=53_000)
    parser.add_argument("--lowpass-hz", type=float, default=7_000.0)
    args = parser.parse_args()

    recorded = json.loads(args.julia_report.read_text(encoding="utf-8"))
    expected_groups = set(recorded["held_out_test"]["test_groups"])
    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Merged holdout evaluation requires a group column")
    group_array = np.asarray(groups)
    valid = np.isin(group_array, list(expected_groups))
    train_idx, valid_idx = np.flatnonzero(~valid), np.flatnonzero(valid)
    if not len(valid_idx) or len(np.unique(labels[valid_idx])) != 2:
        raise ValueError("The local manifest has no two-class intersection with Julia's holdout")
    present_groups = set(group_array[valid_idx])
    baseline = feature_matrix(files, args.feature_cache)
    lfcc = np.vstack([extract_lfcc_features(path) for path in files])
    combined = np.hstack((baseline, lfcc))
    lowpass_baseline, lowpass_lfcc = [], []
    for index, path in enumerate(files, start=1):
        filtered = lowpass_audio(decode_audio(path), args.lowpass_hz)
        lowpass_baseline.append(extract_features(path, filtered))
        lowpass_lfcc.append(extract_lfcc_features(path, filtered))
        if index % 100 == 0 or index == len(files):
            print(f"Prepared common-band features {index}/{len(files)}",
                  file=sys.stderr, flush=True)
    commonband = np.hstack((np.vstack(lowpass_baseline), np.vstack(lowpass_lfcc)))

    train_y, valid_y = labels[train_idx], labels[valid_idx]
    train_files = [files[index] for index in train_idx]
    valid_files = [files[index] for index in valid_idx]
    valid_scores: dict[str, np.ndarray] = {}

    # Julia's saved artifacts were trained on its larger split and, per its
    # report, excluded all these holdout groups.
    for name, model_path in (
            ("julia_saved_optimized", Path("models/hearsay-optimized.joblib")),
            ("julia_saved_forest", Path("models/hearsay-split.joblib"))):
        model = load_model(model_path)
        audio = model.predict_proba(baseline[valid_idx])[:, 1]
        valid_scores[name] = _metadata_adjust(audio, valid_files)

    local_baseline = _fit_ensemble(baseline[train_idx], train_y, args.seed)
    valid_scores["local_julia_forest_baseline"] = _metadata_adjust(
        local_baseline.predict_proba(baseline[valid_idx])[:, 1], valid_files)

    for name, matrix in (
            ("local_julia_optimized_baseline", baseline),
            ("local_julia_optimized_plus_lfcc", combined),
            ("local_julia_optimized_commonband_plus_lfcc", commonband)):
        model = fit_optimized(matrix[train_idx], train_y, seed=args.seed)
        valid_scores[name] = _metadata_adjust(
            model.predict_proba(matrix[valid_idx])[:, 1], valid_files)

    extra = _extra_trees(args.seed).fit(combined[train_idx], train_y)
    valid_scores["local_extra_trees_plus_lfcc"] = _metadata_adjust(
        extra.predict_proba(combined[valid_idx])[:, 1], valid_files)
    commonband_extra = _extra_trees(args.seed).fit(commonband[train_idx], train_y)
    valid_scores["local_extra_trees_commonband_plus_lfcc"] = _metadata_adjust(
        commonband_extra.predict_proba(commonband[valid_idx])[:, 1], valid_files)

    report = {
        "validation_source": "group list from Julia's committed optimized-comparison.json",
        "labels_source": str(args.manifest),
        "training_examples": len(train_idx),
        "heldout_examples_available_locally": len(valid_idx),
        "heldout_class_counts": {
            "real": int(np.sum(valid_y == 0)),
            "synthetic": int(np.sum(valid_y == 1)),
        },
        "expected_groups": sorted(expected_groups),
        "groups_present_locally": sorted(present_groups),
        "groups_missing_locally": sorted(expected_groups - present_groups),
        "metric": "normalized minDCF; Pspoof=.3, Cmiss=1, Cfa=4",
        "results": {name: metrics(valid_y, scores)
                    for name, scores in valid_scores.items()},
        "caveat": (
            "Only the local LJ/DiffSSD rows matching Julia's recorded holdout "
            "groups are scored; five LibriSpeech groups are absent locally. "
            "Do not interpret this partial holdout as the full 293-clip report. "
            "Model selection using these labels can overfit this comparison."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))
    print(f"Heldout rows: {len(valid_idx)}; missing groups: "
          f"{len(expected_groups - present_groups)}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
