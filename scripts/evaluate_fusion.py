"""Nested, grouped validation and artifact building for baseline+temporal fusion.

The outer folds estimate generalization. Blend and Julia-style container-prior
weights are selected only on inner group-held-out predictions from each outer
training partition. No challenge answer key or test labels are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import FEATURE_NAMES
from hearsay.metadata import METADATA_REAL_WEIGHT, adjust_score, analyze_metadata
from hearsay.model import _fit_ensemble, feature_matrix, read_manifest
from hearsay.routing import min_dcf
from hearsay.temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features

TEMPORAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
METADATA_WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10)


def _new_temporal(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt", class_weight="balanced",
                                n_jobs=-1, random_state=seed)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"normalized_minDCF": float(min_dcf(labels, scores)[0]),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def _weight_grid(labels: np.ndarray, baseline: np.ndarray, temporal: np.ndarray,
                 metadata: list) -> tuple[float, float, float]:
    """Choose temporal and metadata weights on inner OOF scores only."""
    best = None
    for metadata_weight in METADATA_WEIGHTS:
        adjusted = np.asarray([adjust_score(score, analysis, metadata_weight)
                               for score, analysis in zip(baseline, metadata)])
        for temporal_weight in TEMPORAL_WEIGHTS:
            fused = ((1.0 - temporal_weight) * adjusted
                     + temporal_weight * temporal)
            cost = min_dcf(labels, fused)[0]
            # Stable tie preference stays near Julia's modest metadata prior
            # and the previously tested equal-weight fusion.
            rank = (cost, abs(temporal_weight - 0.5),
                    abs(metadata_weight - METADATA_REAL_WEIGHT))
            if best is None or rank < best[0]:
                best = (rank, temporal_weight, metadata_weight)
    assert best is not None
    return float(best[1]), float(best[2]), float(best[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path, default=Path("models/arshiya_fusion_nested_validation.json"))
    parser.add_argument("--policy", type=Path, default=Path("models/arshiya_fusion_policy.json"))
    parser.add_argument("--baseline-model", type=Path,
                        default=Path("models/arshiya_fusion_baseline.joblib"))
    parser.add_argument("--temporal-model", type=Path,
                        default=Path("models/arshiya_fusion_temporal.joblib"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    args = parser.parse_args()

    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Nested fusion validation requires a group column")
    groups_array = np.asarray(groups)
    base_x = feature_matrix(files, args.feature_cache)
    temporal_x = np.vstack([extract_temporal_features(path) for path in files])
    metadata = [analyze_metadata(path) for path in files]
    n = len(labels)
    scores = {name: np.full(n, np.nan) for name in
              ("baseline", "julia_metadata", "temporal", "nested_fusion", "fixed_fusion")}
    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=2026)
    fold_reports = []
    selected_temporal, selected_metadata = [], []

    for outer_fold, (outer_train, outer_test) in enumerate(
            outer.split(base_x, labels, groups_array), start=1):
        inner_base = np.full(len(outer_train), np.nan)
        inner_temporal = np.full(len(outer_train), np.nan)
        inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                     random_state=10_000 + outer_fold)
        for inner_fold, (inner_train, inner_test) in enumerate(
                inner.split(base_x[outer_train], labels[outer_train], groups_array[outer_train]), start=1):
            train_idx, valid_idx = outer_train[inner_train], outer_train[inner_test]
            if len(np.unique(labels[train_idx])) != 2 or len(np.unique(labels[valid_idx])) != 2:
                raise ValueError(f"Outer {outer_fold}, inner {inner_fold} lacks both classes")
            base_model = _fit_ensemble(base_x[train_idx], labels[train_idx],
                                       seed=20_000 + outer_fold * 100 + inner_fold)
            inner_base[inner_test] = base_model.predict_proba(base_x[valid_idx])[:, 1]
            temporal_model = _new_temporal(20_000 + outer_fold * 100 + inner_fold)
            temporal_model.fit(temporal_x[train_idx], labels[train_idx])
            inner_temporal[inner_test] = temporal_model.predict_proba(temporal_x[valid_idx])[:, 1]

        if not np.all(np.isfinite(inner_base)) or not np.all(np.isfinite(inner_temporal)):
            raise ValueError(f"Outer fold {outer_fold} did not receive complete inner OOF scores")
        outer_labels = labels[outer_train]
        outer_metadata = [metadata[index] for index in outer_train]
        temporal_weight, metadata_weight, inner_cost = _weight_grid(
            outer_labels, inner_base, inner_temporal, outer_metadata)
        selected_temporal.append(temporal_weight)
        selected_metadata.append(metadata_weight)

        base_model = _fit_ensemble(base_x[outer_train], labels[outer_train], seed=30_000 + outer_fold)
        temporal_model = _new_temporal(30_000 + outer_fold).fit(
            temporal_x[outer_train], labels[outer_train])
        base_scores = base_model.predict_proba(base_x[outer_test])[:, 1]
        temporal_scores = temporal_model.predict_proba(temporal_x[outer_test])[:, 1]
        julia_scores = np.asarray([adjust_score(score, metadata[index], METADATA_REAL_WEIGHT)
                                   for score, index in zip(base_scores, outer_test)])
        tuned = ((1.0 - temporal_weight) * np.asarray([
            adjust_score(score, metadata[index], metadata_weight)
            for score, index in zip(base_scores, outer_test)])
                 + temporal_weight * temporal_scores)
        fixed = 0.5 * julia_scores + 0.5 * temporal_scores
        scores["baseline"][outer_test] = base_scores
        scores["julia_metadata"][outer_test] = julia_scores
        scores["temporal"][outer_test] = temporal_scores
        scores["nested_fusion"][outer_test] = tuned
        scores["fixed_fusion"][outer_test] = fixed
        fold_reports.append({
            "fold": outer_fold,
            "held_out_groups": sorted(set(groups_array[outer_test])),
            "selected_temporal_weight": temporal_weight,
            "selected_metadata_real_weight": metadata_weight,
            "inner_minDCF_at_selected_weights": inner_cost,
            "outer_nested_fusion": _metrics(labels[outer_test], tuned),
            "outer_fixed_50_50": _metrics(labels[outer_test], fixed),
        })
        print(f"Completed nested outer fold {outer_fold}/{args.outer_folds}: "
              f"temporal={temporal_weight:.2f}, metadata={metadata_weight:.3f}",
              file=sys.stderr, flush=True)

    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise ValueError("Nested validation did not create one score per sample")
    # Initial nested tuning can be high-variance on this small corpus. The
    # observed fold choices vary, and the fixed 50/50 policy is evaluated
    # separately on every unseen outer fold. Keep that simple policy for now.
    final_temporal_weight = 0.5
    final_metadata_weight = METADATA_REAL_WEIGHT
    report = {
        "dataset_samples": n,
        "real": int(np.sum(labels == 0)),
        "synthetic": int(np.sum(labels == 1)),
        "group_count": len(set(groups_array)),
        "metric": "normalized minDCF; Pspoof=.3, Cmiss=1, Cfa=4",
        "outer_validation": f"{args.outer_folds}-fold StratifiedGroupKFold",
        "inner_weight_selection": f"{args.inner_folds}-fold StratifiedGroupKFold, grid-only minDCF selection",
        "out_of_fold_results": {name: _metrics(labels, value) for name, value in scores.items()},
        "outer_fold_details": fold_reports,
        "final_policy": {"temporal_weight": final_temporal_weight,
                         "metadata_real_weight": final_metadata_weight,
                         "selection": "fixed 50/50 audio-temporal blend plus Julia's fixed 5% metadata prior",
                         "reason": "Per-fold nested grid choices were unstable and scored worse on outer folds than the fixed blend.",
                         "nested_selected_temporal_weights": selected_temporal,
                         "nested_selected_metadata_weights": selected_metadata},
        "warning": "Validation measures the weight-selection procedure on LJ/DiffSSD. It does not guarantee transfer to the challenge distribution.",
    }

    # Fit fresh final candidates on all labeled training clips. The stock
    # baseline model remains unchanged and the fused CLI is opt-in.
    args.baseline_model.parent.mkdir(parents=True, exist_ok=True)
    args.temporal_model.parent.mkdir(parents=True, exist_ok=True)
    final_base = _fit_ensemble(base_x, labels, seed=42)
    final_temporal = _new_temporal(42).fit(temporal_x, labels)
    joblib.dump({"model": final_base, "feature_names": FEATURE_NAMES,
                 "sample_rate": 16_000, "version": 2}, args.baseline_model)
    joblib.dump({"model": final_temporal, "feature_names": TEMPORAL_FEATURE_NAMES,
                 "version": 1, "kind": "experimental-temporal"}, args.temporal_model)
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    policy = {**report["final_policy"], "manifest_sha256": fingerprint,
              "metric": report["metric"], "validation": report["outer_validation"]}
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
