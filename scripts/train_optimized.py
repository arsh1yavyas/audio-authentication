"""Fit the selected model and compare it with the saved forest on a group holdout.

Model selection uses only the training manifest. In each validation fold, one
synthetic generator and disjoint real-source groups are held out. The supplied
test manifest is used only after the model and its 0.5 cutoff are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold

from hearsay.features import FEATURE_NAMES
from hearsay.metadata import adjust_score, analyze_metadata
from hearsay.model import _fit_ensemble, feature_matrix, fit_optimized, load_model, save_json
from hearsay.routing import min_dcf
from hearsay.utils import read_manifest


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Report metrics using the Hearsay challenge's prior and error costs."""
    if len(set(labels)) != 2:
        raise ValueError("Evaluation needs both real and synthetic clips")
    matrix = confusion_matrix(labels, scores >= 0.5, labels=[0, 1])
    # Labels are 0=real, 1=synthetic and larger scores mean more synthetic.
    # The challenge uses Pspoof=.3, Cmiss=1, Cfa=4 and normalizes by .7.
    fixed_miss = matrix[1, 0] / matrix[1].sum()
    fixed_false_alarm = matrix[0, 1] / matrix[0].sum()
    fixed_dcf = (0.7 * fixed_miss + 1.2 * fixed_false_alarm) / 0.7
    challenge_min_dcf, min_dcf_threshold = min_dcf(labels, scores)
    return {
        "samples": len(labels),
        "real": int(sum(labels == 0)),
        "synthetic": int(sum(labels == 1)),
        "accuracy_at_0.5": float(np.trace(matrix) / len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "min_dcf": challenge_min_dcf,
        "min_dcf_threshold": min_dcf_threshold,
        "dcf_at_0.5": float(fixed_dcf),
        "min_dcf_config": {"p_spoof": 0.3, "c_miss": 1.0, "c_fa": 4.0},
        "confusion_matrix_real_synthetic": matrix.tolist(),
    }


def cross_validate(features: np.ndarray, labels: np.ndarray,
                   groups: list[str], files: list[Path], seed: int) -> dict:
    groups_array = np.asarray(groups)
    real = np.flatnonzero(labels == 0)
    synthetic = np.flatnonzero(labels == 1)
    synthetic_groups = sorted(set(groups_array[synthetic]))
    if len(synthetic_groups) < 2 or len(set(groups_array[real])) < len(synthetic_groups):
        raise ValueError("Need multiple synthetic generator groups and enough real groups for validation")
    real_folds = GroupKFold(n_splits=len(synthetic_groups)).split(
        real, labels[real], groups_array[real])
    candidate_scores = np.full(len(labels), np.nan)
    baseline_scores = np.full(len(labels), np.nan)
    fold_results = []
    for fold, (real_train, real_test) in enumerate(real_folds):
        generator = synthetic_groups[fold]
        test = np.r_[real[real_test], synthetic[groups_array[synthetic] == generator]]
        train = np.r_[real[real_train], synthetic[groups_array[synthetic] != generator]]
        optimized = fit_optimized(features[train], labels[train], seed)
        original = _fit_ensemble(features[train], labels[train], seed)
        candidate_audio = optimized.predict_proba(features[test])[:, 1]
        baseline_audio = original.predict_proba(features[test])[:, 1]
        candidate_scores[test] = [adjust_score(score, analyze_metadata(files[i]))
                                  for score, i in zip(candidate_audio, test)]
        baseline_scores[test] = [adjust_score(score, analyze_metadata(files[i]))
                                 for score, i in zip(baseline_audio, test)]
        fold_results.append({
            "held_out_generator": generator,
            "samples": len(test),
            "original_accuracy_at_0.5": float(np.mean((baseline_scores[test] >= 0.5) == labels[test])),
            "optimized_accuracy_at_0.5": float(np.mean((candidate_scores[test] >= 0.5) == labels[test])),
        })
        print(f"Validated fold {fold + 1}/{len(synthetic_groups)}: {generator}", flush=True)
    if not np.isfinite(candidate_scores).all() or not np.isfinite(baseline_scores).all():
        raise AssertionError("Every training clip must receive exactly one held-out score")
    return {
        "method": "one held-out synthetic generator per fold; disjoint real groups via GroupKFold",
        "folds": len(synthetic_groups),
        "original": summarize(labels, baseline_scores),
        "optimized": summarize(labels, candidate_scores),
        "by_generator": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/split/train.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/split/test.csv"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/split/split_features.npz"))
    parser.add_argument("--original-model", type=Path, default=Path("models/hearsay-split.joblib"))
    parser.add_argument("--model", type=Path, default=Path("models/hearsay-optimized.joblib"))
    parser.add_argument("--report", type=Path, default=Path("models/optimized-comparison.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_files, train_labels, train_groups = read_manifest(args.train_manifest)
    test_files, test_labels, test_groups = read_manifest(args.test_manifest)
    if train_groups is None or test_groups is None:
        raise ValueError("Both manifests need a group column")
    if set(train_files) & set(test_files) or set(train_groups) & set(test_groups):
        raise ValueError("Training and test manifests must have disjoint files and groups")
    matrix = feature_matrix(train_files + test_files, args.feature_cache)
    train_features = matrix[:len(train_files)]
    test_features = matrix[len(train_files):]

    selection = cross_validate(train_features, train_labels, train_groups,
                               train_files, args.seed)
    optimized = fit_optimized(train_features, train_labels, args.seed)
    original = load_model(args.original_model)
    optimized_audio = optimized.predict_proba(test_features)[:, 1]
    original_audio = original.predict_proba(test_features)[:, 1]
    metadata = [analyze_metadata(path) for path in test_files]
    optimized_scores = np.array([adjust_score(score, info)
                                 for score, info in zip(optimized_audio, metadata)])
    original_scores = np.array([adjust_score(score, info)
                                for score, info in zip(original_audio, metadata)])

    report = {
        "model": "75% standardized RBF SVM (C=1) plus 25% standardized logistic regression (C=0.1)",
        "seed": args.seed,
        "training_samples": len(train_files),
        "training_manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "selection_on_training_only": selection,
        "held_out_test": {
            "method": "provided test manifest, no fitting or model selection",
            "test_groups": sorted(set(test_groups)),
            "original": summarize(test_labels, original_scores),
            "optimized": summarize(test_labels, optimized_scores),
            "audio_only": {
                "original": summarize(test_labels, original_audio),
                "optimized": summarize(test_labels, optimized_audio),
            },
            "metadata_status_counts": {
                status: sum(info.status == status for info in metadata)
                for status in ("consistent", "inconsistent", "unknown")
            },
        },
        "min_dcf_definition": "Hearsay: (0.7 * miss + 1.2 * false_alarm) / 0.7, minimized over score thresholds (Pspoof=0.3, Cmiss=1, Cfa=4)",
        "caveat": "The test set holds out only PlayHT and Pro Diff generator families. All training metadata is consistent, while all held-out PlayHT containers are inconsistent. Shared synthetic speaker IDs and the LJ narrator limit generalization claims.",
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": optimized, "feature_names": FEATURE_NAMES,
                 "sample_rate": 16_000, "version": 3}, args.model)
    save_json(report, args.report)
    print(f"Saved optimized model to {args.model}")
    print(f"Saved comparison to {args.report}")


if __name__ == "__main__":
    main()
