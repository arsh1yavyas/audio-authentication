"""Select and fit a model for the project's 70/30, 4:1 normalized minDCF.

Only training rows participate in model selection. The labeled holdout is
reported afterward as an exploratory check because earlier work has already
examined it.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from hearsay.features import FEATURE_NAMES, extract_features
from hearsay.metadata import adjust_score, analyze_metadata
from hearsay.model import feature_matrix, fit_mindcf, fit_optimized, save_json
from hearsay.routing import dcf_at_cutoff, min_dcf
from hearsay.utils import read_manifest
from scripts.train_optimized import summarize


def verified_existing_features(files: list[Path], cache: Path) -> tuple[np.ndarray, list[int]]:
    """Reuse an older cache only after bounded checks of source and row order.

    This explicit fallback is useful when the cache fingerprint changes after
    checkout even though the audio and feature vectors remain identical. It is
    weaker than recomputing every row, so the report records its use.
    """
    cache_mtime = cache.stat().st_mtime_ns
    if any(path.stat().st_mtime_ns > cache_mtime for path in files):
        raise ValueError("Audio has changed since the feature cache was written")
    with np.load(cache, allow_pickle=False) as saved:
        features = saved["features"].copy()
    if features.shape != (len(files), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Existing feature cache has incompatible or invalid rows")
    checked = np.unique(np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int))
    for index in checked:
        fresh = extract_features(files[int(index)])
        if not np.array_equal(fresh, features[int(index)]):
            raise ValueError(f"Existing feature cache differs at row {index}")
    return features, [int(index) for index in checked]


def real_fold_ids(groups: np.ndarray, real: np.ndarray, seed: int | None,
                  fold_count: int) -> np.ndarray:
    ids = np.full(len(groups), -1, dtype=int)
    if seed is None:
        for fold, (_, test) in enumerate(GroupKFold(n_splits=fold_count).split(
                real, groups=groups[real])):
            ids[real[test]] = fold
        return ids
    # Shuffle whole real groups, then place the largest groups in the smallest
    # bins. This keeps each repeat close to class balance without splitting a
    # speaker or source group.
    counts = Counter(groups[real])
    rng = np.random.default_rng(seed)
    order = list(counts)
    rng.shuffle(order)
    order.sort(key=lambda group: counts[group], reverse=True)
    sizes = np.zeros(fold_count, dtype=int)
    allocation = {}
    for group in order:
        smallest = np.flatnonzero(sizes == sizes.min())
        fold = int(rng.choice(smallest))
        allocation[group] = fold
        sizes[fold] += counts[group]
    ids[real] = [allocation[groups[i]] for i in real]
    return ids


def validate_partition(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       metadata: list, seed: int, partition: int) -> dict:
    real = np.flatnonzero(labels == 0)
    spoof = np.flatnonzero(labels == 1)
    generators = sorted(set(groups[spoof]))
    fold_ids = real_fold_ids(groups, real, None if partition == 0 else
                             (17, 29)[partition - 1], len(generators))
    current_scores = np.full(len(labels), np.nan)
    candidate_scores = np.full(len(labels), np.nan)
    for fold in range(len(generators)):
        generator = generators[(fold + partition) % len(generators)]
        test = np.r_[real[fold_ids[real] == fold], spoof[groups[spoof] == generator]]
        train = np.r_[real[fold_ids[real] != fold], spoof[groups[spoof] != generator]]
        current = fit_optimized(features[train], labels[train], seed)
        candidate = fit_mindcf(features[train], labels[train], seed)
        for scores, model in ((current_scores, current), (candidate_scores, candidate)):
            audio_scores = model.predict_proba(features[test])[:, 1]
            scores[test] = [adjust_score(score, metadata[i])
                            for score, i in zip(audio_scores, test)]
        print(f"Partition {partition}, fold {fold + 1}/{len(generators)}: {generator}",
              flush=True)
    if not np.isfinite(current_scores).all() or not np.isfinite(candidate_scores).all():
        raise AssertionError("Every training row must receive one out-of-fold score")
    current_cost, current_cutoff = min_dcf(labels, current_scores)
    candidate_cost, candidate_cutoff = min_dcf(labels, candidate_scores)
    current_summary = summarize(labels, current_scores)
    candidate_summary = summarize(labels, candidate_scores)
    if not (np.isclose(current_summary["min_dcf"], current_cost) and
            np.isclose(candidate_summary["min_dcf"], candidate_cost)):
        raise AssertionError("Training summary and minDCF cutoff use different cost definitions")
    return {
        "partition": partition,
        "current": current_summary,
        "candidate": candidate_summary,
        "current_min_dcf_cutoff": current_cutoff,
        "candidate_min_dcf_cutoff": candidate_cutoff,
        "direct_cost_check": {"current": current_cost, "candidate": candidate_cost},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/split/train.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/split/test.csv"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/split/split_features.npz"))
    parser.add_argument("--reuse-existing-cache", action="store_true",
                        help="Reuse an older cache after checking source mtimes and five recomputed rows")
    parser.add_argument("--model", type=Path, default=Path("models/hearsay-mindcf.joblib"))
    parser.add_argument("--report", type=Path, default=Path("models/mindcf-comparison.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_files, train_labels, train_groups = read_manifest(args.train_manifest)
    test_files, test_labels, test_groups = read_manifest(args.test_manifest)
    if train_groups is None or test_groups is None:
        raise ValueError("Both manifests need groups")
    if set(train_files) & set(test_files) or set(train_groups) & set(test_groups):
        raise ValueError("Training and test files and groups must be disjoint")
    all_files = train_files + test_files
    if args.reuse_existing_cache:
        features, checked_rows = verified_existing_features(all_files, args.feature_cache)
        feature_source = {"method": "existing cache with unchanged audio and five exact row checks",
                          "cache": str(args.feature_cache), "checked_rows": checked_rows,
                          "limitation": "The remaining rows were not recomputed in this run"}
    else:
        features = feature_matrix(all_files, args.feature_cache)
        feature_source = {"method": "fingerprint-validated cache or full re-extraction",
                          "cache": str(args.feature_cache)}
    train_features = features[:len(train_files)]
    test_features = features[len(train_files):]
    train_metadata = [analyze_metadata(path) for path in train_files]
    partitions = [
        validate_partition(train_features, train_labels, np.asarray(train_groups),
                           train_metadata, args.seed, partition)
        for partition in range(3)
    ]
    improved_all_partitions = all(
        item["candidate"]["min_dcf"] < item["current"]["min_dcf"]
        for item in partitions)

    candidate = fit_mindcf(train_features, train_labels, args.seed)
    # Retrain the baseline on these same training rows. A saved artifact may
    # include holdout clips, which would invalidate the comparison.
    current = fit_optimized(train_features, train_labels, args.seed)
    test_metadata = [analyze_metadata(path) for path in test_files]
    candidate_audio = candidate.predict_proba(test_features)[:, 1]
    current_audio = current.predict_proba(test_features)[:, 1]
    candidate_scores = np.asarray([adjust_score(score, info)
                                   for score, info in zip(candidate_audio, test_metadata)])
    current_scores = np.asarray([adjust_score(score, info)
                                 for score, info in zip(current_audio, test_metadata)])
    # Smooth threshold selection across the repeated, group-disjoint training
    # partitions before evaluating the labeled holdout.
    current_cutoff = float(np.median([item["current_min_dcf_cutoff"]
                                      for item in partitions]))
    candidate_cutoff = float(np.median([item["candidate_min_dcf_cutoff"]
                                        for item in partitions]))
    report = {
        "objective": "normalized minDCF with 70% real, 30% spoof, costs 4 for rejecting real and 1 for accepting spoof",
        "model": "standardized RBF SVM, C=1, gamma=scale, balanced class weights",
        "training_samples": len(train_files),
        "feature_source": feature_source,
        "training_manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "selection_on_training_only": {
            "method": (f"{len(set(np.asarray(train_groups)[train_labels == 1]))} held-out "
                       "synthetic generators paired with disjoint real groups "
                       "in each of three balanced partitions"),
            "partitions": partitions,
            "candidate_improved_all_partitions": improved_all_partitions,
            "operating_cutoff_method": "median of three training-only group-disjoint OOF minDCF cutoffs",
            "selected_operating_cutoffs": {"current": current_cutoff,
                                           "candidate": candidate_cutoff},
        },
        "held_out_test": {
            "status": "exploratory; this holdout was examined in earlier model work",
            "test_groups": sorted(set(test_groups)),
            "current": summarize(test_labels, current_scores),
            "candidate": summarize(test_labels, candidate_scores),
            "dcf_at_training_selected_cutoff": {
                "current": dcf_at_cutoff(test_labels, current_scores,
                                         current_cutoff),
                "candidate": dcf_at_cutoff(test_labels, candidate_scores,
                                           candidate_cutoff),
            },
            "audio_only": {
                "current": summarize(test_labels, current_audio),
                "candidate": summarize(test_labels, candidate_audio),
            },
        },
        "cost_assumptions": {
            "real_prior": 0.7, "spoof_prior": 0.3,
            "reject_real_cost": 4, "accept_spoof_cost": 1,
            "normalization": 0.3,
        },
        "caveat": "The labeled holdout has only PlayHT and Pro Diff as unseen generator families; LJ chapters share a narrator, some synthetic speaker IDs recur, and PlayHT metadata is inconsistent.",
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": candidate, "feature_names": FEATURE_NAMES,
                 "sample_rate": 16_000, "version": 3}, args.model)
    save_json(report, args.report)
    print(f"Saved minDCF model to {args.model}")
    print(f"Saved comparison to {args.report}")


if __name__ == "__main__":
    main()
