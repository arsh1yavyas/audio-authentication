"""Compare LFCC models under the 70/30, 4:1 normalized minDCF objective.

Model selection uses only the training manifest. Whole real groups and whole
synthetic generator families are held out together in three partitions. The
separate labeled test manifest is evaluated after model selection and remains
an exploratory check because it has already been examined in this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import FEATURE_NAMES, extract_features
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import METADATA_REAL_WEIGHT, adjust_score, analyze_metadata
from hearsay.model import feature_matrix, fit_optimized
from hearsay.routing import dcf_at_cutoff, min_dcf
from hearsay.utils import read_manifest


def lfcc_matrix(files: list[Path], cache: Path, workers: int) -> np.ndarray:
    """Cache LFCCs only when code, paths, sizes, mtimes, and order still match."""
    source_dir = Path(__file__).resolve().parents[1] / "hearsay"
    signature = hashlib.sha256((source_dir / "lfcc.py").read_bytes())
    signature.update((source_dir / "features.py").read_bytes())
    for path in files:
        stat = path.stat()
        signature.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    fingerprint = signature.hexdigest()
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                matrix = saved["features"]
                if matrix.shape == (len(files), len(LFCC_FEATURE_NAMES)) and np.isfinite(matrix).all():
                    print(f"Loaded cached LFCCs from {cache}", file=sys.stderr)
                    return matrix

    partial = cache.with_suffix(".partial.npz")
    vectors = []
    if partial.is_file():
        with np.load(partial, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                earlier = saved["features"]
                if (earlier.ndim == 2 and earlier.shape[1] == len(LFCC_FEATURE_NAMES)
                        and len(earlier) <= len(files) and np.isfinite(earlier).all()):
                    vectors = list(earlier)
                    print(f"Resuming LFCC extraction at {len(vectors)}/{len(files)}",
                          file=sys.stderr, flush=True)

    def save_partial() -> None:
        partial.parent.mkdir(parents=True, exist_ok=True)
        temporary = partial.with_suffix(".tmp.npz")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, features=np.vstack(vectors), fingerprint=fingerprint)
        temporary.replace(partial)

    if workers == 1:
        extracted = map(extract_lfcc_features, files[len(vectors):])
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        extracted = pool.map(extract_lfcc_features, files[len(vectors):])
    try:
        for index, vector in enumerate(extracted, start=len(vectors) + 1):
            vectors.append(vector)
            if index % 50 == 0 or index == len(files):
                save_partial()
                print(f"Extracted LFCCs from {index}/{len(files)} clips", file=sys.stderr,
                      flush=True)
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    matrix = np.vstack(vectors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as stream:
        np.savez_compressed(stream, features=matrix, fingerprint=fingerprint)
    partial.unlink(missing_ok=True)
    return matrix


def verified_baseline_matrix(files: list[Path], cache: Path) -> tuple[np.ndarray, list[int]]:
    """Explicitly reuse an old cache after source-time and row checks."""
    cache_mtime = cache.stat().st_mtime_ns
    if any(path.stat().st_mtime_ns > cache_mtime for path in files):
        raise ValueError("Audio has changed since the baseline feature cache was written")
    with np.load(cache, allow_pickle=False) as saved:
        matrix = saved["features"].copy()
    if matrix.shape != (len(files), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Existing baseline feature cache has invalid rows")
    checked = np.unique(np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int))
    for index in checked:
        if not np.array_equal(extract_features(files[int(index)]), matrix[int(index)]):
            raise ValueError(f"Existing baseline feature cache differs at row {index}")
    return matrix, [int(index) for index in checked]


def real_fold_ids(groups: np.ndarray, real: np.ndarray, partition: int,
                  fold_count: int) -> np.ndarray:
    """Assign whole real groups to folds with three deterministic allocations."""
    ids = np.full(len(groups), -1, dtype=int)
    if partition == 0:
        for fold, (_, test) in enumerate(GroupKFold(n_splits=fold_count).split(
                real, groups=groups[real])):
            ids[real[test]] = fold
        return ids
    counts = Counter(groups[real])
    rng = np.random.default_rng((17, 29)[partition - 1])
    order = list(counts)
    rng.shuffle(order)
    order.sort(key=lambda group: counts[group], reverse=True)
    sizes = np.zeros(fold_count, dtype=int)
    allocation: dict[str, int] = {}
    for group in order:
        choices = np.flatnonzero(sizes == sizes.min())
        fold = int(rng.choice(choices))
        allocation[group] = fold
        sizes[fold] += counts[group]
    ids[real] = [allocation[groups[i]] for i in real]
    return ids


def fit_model(name: str, matrix: np.ndarray, labels: np.ndarray, seed: int):
    if name in ("optimized_84", "optimized_204"):
        return fit_optimized(matrix, labels, seed)
    if name == "svm_204":
        class_weight = "balanced"
    elif name == "cost_weighted_svm_204":
        # Relative per-clip cost under 70% real, 30% spoof and 4:1 errors.
        n_real = int(np.sum(labels == 0))
        n_spoof = int(np.sum(labels == 1))
        class_weight = {0: (4.0 * 0.7 / n_real) / (0.3 / n_spoof), 1: 1.0}
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(StandardScaler(), SVC(
        C=1.0, gamma="scale", class_weight=class_weight,
        probability=True, random_state=seed,
    )).fit(matrix, labels)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, cutoff = min_dcf(labels, scores)
    return {"normalized_min_dcf": cost, "min_dcf_cutoff": cutoff,
            "roc_auc": float(roc_auc_score(labels, scores))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/split/train.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/split/test.csv"))
    parser.add_argument("--baseline-cache", type=Path,
                        default=Path("data/split/split_features.npz"))
    parser.add_argument("--reuse-existing-baseline-cache", action="store_true",
                        help="Verify five rows before reusing a cache with an old code fingerprint")
    parser.add_argument("--lfcc-cache", type=Path,
                        default=Path("data/split/lfcc_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/lfcc_mindcf_comparison.json"))
    parser.add_argument("--oof-scores", type=Path,
                        default=Path("models/lfcc_mindcf_oof.npz"))
    parser.add_argument("--model", type=Path,
                        default=Path("models/hearsay-lfcc-mindcf-candidate.joblib"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    train_files, train_labels, train_groups = read_manifest(args.train_manifest)
    test_files, test_labels, test_groups = read_manifest(args.test_manifest)
    if train_groups is None or test_groups is None:
        raise ValueError("Both manifests need groups")
    if set(train_files) & set(test_files) or set(train_groups) & set(test_groups):
        raise ValueError("Training and test files and groups must be disjoint")
    files = train_files + test_files
    if args.extract_only:
        lfcc_matrix(files, args.lfcc_cache, args.workers)
        return
    if args.reuse_existing_baseline_cache:
        baseline, checked_baseline_rows = verified_baseline_matrix(
            files, args.baseline_cache)
    else:
        baseline = feature_matrix(files, args.baseline_cache)
        checked_baseline_rows = None
    lfcc = lfcc_matrix(files, args.lfcc_cache, args.workers)
    train_count = len(train_files)
    matrices = {"optimized_84": baseline[:train_count]}
    combined = np.hstack((baseline, lfcc))
    for name in ("optimized_204", "svm_204", "cost_weighted_svm_204"):
        matrices[name] = combined[:train_count]
    groups = np.asarray(train_groups)
    real = np.flatnonzero(train_labels == 0)
    spoof = np.flatnonzero(train_labels == 1)
    generators = sorted(set(groups[spoof]))
    if len(generators) < 2 or len(set(groups[real])) < len(generators):
        raise ValueError("Need multiple generator and real groups for validation")
    train_metadata = [analyze_metadata(path) for path in train_files]
    partitions = []
    oof_score_sets = {name: [] for name in matrices}
    fold_assignments = []
    for partition in range(3):
        fold_ids = real_fold_ids(groups, real, partition, len(generators))
        validation_fold = np.full(train_count, -1, dtype=int)
        oof = {name: np.full(train_count, np.nan) for name in matrices}
        for fold in range(len(generators)):
            generator = generators[(fold + partition) % len(generators)]
            valid = np.r_[real[fold_ids[real] == fold], spoof[groups[spoof] == generator]]
            train = np.r_[real[fold_ids[real] != fold], spoof[groups[spoof] != generator]]
            validation_fold[valid] = fold
            for name, matrix in matrices.items():
                model = fit_model(name, matrix[train], train_labels[train], 42)
                raw = model.predict_proba(matrix[valid])[:, 1]
                oof[name][valid] = [adjust_score(score, train_metadata[i])
                                    for score, i in zip(raw, valid)]
            print(f"Partition {partition + 1}/3 fold {fold + 1}/{len(generators)}: "
                  f"{generator}", file=sys.stderr, flush=True)
        if np.any(validation_fold < 0) or any(not np.isfinite(scores).all()
                                              for scores in oof.values()):
            raise AssertionError("Each training clip must have one out-of-fold score")
        fold_assignments.append(validation_fold)
        for name, scores in oof.items():
            oof_score_sets[name].append(scores.copy())
        partitions.append({"partition": partition, "results": {
            name: metrics(train_labels, scores) for name, scores in oof.items()
        }})

    args.oof_scores.parent.mkdir(parents=True, exist_ok=True)
    with args.oof_scores.open("wb") as stream:
        np.savez_compressed(stream, labels=train_labels, groups=groups,
                            fold_ids=np.vstack(fold_assignments),
                            **{name: np.vstack(scores)
                               for name, scores in oof_score_sets.items()})

    mean_costs = {name: float(np.mean([
        part["results"][name]["normalized_min_dcf"] for part in partitions
    ])) for name in matrices}
    selected = min(mean_costs, key=mean_costs.get)
    # Median OOF cutoff across training partitions dampens fold-calibration shifts.
    training_cutoffs = {name: float(np.median([
        part["results"][name]["min_dcf_cutoff"] for part in partitions
    ])) for name in matrices}
    selected_cutoff = training_cutoffs[selected]
    test_metadata = [analyze_metadata(path) for path in test_files]
    test_results = {}
    for name, matrix in matrices.items():
        model = fit_model(name, matrix, train_labels, 42)
        test_matrix = baseline[train_count:] if name == "optimized_84" else combined[train_count:]
        raw = model.predict_proba(test_matrix)[:, 1]
        scores = np.asarray([adjust_score(score, info)
                             for score, info in zip(raw, test_metadata)])
        test_results[name] = {**metrics(test_labels, scores),
                              "audio_only": metrics(test_labels, raw),
                              "cost_at_training_cutoff": dcf_at_cutoff(
                                  test_labels, scores,
                                  training_cutoffs[name])}
    report = {
        "objective": "(4 * 0.7 * real_false_positive_rate + 1 * 0.3 * spoof_false_negative_rate) / 0.3",
        "training_samples": train_count,
        "test_samples": len(test_files),
        "train_manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "test_manifest_sha256": hashlib.sha256(args.test_manifest.read_bytes()).hexdigest(),
        "baseline_cache_verification": {
            "method": ("existing cache, five decoded rows checked" if checked_baseline_rows
                       is not None else "current-code fingerprint"),
            "checked_row_indices": checked_baseline_rows,
        },
        "validation": "Three generator-held-out partitions with disjoint real groups; selection uses training only",
        "oof_scores": str(args.oof_scores),
        "partitions": partitions,
        "mean_training_min_dcf": mean_costs,
        "training_cutoffs": training_cutoffs,
        "selected_model": selected,
        "selected_training_cutoff": selected_cutoff,
        "held_out_test": {"status": "exploratory; this split was examined in earlier work",
                          "results": test_results},
        "caveat": "LJ chapters share a narrator; synthetic groups come from DiffSSD; no hidden challenge labels used.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_training_min_dcf": mean_costs,
                      "selected_model": selected,
                      "held_out_test": test_results}, indent=2))

    # Save the selected LFCC candidate trained on every available labeled clip.
    # The report above keeps the test check tied to train-only fits.
    if selected != "optimized_84":
        full_labels = np.r_[train_labels, test_labels]
        model = fit_model(selected, combined, full_labels, 42)
        bundle = {"model": model, "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                  "version": 1, "kind": "experimental-baseline-plus-lfcc",
                  "architecture": selected, "lowpass_hz": None,
                  "metadata_real_weight": METADATA_REAL_WEIGHT}
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model)
        print(f"Saved selected full-data LFCC model to {args.model}")


if __name__ == "__main__":
    main()
