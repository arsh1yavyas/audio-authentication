"""Benchmark Hearsay models on an official, unseen-attack ASVspoof 5 split.

The ASVspoof development manifest is scored only after model selection on
group-held-out training data. It is never included in a fit or router policy.
Balanced, group-aware sampling preserves all speakers and attack families
while keeping this experiment practical on CPU-only Docker installations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import FEATURE_NAMES, decode_audio, extract_features
from hearsay.forensics import QUALITY_FEATURE_NAMES, extract_quality_features
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.metadata import analyze_metadata
from hearsay.routing import fit_router, min_dcf
from hearsay.temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features
from hearsay.utils import read_manifest


def _balanced_group_sample(labels: np.ndarray, groups: list[str], cap: int,
                           seed: int) -> np.ndarray:
    """Sample up to ``cap`` rows per class while representing every group."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        class_rows = np.flatnonzero(labels == label)
        if len(class_rows) <= cap:
            selected.extend(class_rows.tolist())
            continue
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in class_rows:
            grouped[groups[int(index)]].append(int(index))
        names = sorted(grouped)
        quota, remainder = divmod(cap, len(names))
        picked: list[int] = []
        leftovers: list[int] = []
        for position, name in enumerate(names):
            rows = np.asarray(grouped[name], dtype=int)
            rng.shuffle(rows)
            take = min(len(rows), quota + int(position < remainder))
            picked.extend(rows[:take].tolist())
            leftovers.extend(rows[take:].tolist())
        if len(picked) < cap:
            rng.shuffle(leftovers)
            picked.extend(leftovers[:cap - len(picked)])
        selected.extend(picked)
    return np.asarray(sorted(selected), dtype=int)


def _fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for name in ("features.py", "lfcc.py", "temporal.py", "forensics.py"):
        digest.update((root / "hearsay" / name).read_bytes())
    for path in files:
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _extract_one(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio = decode_audio(path)
    return (extract_features(path, audio), extract_lfcc_features(path, audio),
            extract_temporal_features(path, audio), extract_quality_features(path, audio))


def feature_views(files: list[Path], cache: Path, workers: int) -> dict[str, np.ndarray]:
    """Cache four aligned feature views and resume interrupted extraction."""
    fingerprint = _fingerprint(files)
    names = {
        "base": FEATURE_NAMES,
        "lfcc": LFCC_FEATURE_NAMES,
        "temporal": TEMPORAL_FEATURE_NAMES,
        "quality": QUALITY_FEATURE_NAMES,
    }
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                arrays = {key: saved[key].copy() for key in names}
                if all(arrays[key].shape == (len(files), len(names[key])) and
                       np.isfinite(arrays[key]).all() for key in names):
                    print(f"Reusing feature cache {cache}", file=sys.stderr, flush=True)
                    return arrays

    partial = cache.with_suffix(".partial.npz")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in names}
    if partial.is_file():
        with np.load(partial, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                earlier = {key: saved[key] for key in names}
                count = len(earlier["base"])
                if count <= len(files) and all(
                        earlier[key].shape == (count, len(names[key])) and
                        np.isfinite(earlier[key]).all() for key in names):
                    arrays = {key: list(earlier[key]) for key in names}
                    print(f"Resuming feature extraction at {count}/{len(files)}",
                          file=sys.stderr, flush=True)
    done = len(arrays["base"])

    def save_partial() -> None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = partial.with_suffix(".tmp.npz")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, fingerprint=fingerprint,
                **{key: np.vstack(value) for key, value in arrays.items()},
            )
        temporary.replace(partial)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset, values in enumerate(pool.map(_extract_one, files[done:]), start=done + 1):
            for key, vector in zip(names, values):
                arrays[key].append(vector)
            if offset % 100 == 0 or offset == len(files):
                save_partial()
                print(f"Extracted {offset}/{len(files)} clips", file=sys.stderr, flush=True)
    result = {key: np.vstack(value).astype(np.float32) for key, value in arrays.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as stream:
        np.savez_compressed(stream, fingerprint=fingerprint, **result)
    partial.unlink(missing_ok=True)
    return result


def _fold_ids(labels: np.ndarray, groups: list[str], folds: int = 5) -> np.ndarray:
    """Hold complete real-speaker and spoof-generator groups out by fold."""
    ids = np.full(len(labels), -1, dtype=int)
    for label in (0, 1):
        rows = np.flatnonzero(labels == label)
        counts = Counter(groups[int(index)] for index in rows)
        order = sorted(counts, key=lambda group: (-counts[group], group))
        bins = np.zeros(folds, dtype=int)
        assignment: dict[str, int] = {}
        for group in order:
            fold = int(np.argmin(bins))
            assignment[group] = fold
            bins[fold] += counts[group]
        ids[rows] = [assignment[groups[int(index)]] for index in rows]
    if np.any(ids < 0):
        raise AssertionError("Fold assignment omitted rows")
    return ids


def _models(labels: np.ndarray, seed: int = 42) -> dict[str, tuple[str, object]]:
    """Build fresh models; cost weights encode the challenge's false-alarm cost."""
    n_real = int(np.sum(labels == 0))
    n_spoof = int(np.sum(labels == 1))
    if not n_real or not n_spoof:
        raise ValueError("Each training fold must contain both labels")
    real_cost_weight = (4.0 * 0.7 / n_real) / (1.0 * 0.3 / n_spoof)
    fa_weights = {0: real_cost_weight, 1: 1.0}
    return {
        "logistic_204": ("base_lfcc", make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                 max_iter=2000, random_state=seed))),
        "rbf_svm_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=1.0, gamma="scale", class_weight="balanced", random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        # A compact regularization sweep tests whether the default C=1.0 is
        # over- or under-regularized on held-out speakers and attack families.
        "rbf_svm_c0_3_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=0.3, gamma="scale", class_weight="balanced", random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        "rbf_svm_c3_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=3.0, gamma="scale", class_weight="balanced", random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        "cost_logistic_204": ("base_lfcc", make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight=fa_weights,
                                                 max_iter=2000, random_state=seed))),
        "cost_rbf_svm_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=1.0, gamma="scale", class_weight=fa_weights, random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        "cost_rbf_svm_c0_3_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=0.3, gamma="scale", class_weight=fa_weights, random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        "cost_rbf_svm_c3_204": ("base_lfcc", CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), SVC(
                C=3.0, gamma="scale", class_weight=fa_weights, random_state=seed)),
            method="sigmoid", cv=3, ensemble=False)),
        "logistic_temporal": ("temporal", make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                 max_iter=2000, random_state=seed))),
        "extra_trees_base": ("base", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=1, random_state=seed)),
    }


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    cost, threshold = min_dcf(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    at_min = scores >= threshold
    return {
        "normalized_minDCF": float(cost),
        "minDCF_threshold": float(threshold),
        "FAR_at_minDCF": float(np.mean(at_min[labels == 0])),
        "miss_rate_at_minDCF": float(np.mean(~at_min[labels == 1])),
        "EER": float((fpr[index] + fnr[index]) / 2),
        "AUC": float(roc_auc_score(labels, scores)),
    }


def _grouped_oof(train_views: dict[str, np.ndarray], labels: np.ndarray,
                 groups: list[str], quality: np.ndarray) -> tuple[dict, dict, dict]:
    matrices = {
        "base_lfcc": np.hstack((train_views["base"], train_views["lfcc"])),
        "temporal": train_views["temporal"],
        "base": train_views["base"],
    }
    model_names = tuple(_models(labels))
    oof = {name: np.full(len(labels), np.nan) for name in model_names}
    fold_ids = _fold_ids(labels, groups)
    for fold in range(5):
        valid = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        specifications = _models(labels[fit], seed=42 + fold)
        for name, (view, estimator) in specifications.items():
            fitted = estimator.fit(matrices[view][fit], labels[fit])
            oof[name][valid] = fitted.predict_proba(matrices[view][valid])[:, 1]
        print(f"Completed grouped OOF fold {fold + 1}/5; held out "
              f"{len(set(np.asarray(groups)[valid]))} complete source groups",
              file=sys.stderr, flush=True)
    if any(not np.isfinite(scores).all() for scores in oof.values()):
        raise ValueError("Grouped OOF scoring left missing predictions")
    base_metrics = {name: _metrics(labels, values) for name, values in oof.items()}

    # Cross-fit the conditional policy itself: for each held-out fold, learn
    # blend weights and K-means conditions only from the other OOF folds.
    global_route = np.full(len(labels), np.nan)
    conditional_route = np.full(len(labels), np.nan)
    route_folds = []
    for fold in range(5):
        valid = fold_ids == fold
        fit = ~valid
        ranked = sorted(oof, key=lambda name: _metrics(labels[fit], oof[name][fit])["normalized_minDCF"])
        first, second = ranked[:2]
        router = fit_router(quality[fit], labels[fit], oof[first][fit], oof[second][fit],
                            clusters=3, minimum_cluster_size=40)
        alpha = router.global_alpha
        global_route[valid] = alpha * oof[first][valid] + (1.0 - alpha) * oof[second][valid]
        conditional_route[valid] = router.predict(oof[first][valid], oof[second][valid],
                                                   quality[valid])
        route_folds.append({"fold": fold + 1, "model_pair": [first, second],
                            "global_alpha": float(alpha),
                            "conditional_cluster_weights": router.cluster_alphas})
    if not np.isfinite(global_route).all() or not np.isfinite(conditional_route).all():
        raise ValueError("Cross-fitted route predictions are incomplete")
    routing = {
        "folds": route_folds,
        "global_blend": _metrics(labels, global_route),
        "conditional_kmeans": _metrics(labels, conditional_route),
        "conditional_improves_each_fold": [
            _metrics(labels[fold_ids == fold], conditional_route[fold_ids == fold])["normalized_minDCF"]
            < _metrics(labels[fold_ids == fold], global_route[fold_ids == fold])["normalized_minDCF"]
            for fold in range(5)
        ],
    }
    return base_metrics, routing, {**oof, "global_blend": global_route,
                                  "conditional_kmeans": conditional_route}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hearsay-train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--asv-train", type=Path,
                        default=Path("data/external/asvspoof5/manifests/train.tsv"))
    parser.add_argument("--asv-dev", type=Path,
                        default=Path("data/external/asvspoof5/manifests/dev.tsv"))
    parser.add_argument("--train-cap-per-class", type=int, default=6000)
    parser.add_argument("--dev-cap-per-class", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-cache", type=Path,
                        default=Path("data/external/asvspoof5/cache/train_features.npz"))
    parser.add_argument("--dev-cache", type=Path,
                        default=Path("data/external/asvspoof5/cache/dev_features.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("models/asvspoof5_external_benchmark.json"))
    parser.add_argument("--candidate-model", type=Path,
                        default=Path("models/asvspoof5_lfcc_candidate.joblib"))
    parser.add_argument("--existing-model", type=Path,
                        default=Path("models/arshiya_librispeech_julia_lfcc.joblib"))
    args = parser.parse_args()
    if args.workers < 1 or args.train_cap_per_class < 100 or args.dev_cap_per_class < 100:
        parser.error("workers must be positive and sample caps must be at least 100")

    hearsay_files, hearsay_y, hearsay_groups = read_manifest(args.hearsay_train)
    asv_files, asv_y, asv_groups = read_manifest(args.asv_train)
    dev_files_all, dev_y_all, dev_groups_all = read_manifest(args.asv_dev)
    assert hearsay_groups is not None and asv_groups is not None and dev_groups_all is not None
    asv_keep = _balanced_group_sample(asv_y, asv_groups, args.train_cap_per_class, 501)
    dev_keep = _balanced_group_sample(dev_y_all, dev_groups_all, args.dev_cap_per_class, 502)
    train_files = hearsay_files + [asv_files[int(i)] for i in asv_keep]
    train_y = np.concatenate((hearsay_y, asv_y[asv_keep]))
    train_groups = hearsay_groups + [asv_groups[int(i)] for i in asv_keep]
    dev_files = [dev_files_all[int(i)] for i in dev_keep]
    dev_y = dev_y_all[dev_keep]
    dev_groups = [dev_groups_all[int(i)] for i in dev_keep]
    if set(train_files) & set(dev_files):
        raise ValueError("Train and dev audio overlap")

    train_views = feature_views(train_files, args.train_cache, args.workers)
    dev_views = feature_views(dev_files, args.dev_cache, args.workers)
    oof_metrics, routing, oof_scores = _grouped_oof(
        train_views, train_y, train_groups, train_views["quality"])
    asv_domain = np.asarray(["/external/asvspoof5/" in str(path)
                             for path in train_files], dtype=bool)
    oof_by_corpus = {
        "hearsay_sources": {
            "count": int(np.sum(~asv_domain)),
            "models": {name: _metrics(train_y[~asv_domain], scores[~asv_domain])
                       for name, scores in oof_scores.items()},
        },
        "asvspoof5_training_families": {
            "count": int(np.sum(asv_domain)),
            "models": {name: _metrics(train_y[asv_domain], scores[asv_domain])
                       for name, scores in oof_scores.items()},
        },
    }
    matrices = {
        "base_lfcc": np.hstack((train_views["base"], train_views["lfcc"])),
        "temporal": train_views["temporal"], "base": train_views["base"],
    }
    dev_matrices = {
        "base_lfcc": np.hstack((dev_views["base"], dev_views["lfcc"])),
        "temporal": dev_views["temporal"], "base": dev_views["base"],
    }
    specifications = _models(train_y)
    train_oof_best = min(oof_metrics, key=lambda name: oof_metrics[name]["normalized_minDCF"])
    ranking = sorted(oof_metrics, key=lambda name: oof_metrics[name]["normalized_minDCF"])
    fitted_models = {}
    dev_scores = {}
    for name in ranking:
        view = specifications[name][0]
        fitted = specifications[name][1].fit(matrices[view], train_y)
        fitted_models[name] = fitted
        dev_scores[name] = fitted.predict_proba(dev_matrices[view])[:, 1]
    dev_metrics = {name: _metrics(dev_y, scores) for name, scores in dev_scores.items()}
    existing_bundle = joblib.load(args.existing_model)
    expected_feature_names = (*FEATURE_NAMES, *LFCC_FEATURE_NAMES)
    if tuple(existing_bundle.get("feature_names", ())) != expected_feature_names:
        raise ValueError("Existing model does not match the current 204-feature view")
    existing_model = existing_bundle["model"]
    existing_scores = existing_model.predict_proba(dev_matrices["base_lfcc"])[:, 1]
    dev_metrics["unchanged_current_hearsay_model"] = _metrics(dev_y, existing_scores)

    # The same nested-on-OOF policy comparison selects its top pair using OOF
    # performance only. The dev partition is used only for this final report.
    first, second = ranking[:2]
    final_router = fit_router(train_views["quality"], train_y,
                              oof_scores[first], oof_scores[second],
                              clusters=3, minimum_cluster_size=40)
    global_dev = (final_router.global_alpha * dev_scores[first] +
                  (1.0 - final_router.global_alpha) * dev_scores[second])
    conditional_dev = final_router.predict(dev_scores[first], dev_scores[second],
                                           dev_views["quality"])
    route_dev = {"models": [first, second],
                 "global_blend": _metrics(dev_y, global_dev),
                 "conditional_kmeans": _metrics(dev_y, conditional_dev),
                 "router_training_summary": final_router.metric}

    # Select only among the 204-feature bundles understood by predict-lfcc.
    # The development shard does not enter this choice.
    deployable = ("logistic_204", "rbf_svm_204", "rbf_svm_c0_3_204",
                  "rbf_svm_c3_204", "cost_logistic_204", "cost_rbf_svm_204",
                  "cost_rbf_svm_c0_3_204", "cost_rbf_svm_c3_204")
    selected_bundle_name = min(deployable,
                               key=lambda name: oof_metrics[name]["normalized_minDCF"])
    selected_bundle_model = fitted_models[selected_bundle_name]
    args.candidate_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": selected_bundle_model,
        "feature_names": expected_feature_names,
        "version": 1,
        "kind": "experimental-baseline-plus-lfcc",
        "architecture": selected_bundle_name,
        "metadata_real_weight": 0.0,
        "selection_metric": "five-fold grouped OOF Hearsay + ASVspoof5 T_aa minDCF",
        "training_count": len(train_files),
        "training_groups": len(set(train_groups)),
    }, args.candidate_model)

    threshold_report = {}
    for name in ranking:
        _, cutoff = min_dcf(train_y, oof_scores[name])
        accepted = dev_scores[name] >= cutoff
        threshold_report[name] = {
            "oof_selected_threshold": float(cutoff),
            "dev_FAR_at_oof_threshold": float(np.mean(accepted[dev_y == 0])),
            "dev_miss_at_oof_threshold": float(np.mean(~accepted[dev_y == 1])),
        }
    attack_metrics = {}
    for group in sorted(set(dev_groups[i] for i, label in enumerate(dev_y) if label == 1)):
        mask = np.asarray([(dev_groups[i] == group and dev_y[i] == 1)
                           for i in range(len(dev_y))])
        attack_metrics[group] = {
            "spoof_examples": int(mask.sum()),
            "recall_at_oof_selected_threshold": {
                name: float(np.mean(dev_scores[name][mask] >= min_dcf(train_y, oof_scores[name])[1]))
                for name in ranking
            },
        }

    report = {
        "objective": "Hearsay normalized minDCF (Pspoof=0.3, Cmiss=1, Cfa=4); lower is better",
        "score_direction": "higher means synthetic/spoof",
        "train": {
            "hearsay_count": len(hearsay_files),
            "official_asvspoof5_count": len(asv_keep),
            "selected_asvspoof5_counts": dict(Counter("spoof" if asv_y[i] else "real" for i in asv_keep)),
            "total_count": len(train_files),
            "real": int(np.sum(train_y == 0)), "synthetic": int(np.sum(train_y == 1)),
            "complete_source_groups": len(set(train_groups)),
            "asvspoof_attack_families": sorted({group.rsplit(":", 1)[-1]
                                                 for group in train_groups if ":spoof:" in group}),
            "grouped_oof_models": oof_metrics,
            "grouped_oof_by_corpus": oof_by_corpus,
            "cross_fitted_routing": routing,
            "model_selection": f"lowest grouped-OOF minDCF: {train_oof_best}",
            "deployable_204_feature_model": selected_bundle_name,
        },
        "unseen_attack_development": {
            "status": "never fit; never used to select models, weights, or thresholds",
            "official_dev_count": len(dev_files),
            "real": int(np.sum(dev_y == 0)), "synthetic": int(np.sum(dev_y == 1)),
            "complete_source_groups": len(set(dev_groups)),
            "unseen_attack_families": sorted({group.rsplit(":", 1)[-1]
                                               for group in dev_groups if ":spoof:" in group}),
            "model_metrics": dev_metrics,
            "rates_at_oof_thresholds": threshold_report,
            "per_unseen_attack_recall": attack_metrics,
            "routing": route_dev,
            "candidate_model_artifact": str(args.candidate_model),
        },
        "limitations": [
            "ASVspoof development metrics are an external generalization benchmark, not the Hearsay organizer's hidden test result.",
            "MinDCF is threshold-swept; a monotonic remapping such as multiplying every score by a constant cannot change minDCF or EER.",
            "Development-set minDCF is reported descriptively; never tune a submission on this partition.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
