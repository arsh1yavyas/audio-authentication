"""Training, evaluation, and faithful random-forest path explanations."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

from .features import FEATURE_NAMES, extract_features
from .metadata import METADATA_REAL_WEIGHT, MetadataAnalysis, adjust_score, analyze_metadata
from .utils import read_manifest, split_data

SPECTRAL_INDICES = tuple(i for i, name in enumerate(FEATURE_NAMES)
                         if not name.startswith(("mfcc_", "chroma_")))


@dataclass
class ForestEnsemble:
    """Weighted forests trained on different subsets of one feature vector."""

    components: list[tuple[RandomForestClassifier, tuple[int, ...], float]]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(features)
        result = np.zeros((len(matrix), 2), dtype=np.float64)
        for forest, indices, weight in self.components:
            result += weight * forest.predict_proba(matrix[:, indices])
        return result


def _new_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )


def _fit_ensemble(features: np.ndarray, labels: np.ndarray, seed: int) -> ForestEnsemble:
    full = tuple(range(len(FEATURE_NAMES)))
    components = []
    for indices, weight in ((full, 0.5), (SPECTRAL_INDICES, 0.5)):
        forest = _new_forest(seed).fit(features[:, indices], labels)
        components.append((forest, indices, weight))
    return ForestEnsemble(components)


def feature_matrix(files: list[Path], cache: Path | None = None) -> np.ndarray:
    """Extract features once and reuse them when all source files still match."""
    signature = hashlib.sha256(repr(FEATURE_NAMES).encode())
    signature.update(Path(__file__).with_name("features.py").read_bytes())
    for path in files:
        stat = path.stat()
        signature.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    fingerprint = signature.hexdigest()
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                matrix = saved["features"]
                if matrix.shape == (len(files), len(FEATURE_NAMES)) and np.all(np.isfinite(matrix)):
                    print(f"Loaded cached features from {cache}", file=sys.stderr, flush=True)
                    return matrix
    vectors = []
    for index, path in enumerate(files, start=1):
        vectors.append(extract_features(path))
        if index % 50 == 0 or index == len(files):
            print(f"Extracted features from {index}/{len(files)} training clips", file=sys.stderr, flush=True)
    matrix = np.vstack(vectors)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as stream:
            np.savez_compressed(stream, features=matrix, fingerprint=fingerprint)
    return matrix


def train(manifest: Path, output: Path, seed: int = 42, cache: Path | None = None,
          test_manifest: Path | None = None) -> dict:
    files, labels, groups = read_manifest(manifest)
    provided_split = None
    if test_manifest is not None:
        test_files, test_labels, test_groups = read_manifest(test_manifest)
        if set(files) & set(test_files):
            raise ValueError("Training and test manifests contain overlapping audio files")
        if (groups is None) != (test_groups is None):
            raise ValueError("Training and test manifests must both include groups or both omit them")
        if groups is not None and set(groups) & set(test_groups):
            raise ValueError("Training and test manifests contain overlapping groups")
        provided_split = (np.arange(len(files)), np.arange(len(files), len(files) + len(test_files)))
        files += test_files
        labels = np.concatenate((labels, test_labels))
        if groups is not None:
            groups += test_groups
    metadata = [analyze_metadata(path) for path in files]
    features = feature_matrix(files, cache)
    report: dict = {
        "samples": len(files), "real": int(sum(labels == 0)),
        "synthetic": int(sum(labels == 1)), "seed": seed,
        "feature_count": len(FEATURE_NAMES),
        "validation": None,
    }
    split = provided_split if provided_split is not None else split_data(labels, groups, seed=seed)
    if split is not None:
        train_idx, test_idx = split
        final = _fit_ensemble(features[train_idx], labels[train_idx], seed)
        audio_scores = final.predict_proba(features[test_idx])[:, 1]
        probabilities = np.asarray([
            adjust_score(score, metadata[index])
            for score, index in zip(audio_scores, test_idx)
        ])
        report["training_samples"] = len(train_idx)
        report["validation"] = {
            "method": ("provided test manifest" if provided_split is not None else
                       "group holdout" if groups is not None else "stratified clip holdout"),
            "test_samples": len(test_idx),
            "test_real": int(sum(labels[test_idx] == 0)),
            "test_synthetic": int(sum(labels[test_idx] == 1)),
            "roc_auc": round(float(roc_auc_score(labels[test_idx], probabilities)), 4),
            "average_precision": round(float(average_precision_score(labels[test_idx], probabilities)), 4),
            "accuracy_at_0.5": round(float(accuracy_score(labels[test_idx], probabilities >= 0.5)), 4),
            "balanced_accuracy_at_0.5": round(float(balanced_accuracy_score(labels[test_idx], probabilities >= 0.5)), 4),
            "metadata_real_weight": METADATA_REAL_WEIGHT,
            "metadata_consistent": sum(metadata[index].status == "consistent" for index in test_idx),
            "metadata_inconsistent": sum(metadata[index].status == "inconsistent" for index in test_idx),
        }
        if groups is not None:
            report["validation"]["test_groups"] = sorted({groups[i] for i in test_idx})
    else:
        report["validation_note"] = "Too few examples or groups for a valid two-class holdout; no accuracy estimate."
        report["training_samples"] = len(files)
        final = _fit_ensemble(features, labels, seed)
    report["caveat"] = (
        "A clip holdout can overestimate performance when speakers, generators, or recording "
        "conditions overlap. Supply a group column to keep related clips together. "
        "Coherent container metadata is a forgeable, weak signal; its real-side weight is heuristic."
    )
    importance = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for forest, indices, weight in final.components:
        importance[list(indices)] += weight * forest.feature_importances_
    report["model"] = "equal-weight full-feature and spectral-statistics random forests"
    report["feature_importance"] = dict(sorted(
        ((name, round(float(value), 5)) for name, value in zip(FEATURE_NAMES, importance)),
        key=lambda item: item[1], reverse=True,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_names": FEATURE_NAMES,
                 "sample_rate": 16_000, "version": 2}, output)
    return report


def load_model(path: Path) -> ForestEnsemble:
    # Joblib uses pickle. Model files must come from a trusted source.
    bundle = joblib.load(path)
    if bundle.get("version") != 2 or tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Model was trained with an incompatible feature extractor")
    return bundle["model"]


def explain(model: ForestEnsemble, vector: np.ndarray,
            metadata: MetadataAnalysis | None = None) -> dict:
    """Decompose the prediction into changes along each tree's decision path."""
    contributions = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    base = 0.0
    for forest, indices, weight in model.components:
        tree_weight = weight / len(forest.estimators_)
        for estimator in forest.estimators_:
            tree = estimator.tree_

            def probability(node: int) -> float:
                counts = tree.value[node, 0]
                return float(counts[1] / counts.sum())

            node = 0
            base += tree_weight * probability(node)
            while tree.feature[node] >= 0:
                feature = indices[int(tree.feature[node])]
                child = (tree.children_left[node] if vector[feature] <= tree.threshold[node]
                         else tree.children_right[node])
                contributions[feature] += tree_weight * (probability(child) - probability(node))
                node = child
    # The root probability plus every path change equals the average leaf
    # probability. Avoid invoking another parallel forest prediction per clip.
    audio_score = float(np.clip(base + contributions.sum(), 0.0, 1.0))
    metadata = metadata or MetadataAnalysis("unknown")
    score = adjust_score(audio_score, metadata)
    rank = np.argsort(-np.abs(contributions))[:8]
    return {
        "cm-score": score,
        "audio_score": audio_score,
        "metadata_adjustment": score - audio_score,
        "metadata": metadata.as_dict(),
        "baseline_probability": float(base),
        "total_contribution": float(contributions.sum()),
        "feature_evidence": [
            {"feature": FEATURE_NAMES[i], "value": float(vector[i]),
             "contribution": float(contributions[i]),
             "direction": "synthetic" if contributions[i] > 0 else "real"}
            for i in rank if abs(contributions[i]) > 1e-8
        ],
        "interpretation": "Audio contributions sum to audio_score; coherent metadata gives a small real-side adjustment, not proof of origin.",
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
