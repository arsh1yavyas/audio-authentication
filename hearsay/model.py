"""Training, evaluation, and faithful random-forest path explanations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import FEATURE_NAMES, extract_features

LABELS = {"0": 0, "real": 0, "bonafide": 0, "bona fide": 0,
          "1": 1, "synthetic": 1, "spoof": 1, "fake": 1}
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


def read_manifest(path: Path) -> tuple[list[Path], np.ndarray, list[str] | None]:
    """Read filename,label[,group] with paths relative to the manifest."""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        if not reader.fieldnames or not {"filename", "label"}.issubset(reader.fieldnames):
            raise ValueError("Manifest needs filename and label columns")
        has_group = "group" in reader.fieldnames
        files, labels, groups = [], [], []
        for line, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not filename or label not in LABELS:
                raise ValueError(f"Invalid filename or label on manifest line {line}")
            audio_path = (path.parent / filename).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio on manifest line {line}: {audio_path}")
            files.append(audio_path)
            labels.append(LABELS[label])
            if has_group:
                group = (row.get("group") or "").strip()
                if not group:
                    raise ValueError(f"Missing group on manifest line {line}")
                groups.append(group)
    if len(files) != len(set(files)):
        raise ValueError("Manifest contains duplicate audio paths")
    if len(set(labels)) != 2:
        raise ValueError("Manifest must contain both real and synthetic examples")
    return files, np.asarray(labels, dtype=np.int8), groups if has_group else None


def _validation_indices(labels: np.ndarray, groups: list[str] | None, seed: int):
    if len(labels) < 10 or np.bincount(labels, minlength=2).min() < 3:
        return None
    indices = np.arange(len(labels))
    if groups is None:
        n_test = max(2, math.ceil(len(labels) * 0.2))
        return train_test_split(indices, test_size=n_test, stratify=labels, random_state=seed)
    if len(set(groups)) < 4:
        return None
    for attempt in range(30):
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=seed + attempt).split(indices, labels, groups))
        if len(set(labels[train])) == 2 and len(set(labels[test])) == 2:
            return train, test
    return None


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


def train(manifest: Path, output: Path, seed: int = 42, cache: Path | None = None) -> dict:
    files, labels, groups = read_manifest(manifest)
    features = feature_matrix(files, cache)
    report: dict = {
        "samples": len(files), "real": int(sum(labels == 0)),
        "synthetic": int(sum(labels == 1)), "seed": seed,
        "feature_count": len(FEATURE_NAMES),
        "validation": None,
    }
    split = _validation_indices(labels, groups, seed)
    if split is not None:
        train_idx, test_idx = split
        candidate = _fit_ensemble(features[train_idx], labels[train_idx], seed)
        probabilities = candidate.predict_proba(features[test_idx])[:, 1]
        report["validation"] = {
            "method": "group holdout" if groups is not None else "stratified clip holdout",
            "test_samples": len(test_idx),
            "test_real": int(sum(labels[test_idx] == 0)),
            "test_synthetic": int(sum(labels[test_idx] == 1)),
            "roc_auc": round(float(roc_auc_score(labels[test_idx], probabilities)), 4),
            "average_precision": round(float(average_precision_score(labels[test_idx], probabilities)), 4),
            "accuracy_at_0.5": round(float(accuracy_score(labels[test_idx], probabilities >= 0.5)), 4),
            "balanced_accuracy_at_0.5": round(float(balanced_accuracy_score(labels[test_idx], probabilities >= 0.5)), 4),
        }
        if groups is not None:
            report["validation"]["test_groups"] = sorted({groups[i] for i in test_idx})
    else:
        report["validation_note"] = "Too few examples or groups for a valid two-class holdout; no accuracy estimate."
    report["caveat"] = (
        "A clip holdout can overestimate performance when speakers, generators, or recording "
        "conditions overlap. Supply a group column to keep related clips together."
    )
    final = _fit_ensemble(features, labels, seed)
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


def explain(model: ForestEnsemble, vector: np.ndarray) -> dict:
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
    score = float(np.clip(base + contributions.sum(), 0.0, 1.0))
    rank = np.argsort(-np.abs(contributions))[:8]
    return {
        "cm-score": score,
        "baseline_probability": float(base),
        "total_contribution": float(contributions.sum()),
        "feature_evidence": [
            {"feature": FEATURE_NAMES[i], "value": float(vector[i]),
             "contribution": float(contributions[i]),
             "direction": "synthetic" if contributions[i] > 0 else "real"}
            for i in rank if abs(contributions[i]) > 1e-8
        ],
        "interpretation": "Contributions are shifts in this forest's probability, not proof of origin.",
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
