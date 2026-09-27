"""Evaluate global and condition-routed fusion from aligned OOF score files.

Input CSV/TSV columns: filename,label,group,arshiya_score,julia_score. Scores
must be out-of-fold scores; they must not be made by models trained on the same
clips. This script never reads challenge answer keys or test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.forensics import extract_quality_features
from hearsay.routing import fit_router, fit_router_from_clusters, min_dcf


def read_rows(path: Path, primary_column: str, secondary_column: str):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = set(reader.fieldnames or ())
        rows = list(reader)
    required = {"filename", "label", "group", primary_column, secondary_column}
    if not rows or not required.issubset(columns):
        raise ValueError(f"Score file must contain: {', '.join(sorted(required))}")
    labels = []
    files = []
    seen_files = set()
    for line_number, row in enumerate(rows, start=2):
        filename = (row["filename"] or "").strip()
        group = (row["group"] or "").strip()
        tag = (row["label"] or "").strip().lower()
        if not filename or not group:
            raise ValueError(f"Missing filename or group on line {line_number}")
        row["filename"] = filename
        row["group"] = group
        if tag in {"0", "real", "bonafide", "bona fide"}:
            labels.append(0)
        elif tag in {"1", "synthetic", "spoof", "fake"}:
            labels.append(1)
        else:
            raise ValueError(f"Unrecognized label on line {line_number}: {tag}")
        for column in (primary_column, secondary_column):
            try:
                score = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {column} on line {line_number}") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{column} must be finite and within [0, 1] on line {line_number}")
        file = (path.parent / filename).resolve()
        if file in seen_files:
            raise ValueError(f"Duplicate filename on line {line_number}: {filename}")
        files.append(file)
        seen_files.add(file)
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("A filename in score file does not exist relative to that file")
    return rows, files, np.asarray(labels, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/router.joblib"))
    parser.add_argument("--report", type=Path, default=Path("models/routing_validation.json"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--minimum-cluster-size", type=int, default=40)
    parser.add_argument("--clusterer", choices=("kmeans", "qiskit"), default="kmeans")
    parser.add_argument("--quantum-max-samples", type=int, default=120)
    parser.add_argument("--primary-column", default="arshiya_score")
    parser.add_argument("--secondary-column", default="julia_score",
                        help="Use arshiya_spectral_score for the in-branch plumbing comparison")
    args = parser.parse_args()

    rows, files, labels = read_rows(args.scores, args.primary_column, args.secondary_column)
    groups = np.asarray([row["group"] for row in rows])
    arshiya = np.asarray([float(row[args.primary_column]) for row in rows])
    julia = np.asarray([float(row[args.secondary_column]) for row in rows])
    quality = np.vstack([extract_quality_features(p) for p in files])

    def fit_condition_router(index: np.ndarray):
        if args.clusterer == "kmeans":
            return fit_router(quality[index], labels[index], arshiya[index], julia[index],
                              args.clusters, args.minimum_cluster_size)
        try:
            from qiskit.circuit.library import zz_feature_map
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from sklearn.cluster import SpectralClustering
        except ImportError as exc:
            raise RuntimeError("Install requirements-quantum.txt to use --clusterer qiskit") from exc
        # Bound the O(n^2) kernel matrix and use deterministic selection.
        chosen = np.linspace(0, len(index) - 1,
                             min(args.quantum_max_samples, len(index)), dtype=int)
        sampled = index[chosen]
        scaler = StandardScaler().fit(quality[sampled])
        scaled = scaler.transform(quality[sampled])
        angles = np.clip(scaled[:, :4], -2.0, 2.0) * (np.pi / 2)
        feature_map = zz_feature_map(feature_dimension=4, reps=1)
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        gram = kernel.evaluate(x_vec=angles)
        count = min(args.clusters, max(1, len(sampled) // args.minimum_cluster_size))
        assignments = SpectralClustering(n_clusters=count, affinity="precomputed",
                                         assign_labels="kmeans", random_state=42).fit_predict(gram)
        centers = np.vstack([scaled[assignments == c].mean(axis=0) for c in range(count)])
        return fit_router_from_clusters(
            quality[sampled], labels[sampled], arshiya[sampled], julia[sampled],
            assignments, centers, scaler, args.minimum_cluster_size,
            method="Qiskit fidelity quantum kernel + spectral clustering",
        )

    # Second-level group holdout measures the route-selection procedure itself.
    # The component scores are expected to already be out-of-fold predictions.
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=73)
    fold_results = []
    for fold, (train, test) in enumerate(splitter.split(quality, labels, groups), start=1):
        if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
            continue
        router = fit_condition_router(train)
        blend = router.global_alpha * arshiya[test] + (1 - router.global_alpha) * julia[test]
        routed = router.predict(arshiya[test], julia[test], quality[test])
        fold_results.append({"fold": fold, "test_groups": sorted(set(groups[test])),
                             "global_blend_min_dcf": min_dcf(labels[test], blend)[0],
                             "conditional_min_dcf": min_dcf(labels[test], routed)[0],
                             "global_alpha": router.global_alpha})

    final_router = fit_condition_router(np.arange(len(labels)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"router": final_router, "version": 1}, args.output)
    report = {
        "metric": "normalized minDCF; Pspoof=0.3, Cmiss=1, Cfa=4",
        "score_direction": "larger means synthetic",
        "clusterer": args.clusterer,
        "primary_score_column": args.primary_column,
        "secondary_score_column": args.secondary_column,
        "score_source": "out-of-fold component scores; inspect provenance before use",
        "folds": fold_results,
        "mean_global_blend_min_dcf": float(np.mean([r["global_blend_min_dcf"] for r in fold_results])) if fold_results else None,
        "mean_conditional_min_dcf": float(np.mean([r["conditional_min_dcf"] for r in fold_results])) if fold_results else None,
        "final_router_training_summary": final_router.metric,
        "artifact": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
