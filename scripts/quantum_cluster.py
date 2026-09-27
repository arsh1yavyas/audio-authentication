"""Optional Qiskit quantum-kernel clustering experiment for quality routing.

This is an offline experiment, not a required Docker inference dependency. It
clusters a small deterministic sample of quality vectors, then writes classical
cluster centroids so the final container can assign routes without Qiskit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.forensics import extract_quality_features
from hearsay.routing import fit_router_from_clusters
from scripts.evaluate_routing import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True,
                        help="OOF score table used to find training audio filenames")
    parser.add_argument("--output", type=Path, default=Path("models/quantum_router.joblib"))
    parser.add_argument("--report", type=Path, default=Path("models/quantum_cluster_report.json"))
    parser.add_argument("--max-samples", type=int, default=120,
                        help="Quantum kernel uses O(n^2) pairwise evaluations")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--secondary-column", default="julia_score",
                        help="Can be arshiya_spectral_score for a routing plumbing check")
    args = parser.parse_args()
    rows, files, labels = read_rows(args.scores, "arshiya_score", args.secondary_column)
    sample_count = min(args.max_samples, len(files))
    if sample_count < 2 or not 2 <= args.clusters <= sample_count:
        raise ValueError("Use at least two samples and between two and the sample count clusters")
    try:
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
    except ImportError as exc:
        raise SystemExit("Install optional dependencies from requirements-quantum.txt") from exc

    # Deterministic subsampling bounds simulator cost. The labels are not used
    # in quantum feature-map fitting or unsupervised cluster assignments.
    chosen = np.linspace(0, len(files) - 1, sample_count, dtype=int)
    quality = np.vstack([extract_quality_features(files[i]) for i in chosen])
    scaler = StandardScaler().fit(quality)
    scaled = scaler.transform(quality)
    # Four angle inputs keep the quantum circuit and pairwise kernel tractable.
    angles = np.clip(scaled[:, :4], -2.0, 2.0) * (np.pi / 2)
    feature_map = zz_feature_map(feature_dimension=4, reps=1)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    gram = kernel.evaluate(x_vec=angles)
    q_labels = SpectralClustering(n_clusters=args.clusters, affinity="precomputed",
                                  assign_labels="kmeans", random_state=42).fit_predict(gram)
    if len(np.unique(q_labels)) != args.clusters:
        raise RuntimeError("Quantum clustering produced an empty cluster")

    # Store centroids of Q-kernel clusters in ordinary quality-feature space.
    # This approximates the discovered partition at inference, avoiding Qiskit
    # installation and kernel evaluation for every submission audio file.
    centroids = np.vstack([scaled[q_labels == c].mean(axis=0) for c in range(args.clusters)])
    classical = KMeans(n_clusters=args.clusters, n_init=20, random_state=42).fit(scaled)
    arshiya = np.asarray([float(row["arshiya_score"]) for row in rows])[chosen]
    julia = np.asarray([float(row[args.secondary_column]) for row in rows])[chosen]
    routed = fit_router_from_clusters(quality, labels[chosen], arshiya, julia,
                                      q_labels, centroids, scaler,
                                      minimum_cluster_size=10,
                                      method="Qiskit FidelityQuantumKernel + SpectralClustering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": 1, "router": routed}, args.output)
    report = {"samples": len(chosen), "features": 4, "clusters": args.clusters,
              "quantum_vs_classical_ari": float(adjusted_rand_score(q_labels, classical.labels_)),
              "quantum_vs_classical_nmi": float(normalized_mutual_info_score(q_labels, classical.labels_)),
              "global_blend_alpha": routed.global_alpha,
              "global_blend_min_dcf_in_sample": routed.metric["global_blend_min_dcf"],
              "quantum_conditional_min_dcf_in_sample": routed.metric["conditional_min_dcf_in_sample"],
              "artifact": str(args.output),
              "caveat": "Unsupervised cluster agreement is not spoof-detection accuracy; compare routed minDCF on grouped OOF scores."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
