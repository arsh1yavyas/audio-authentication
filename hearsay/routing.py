"""Score fusion, minDCF evaluation, and condition-aware model routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .forensics import QUALITY_FEATURE_NAMES


@dataclass
class ConditionalRouter:
    """Small K-means router whose cluster weights are learned on OOF scores."""

    scaler: StandardScaler
    clusterer: object
    global_alpha: float
    cluster_alphas: dict[int, float]
    minimum_cluster_size: int = 40
    metric: dict | None = None

    def predict_alpha(self, quality: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(quality)
        clusters = self.clusterer.predict(self.scaler.transform(matrix))
        return np.asarray([self.cluster_alphas.get(int(c), self.global_alpha) for c in clusters])

    def predict(self, arshiya: np.ndarray, julia: np.ndarray,
                quality: np.ndarray) -> np.ndarray:
        """Blend component scores; alpha weights Arshiya, 1-alpha weights Julia."""
        a = np.asarray(arshiya, dtype=float).reshape(-1)
        j = np.asarray(julia, dtype=float).reshape(-1)
        if len(a) != len(j):
            raise ValueError("Arshiya and Julia score counts differ")
        alpha = self.predict_alpha(quality)
        if len(alpha) != len(a):
            raise ValueError("Quality feature count differs from score count")
        return np.clip(alpha * a + (1.0 - alpha) * j, 0.0, 1.0)


@dataclass
class CentroidClusterer:
    """Nearest-centroid inference for partitions discovered by quantum kernels."""

    cluster_centers_: np.ndarray

    def predict(self, scaled_quality: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(scaled_quality)
        distances = ((matrix[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)


def min_dcf(labels: np.ndarray, scores: np.ndarray, p_spoof: float = 0.3,
            c_miss: float = 1.0, c_fa: float = 4.0) -> tuple[float, float]:
    """Official ASVspoof-style normalized minDCF for scores increasing as spoof."""
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("minDCF needs equally sized scores and both labels (0=real, 1=spoof)")
    if not np.all(np.isfinite(s)) or not 0 < p_spoof < 1:
        raise ValueError("Scores must be finite and Pspoof must be between zero and one")
    thresholds = np.r_[-np.inf, np.unique(s), np.inf]
    real = y == 0
    spoof = y == 1
    costs = []
    for threshold in thresholds:
        predicted_spoof = s >= threshold
        p_miss = float(np.mean(~predicted_spoof[spoof]))
        p_fa = float(np.mean(predicted_spoof[real]))
        raw = c_miss * p_miss * (1 - p_spoof) + c_fa * p_fa * p_spoof
        normalizer = min(c_miss * (1 - p_spoof), c_fa * p_spoof)
        costs.append(raw / normalizer)
    index = int(np.argmin(costs))
    return float(costs[index]), float(thresholds[index])


def _select_alpha(labels: np.ndarray, arshiya: np.ndarray, julia: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.0, 1.0, 21)
    ranked = [(min_dcf(labels, alpha * arshiya + (1 - alpha) * julia)[0], float(alpha))
              for alpha in candidates]
    cost, alpha = min(ranked)
    return alpha, cost


def fit_router(quality: np.ndarray, labels: np.ndarray, arshiya: np.ndarray,
               julia: np.ndarray, clusters: int = 3,
               minimum_cluster_size: int = 40) -> ConditionalRouter:
    """Fit unsupervised quality clusters and shrink small groups to global blend."""
    q = np.asarray(quality, dtype=float)
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(arshiya, dtype=float)
    j = np.asarray(julia, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(QUALITY_FEATURE_NAMES):
        raise ValueError("Unexpected quality feature shape")
    if not (len(q) == len(y) == len(a) == len(j)):
        raise ValueError("Quality, labels, and component scores must align")
    scaler = StandardScaler().fit(q)
    n_clusters = min(max(1, int(clusters)), max(1, len(q) // max(1, minimum_cluster_size)))
    clusterer = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(scaler.transform(q))
    assignment = clusterer.labels_
    global_alpha, global_cost = _select_alpha(y, a, j)
    per_cluster: dict[int, float] = {}
    cluster_sizes = {}
    for cluster in range(n_clusters):
        mask = assignment == cluster
        cluster_sizes[str(cluster)] = int(mask.sum())
        # Small/one-class clusters cannot support a useful independent policy.
        if mask.sum() < minimum_cluster_size or len(np.unique(y[mask])) < 2:
            per_cluster[cluster] = global_alpha
        else:
            per_cluster[cluster] = _select_alpha(y[mask], a[mask], j[mask])[0]
    conditional = np.asarray([per_cluster[int(c)] * a[i] + (1 - per_cluster[int(c)]) * j[i]
                              for i, c in enumerate(assignment)])
    routed_cost = min_dcf(y, conditional)[0]
    return ConditionalRouter(
        scaler=scaler, clusterer=clusterer, global_alpha=global_alpha,
        cluster_alphas=per_cluster, minimum_cluster_size=minimum_cluster_size,
        metric={"p_spoof": 0.3, "c_miss": 1.0, "c_fa": 4.0,
                "global_blend_alpha": global_alpha, "global_blend_min_dcf": global_cost,
                "conditional_min_dcf_in_sample": routed_cost,
                "cluster_sizes": cluster_sizes,
                "quality_features": list(QUALITY_FEATURE_NAMES),
                "note": "Choose final policy using grouped out-of-fold predictions; this fit metric is descriptive."},
    )


def fit_router_from_clusters(quality: np.ndarray, labels: np.ndarray,
                             arshiya: np.ndarray, julia: np.ndarray,
                             assignments: np.ndarray, centroids: np.ndarray,
                             scaler: StandardScaler,
                             minimum_cluster_size: int = 40,
                             method: str = "external clustering") -> ConditionalRouter:
    """Attach OOF-derived blend weights to a supplied unsupervised partition."""
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(arshiya, dtype=float)
    j = np.asarray(julia, dtype=float)
    groups = np.asarray(assignments, dtype=int)
    global_alpha, global_cost = _select_alpha(y, a, j)
    cluster_alphas: dict[int, float] = {}
    sizes = {}
    for cluster in range(len(centroids)):
        mask = groups == cluster
        sizes[str(cluster)] = int(mask.sum())
        if mask.sum() >= minimum_cluster_size and len(np.unique(y[mask])) == 2:
            cluster_alphas[cluster] = _select_alpha(y[mask], a[mask], j[mask])[0]
        else:
            cluster_alphas[cluster] = global_alpha
    routed = np.asarray([cluster_alphas[int(c)] * a[i] +
                         (1 - cluster_alphas[int(c)]) * j[i]
                         for i, c in enumerate(groups)])
    return ConditionalRouter(
        scaler=scaler, clusterer=CentroidClusterer(np.asarray(centroids)),
        global_alpha=global_alpha, cluster_alphas=cluster_alphas,
        minimum_cluster_size=minimum_cluster_size,
        metric={"p_spoof": 0.3, "c_miss": 1.0, "c_fa": 4.0,
                "global_blend_alpha": global_alpha,
                "global_blend_min_dcf": global_cost,
                "conditional_min_dcf_in_sample": min_dcf(y, routed)[0],
                "cluster_sizes": sizes, "cluster_method": method,
                "note": "In-sample fit summary; compare using grouped out-of-fold predictions."},
    )
