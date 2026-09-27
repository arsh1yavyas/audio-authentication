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


def _dcf_inputs(labels: np.ndarray, scores: np.ndarray, p_spoof: float,
                c_miss: float, c_fa: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the shared score convention and return the DCF normalizer."""
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("minDCF needs equally sized scores and both labels (0=real, 1=spoof)")
    if not np.all(np.isfinite(s)) or not 0 < p_spoof < 1:
        raise ValueError("Scores must be finite and Pspoof must be between zero and one")
    if not np.isfinite(c_miss) or not np.isfinite(c_fa) or c_miss <= 0 or c_fa <= 0:
        raise ValueError("Miss and false-alarm costs must be finite and positive")
    return y, s, min(c_miss * p_spoof, c_fa * (1.0 - p_spoof))


def dcf_at_cutoff(labels: np.ndarray, scores: np.ndarray, cutoff: float,
                  p_spoof: float = 0.3, c_miss: float = 1.0,
                  c_fa: float = 4.0) -> float:
    """Normalized DCF for declaring scores at or above ``cutoff`` spoof.

    This is the deployable cost when the cutoff was chosen without looking at
    these labels. A miss accepts spoof as real; a false alarm rejects real.
    Positive or negative infinite cutoffs represent the two constant decisions.
    """
    y, s, normalizer = _dcf_inputs(labels, scores, p_spoof, c_miss, c_fa)
    if np.asarray(cutoff).ndim != 0:
        raise ValueError("Cutoff must be a scalar number other than NaN")
    try:
        threshold = float(cutoff)
    except (TypeError, ValueError) as error:
        raise ValueError("Cutoff must be a scalar number other than NaN") from error
    if np.isnan(threshold):
        raise ValueError("Cutoff must be a scalar number other than NaN")
    miss_rate = np.mean(s[y == 1] < threshold)
    false_alarm_rate = np.mean(s[y == 0] >= threshold)
    return float((c_miss * p_spoof * miss_rate +
                  c_fa * (1.0 - p_spoof) * false_alarm_rate) / normalizer)


def min_dcf(labels: np.ndarray, scores: np.ndarray, p_spoof: float = 0.3,
            c_miss: float = 1.0, c_fa: float = 4.0) -> tuple[float, float]:
    """Minimize normalized DCF for scores increasing as spoof.

    A miss accepts spoof as real; a false alarm flags real as spoof. For the
    default 30% spoof prior and 4:1 false-alarm cost, the numerator is
    ``0.3 * FNR + 2.8 * FPR`` and the normalizer is ``0.3``. The result is
    an optimistic ranking summary; select a deployment cutoff on separate data.
    """
    y, s, normalizer = _dcf_inputs(labels, scores, p_spoof, c_miss, c_fa)

    # For score >= threshold, every tied score must receive the same decision.
    # Evaluate each distinct score and one cutoff above the maximum (all real).
    order = np.argsort(s, kind="stable")
    sorted_scores = s[order]
    sorted_labels = y[order]
    distinct_scores, first_indices = np.unique(sorted_scores, return_index=True)
    thresholds = np.r_[distinct_scores, np.nextafter(distinct_scores[-1], np.inf)]
    positions = np.r_[first_indices, len(s)]
    real_below = np.r_[0, np.cumsum(sorted_labels == 0)][positions]
    spoof_below = np.r_[0, np.cumsum(sorted_labels == 1)][positions]
    n_real = int(np.sum(y == 0))
    n_spoof = len(y) - n_real
    p_miss = spoof_below / n_spoof
    p_fa = (n_real - real_below) / n_real
    costs = (c_miss * p_spoof * p_miss +
             c_fa * (1.0 - p_spoof) * p_fa)
    index = int(np.argmin(costs))
    return float(costs[index] / normalizer), float(thresholds[index])


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
