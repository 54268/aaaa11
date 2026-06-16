from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
    module="sklearn.cluster._kmeans",
)

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class OfscilSubdivisionResult:
    labels: np.ndarray
    centers: np.ndarray
    resolved_k: int
    suspected_known_mask: np.ndarray
    k_search_history: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass
class FeaturePreprocessor:
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float64))
        if self.pca is not None:
            transformed = self.pca.transform(transformed)
        return transformed


def fit_feature_preprocessor(features: np.ndarray, pca_dim: int = 16) -> FeaturePreprocessor:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(features, dtype=np.float64))
    max_dim = min(int(pca_dim), scaled.shape[1], max(len(scaled) - 1, 1))
    pca = None
    if max_dim >= 2 and scaled.shape[1] > max_dim:
        pca = PCA(n_components=max_dim, random_state=0)
        pca.fit(scaled)
    return FeaturePreprocessor(scaler=scaler, pca=pca)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def cosine_distance_matrix(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    features = l2_normalize(features)
    centers = l2_normalize(centers)
    return 1.0 - np.clip(features @ centers.T, -1.0, 1.0)


def _safe_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    max_samples: int = 1500,
) -> float:
    if len(features) <= 2:
        return -1.0
    eval_features = features
    eval_labels = labels
    if max_samples > 0 and len(features) > max_samples:
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(features), size=int(max_samples), replace=False)
        eval_features = features[sample_indices]
        eval_labels = labels[sample_indices]
    unique_labels = np.unique(eval_labels)
    if len(unique_labels) < 2 or len(eval_labels) <= len(unique_labels):
        return -1.0
    return float(silhouette_score(eval_features, eval_labels, metric="cosine"))


def _safe_calinski_harabasz(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(features) <= len(unique):
        return 0.0
    try:
        return float(calinski_harabasz_score(features, labels))
    except Exception:
        return 0.0


def _fit_centers_kmeans(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
) -> np.ndarray:
    centers = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=int(n_init),
        max_iter=400,
        tol=1e-5,
        random_state=seed,
    ).fit(features).cluster_centers_
    return l2_normalize(centers)


def _fit_centers_agglomerative(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    sample_size: int = 8000,
) -> np.ndarray:
    """对样本子集做凝聚式聚类得到初始中心，再传入分配。

    适合接近球面分布、簇形不规则的情况；大数据时使用子采样以控制 O(N^2) 距离矩阵。
    """
    n_samples = len(features)
    if n_samples <= int(sample_size):
        anchor_features = features
    else:
        rng = np.random.default_rng(seed)
        anchor_idx = rng.choice(n_samples, size=int(sample_size), replace=False)
        anchor_features = features[anchor_idx]

    anchor_features = l2_normalize(anchor_features)
    clustering = AgglomerativeClustering(
        n_clusters=int(num_clusters),
        metric="cosine",
        linkage="average",
    )
    sub_labels = clustering.fit_predict(anchor_features)

    centers = np.zeros((int(num_clusters), features.shape[1]), dtype=np.float64)
    for cluster_id in range(int(num_clusters)):
        mask = sub_labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = anchor_features[mask].mean(axis=0)
        else:
            centers[cluster_id] = anchor_features[0]
    return l2_normalize(centers)


def _fit_centers_gmm(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    covariance_type: str = "diag",
) -> np.ndarray:
    """用 GMM 估计每个簇的中心，再投影到单位球上提供给后续分配。"""
    gmm = GaussianMixture(
        n_components=int(num_clusters),
        covariance_type=covariance_type,
        n_init=max(1, int(n_init) // 4),
        max_iter=300,
        reg_covar=1e-3,
        random_state=seed,
    )
    gmm.fit(np.asarray(features, dtype=np.float32))
    return l2_normalize(gmm.means_.astype(np.float64))


def _fit_labels_gmm_direct(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    covariance_type: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """直接用 GMM 完成聚类（不经过原型引导的余弦再分配）。

    返回 (labels, centers)，centers 是原始空间的高斯均值。
    """
    gmm = GaussianMixture(
        n_components=int(num_clusters),
        covariance_type=covariance_type,
        n_init=max(1, int(n_init) // 5),
        max_iter=300,
        reg_covar=1e-3,
        random_state=seed,
    )
    features32 = np.asarray(features, dtype=np.float32)
    labels = gmm.fit_predict(features32).astype(np.int64)
    confidence = gmm.predict_proba(features32).max(axis=1).astype(np.float64)
    diagnostics = {
        "gmm_bic": float(gmm.bic(features32)),
        "gmm_aic": float(gmm.aic(features32)),
        "gmm_lower_bound": float(gmm.lower_bound_),
        "gmm_mean_confidence": float(confidence.mean()),
        "gmm_median_confidence": float(np.median(confidence)),
    }
    return labels, gmm.means_.astype(np.float64), confidence, diagnostics


def _build_initial_centers(
    backend: str,
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    agg_sample_size: int,
) -> np.ndarray:
    backend = str(backend).lower()
    normalized_features = l2_normalize(features)
    if backend in {"kmeans", "spherical_kmeans"}:
        return _fit_centers_kmeans(normalized_features, num_clusters, seed=seed, n_init=n_init)
    if backend in {"agglomerative", "agglomerative_cosine"}:
        return _fit_centers_agglomerative(
            normalized_features,
            num_clusters,
            seed=seed,
            sample_size=agg_sample_size,
        )
    if backend in {"gmm", "gmm_diag"}:
        return _fit_centers_gmm(normalized_features, num_clusters, seed=seed, n_init=n_init, covariance_type="diag")
    if backend in {"gmm_full"}:
        return _fit_centers_gmm(normalized_features, num_clusters, seed=seed, n_init=n_init, covariance_type="full")
    raise ValueError(f"Unsupported clustering backend: {backend}")


_DIRECT_GMM_BACKENDS = {"gmm_full_direct", "gmm_direct"}
_DENSITY_BACKENDS = {"hdbscan", "density_hdbscan"}


def _fit_labels_hdbscan(
    features: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model = HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=None if min_samples is None else max(1, int(min_samples)),
        cluster_selection_epsilon=max(0.0, float(cluster_selection_epsilon)),
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
    )
    labels = model.fit_predict(np.asarray(features, dtype=np.float64)).astype(np.int64)
    cluster_ids = sorted(int(label) for label in np.unique(labels) if int(label) != -1)
    remap = {cluster_id: new_id for new_id, cluster_id in enumerate(cluster_ids)}
    labels = np.asarray([remap.get(int(label), -1) for label in labels], dtype=np.int64)
    centers = np.asarray(
        [features[labels == cluster_id].mean(axis=0) for cluster_id in range(len(cluster_ids))],
        dtype=np.float64,
    )
    probabilities = np.asarray(getattr(model, "probabilities_", np.ones(len(labels))), dtype=np.float64)
    diagnostics = {
        "density_backend": "hdbscan",
        "density_min_cluster_size": int(min_cluster_size),
        "density_min_samples": None if min_samples is None else int(min_samples),
        "density_cluster_selection_epsilon": float(cluster_selection_epsilon),
        "density_noise_ratio": float((labels == -1).mean()) if len(labels) else 0.0,
        "density_mean_membership_probability": float(probabilities[labels != -1].mean())
        if np.any(labels != -1)
        else 0.0,
    }
    return labels, centers, probabilities, diagnostics


def _merge_close_centers(
    centers: np.ndarray,
    similarity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """如果两个中心余弦相似度高于阈值，把它们合并为一个，避免冗余簇。

    返回 (合并后的中心, 旧 ID -> 新 ID 的映射)。
    """
    if similarity_threshold is None or similarity_threshold >= 1.0 or len(centers) <= 1:
        return centers, np.arange(len(centers), dtype=np.int64)

    normalized = l2_normalize(centers)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -np.inf)
    n = len(centers)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    rows, cols = np.where(similarity >= float(similarity_threshold))
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < c:
            union(r, c)

    roots: list[int] = []
    mapping = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        root = find(idx)
        if root not in roots:
            roots.append(root)
        mapping[idx] = roots.index(root)

    merged_centers = np.zeros((len(roots), centers.shape[1]), dtype=np.float64)
    for new_id, old_root in enumerate(roots):
        members = [idx for idx in range(n) if find(idx) == old_root]
        merged_centers[new_id] = centers[members].mean(axis=0)
    return l2_normalize(merged_centers), mapping


def prototype_guided_clustering(
    features: np.ndarray,
    known_anchor_features: np.ndarray | None,
    num_clusters: int,
    seed: int = 42,
    max_iter: int = 30,
    assignment_margin: float = 0.0,
    known_reject_margin: float = 0.0,
    n_init: int = 30,
    backend: str = "kmeans",
    agg_sample_size: int = 8000,
    merge_similarity_threshold: float | None = None,
    direct_confidence_quantile: float = 0.0,
    direct_min_cluster_size: int = 0,
    density_min_cluster_size: int = 20,
    density_min_samples: int | None = None,
    density_cluster_selection_epsilon: float = 0.0,
) -> OfscilSubdivisionResult:
    """通用入口：先用指定后端拟合初始中心，然后做带原型约束的迭代再分配。

    若 backend 属于 gmm_full_direct，则直接用 GMM 的预测作为标签，跳过余弦再分配，
    适合各簇方差差异较大、不在单位球面上的情形。
    """
    features = np.asarray(features, dtype=np.float32)
    n_samples = len(features)
    if n_samples == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return OfscilSubdivisionResult(empty, np.zeros((0, features.shape[1])), 0, empty.astype(bool), [], {})

    k = max(1, min(int(num_clusters), n_samples))

    if str(backend).lower() in _DENSITY_BACKENDS:
        labels, density_means, _, diagnostics = _fit_labels_hdbscan(
            features,
            min_cluster_size=density_min_cluster_size,
            min_samples=density_min_samples,
            cluster_selection_epsilon=density_cluster_selection_epsilon,
        )
        noise_mask = labels == -1
        return OfscilSubdivisionResult(
            labels=labels,
            centers=l2_normalize(density_means) if len(density_means) else density_means,
            resolved_k=int(len(np.unique(labels[labels != -1]))),
            suspected_known_mask=noise_mask,
            k_search_history=[],
            diagnostics=diagnostics,
        )

    if str(backend).lower() in _DIRECT_GMM_BACKENDS:
        labels, gmm_means, confidence, diagnostics = _fit_labels_gmm_direct(features, k, seed=seed, n_init=n_init, covariance_type="full")
        uncertain_mask = np.zeros(n_samples, dtype=bool)
        confidence_quantile = float(direct_confidence_quantile)
        if confidence_quantile > 0.0:
            confidence_quantile = min(max(confidence_quantile, 0.0), 0.95)
            confidence_threshold = float(np.quantile(confidence, confidence_quantile))
            uncertain_mask |= confidence < confidence_threshold
            labels = np.where(uncertain_mask, -1, labels).astype(np.int64)

        min_cluster_size = int(direct_min_cluster_size)
        if min_cluster_size > 0:
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                if int(cluster_mask.sum()) < min_cluster_size:
                    uncertain_mask |= cluster_mask
            labels = np.where(uncertain_mask, -1, labels).astype(np.int64)

        if known_anchor_features is not None and len(known_anchor_features):
            known_distances = cosine_distance_matrix(features, known_anchor_features)
            nearest_known_dist = known_distances.min(axis=1)
            unknown_distances = cosine_distance_matrix(features, gmm_means)
            nearest_unknown_dist = unknown_distances.min(axis=1)
            suspected_known = nearest_known_dist <= (nearest_unknown_dist + float(known_reject_margin))
            uncertain_mask |= suspected_known
            labels = np.where(uncertain_mask, -1, labels).astype(np.int64)
        return OfscilSubdivisionResult(
            labels=labels,
            centers=l2_normalize(gmm_means),
            resolved_k=int(len(np.unique(labels[labels != -1]))),
            suspected_known_mask=uncertain_mask,
            k_search_history=[],
            diagnostics=diagnostics,
        )

    normalized_features = l2_normalize(features)
    centers = _build_initial_centers(
        backend=backend,
        features=features,
        num_clusters=k,
        seed=seed,
        n_init=n_init,
        agg_sample_size=agg_sample_size,
    )

    known_anchors = l2_normalize(known_anchor_features) if known_anchor_features is not None and len(known_anchor_features) else None

    labels = np.full(n_samples, -1, dtype=np.int64)
    suspected_known = np.zeros(n_samples, dtype=bool)
    for _ in range(max(1, int(max_iter))):
        previous = labels.copy()
        unknown_distances = 1.0 - np.clip(normalized_features @ centers.T, -1.0, 1.0)
        order = np.argsort(unknown_distances, axis=1)
        nearest_unknown = order[:, 0]
        nearest_unknown_dist = unknown_distances[np.arange(n_samples), nearest_unknown]

        if k > 1:
            second_unknown_dist = unknown_distances[np.arange(n_samples), order[:, 1]]
            margin_ok = (second_unknown_dist - nearest_unknown_dist) >= float(assignment_margin)
        else:
            margin_ok = np.ones(n_samples, dtype=bool)

        if known_anchors is not None:
            known_distances = cosine_distance_matrix(features, known_anchors)
            nearest_known_dist = known_distances.min(axis=1)
            suspected_known = nearest_known_dist <= (nearest_unknown_dist + float(known_reject_margin))
        else:
            suspected_known = np.zeros(n_samples, dtype=bool)

        labels = np.where((~suspected_known) & margin_ok, nearest_unknown, -1).astype(np.int64)
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = l2_normalize(normalized_features[mask].mean(axis=0, keepdims=True))[0]
        if np.array_equal(previous, labels):
            break

    if merge_similarity_threshold is not None and len(centers) > 1:
        merged_centers, mapping = _merge_close_centers(centers, float(merge_similarity_threshold))
        if len(merged_centers) != len(centers):
            mapped_labels = np.where(labels == -1, -1, mapping[labels.clip(min=0)])
            centers = merged_centers
            labels = mapped_labels.astype(np.int64)

    return OfscilSubdivisionResult(
        labels=labels,
        centers=centers,
        resolved_k=int(len(np.unique(labels[labels != -1]))),
        suspected_known_mask=suspected_known,
        k_search_history=[],
        diagnostics={},
    )


# 为兼容旧调用保留 prototype_guided_kmeans 名称
def prototype_guided_kmeans(
    features: np.ndarray,
    known_anchor_features: np.ndarray | None,
    num_clusters: int,
    seed: int = 42,
    max_iter: int = 30,
    assignment_margin: float = 0.0,
    known_reject_margin: float = 0.0,
    n_init: int = 20,
) -> OfscilSubdivisionResult:
    return prototype_guided_clustering(
        features=features,
        known_anchor_features=known_anchor_features,
        num_clusters=num_clusters,
        seed=seed,
        max_iter=max_iter,
        assignment_margin=assignment_margin,
        known_reject_margin=known_reject_margin,
        n_init=n_init,
        backend="kmeans",
    )


def _scoring_for_k(
    eval_features: np.ndarray,
    labels: np.ndarray,
    second_labels: np.ndarray,
    centers: np.ndarray,
    known_anchor_features: np.ndarray | None,
    seed: int,
    k: int,
    uncertain_penalty: float,
    stability_weight: float,
    db_weight: float,
    ch_weight: float,
    target_k: int | None,
    target_k_strength: float,
) -> dict[str, Any]:
    valid = labels != -1
    common = valid & (second_labels != -1)
    unique_clusters = np.unique(labels[valid])
    if len(unique_clusters) >= 2 and int(valid.sum()) > len(unique_clusters):
        normalized_valid_features = l2_normalize(eval_features[valid])
        silhouette = _safe_silhouette_score(normalized_valid_features, labels[valid], seed=seed + k)
        db = float(davies_bouldin_score(normalized_valid_features, labels[valid]))
        db_score = 1.0 / (1.0 + max(db, 0.0))
        ch = _safe_calinski_harabasz(normalized_valid_features, labels[valid])
        ch_score = float(ch / (ch + 1000.0))
    else:
        silhouette = -1.0
        db_score = 0.0
        ch_score = 0.0
    stability = float(adjusted_rand_score(labels[common], second_labels[common])) if int(common.sum()) else 0.0
    uncertain_rate = float((labels == -1).mean())
    center_known_penalty = 0.0
    if known_anchor_features is not None and len(known_anchor_features) and len(centers):
        center_known = cosine_distance_matrix(centers, known_anchor_features).min(axis=1)
        center_known_penalty = float(np.maximum(0.0, 0.05 - center_known).mean())

    target_bonus = 0.0
    if target_k is not None and target_k_strength > 0:
        target_bonus = -float(target_k_strength) * abs(int(k) - int(target_k))

    score = (
        silhouette
        + float(db_weight) * db_score
        + float(stability_weight) * stability
        + float(ch_weight) * ch_score
        - float(uncertain_penalty) * uncertain_rate
        - 0.20 * center_known_penalty
        + target_bonus
    )
    return {
        "k": int(k),
        "score": float(score),
        "silhouette": float(silhouette),
        "davies_bouldin_score": float(db_score),
        "calinski_harabasz_norm": float(ch_score),
        "stability": float(stability),
        "uncertain_rate": float(uncertain_rate),
        "center_known_penalty": float(center_known_penalty),
        "target_bonus": float(target_bonus),
    }


def estimate_k(
    features: np.ndarray,
    known_anchor_features: np.ndarray | None,
    k_min: int = 2,
    k_max: int = 15,
    seed: int = 42,
    sample_size: int = 3000,
    assignment_margin: float = 0.0,
    known_reject_margin: float = 0.0,
    backend: str = "kmeans",
    target_num_clusters: int | None = None,
    target_k_strength: float = 0.10,
    uncertain_penalty: float = 0.15,
    stability_weight: float = 0.30,
    db_weight: float = 0.25,
    ch_weight: float = 0.30,
) -> tuple[int, list[dict[str, Any]]]:
    features = np.asarray(features, dtype=np.float64)
    if len(features) <= 3:
        return 1, []

    eval_features = features
    if sample_size > 0 and len(features) > sample_size:
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(features), size=int(sample_size), replace=False)
        eval_features = features[sample_indices]

    upper = max(1, min(int(k_max), len(eval_features) - 1))
    lower = max(1, min(int(k_min), upper))
    best_k = lower
    best_score = -np.inf
    history: list[dict[str, Any]] = []

    for k in range(lower, upper + 1):
        first = prototype_guided_clustering(
            eval_features,
            known_anchor_features,
            num_clusters=k,
            seed=seed,
            assignment_margin=assignment_margin,
            known_reject_margin=known_reject_margin,
            n_init=20,
            backend=backend,
            agg_sample_size=min(len(eval_features), 4000),
        )
        second = prototype_guided_clustering(
            eval_features,
            known_anchor_features,
            num_clusters=k,
            seed=seed + 97,
            assignment_margin=assignment_margin,
            known_reject_margin=known_reject_margin,
            n_init=20,
            backend=backend,
            agg_sample_size=min(len(eval_features), 4000),
        )
        scoring = _scoring_for_k(
            eval_features=eval_features,
            labels=first.labels,
            second_labels=second.labels,
            centers=first.centers,
            known_anchor_features=known_anchor_features,
            seed=seed,
            k=k,
            uncertain_penalty=uncertain_penalty,
            stability_weight=stability_weight,
            db_weight=db_weight,
            ch_weight=ch_weight,
            target_k=target_num_clusters,
            target_k_strength=target_k_strength,
        )
        scoring["resolved_clusters"] = int(first.resolved_k)
        scoring["backend"] = str(backend)
        history.append(scoring)
        if scoring["score"] > best_score:
            best_score = float(scoring["score"])
            best_k = int(k)

    return best_k, history


def run_ofscil_subdivision(
    features: np.ndarray,
    known_anchor_features: np.ndarray | None,
    k_min: int = 2,
    k_max: int = 15,
    seed: int = 42,
    auto_sample_size: int = 3000,
    assignment_margin: float = 0.0,
    known_reject_margin: float = 0.0,
    backend: str = "kmeans",
    target_num_clusters: int | None = None,
    target_k_strength: float = 0.10,
    uncertain_penalty: float = 0.15,
    stability_weight: float = 0.30,
    db_weight: float = 0.25,
    ch_weight: float = 0.30,
    n_init: int = 30,
    merge_similarity_threshold: float | None = None,
    agg_sample_size: int = 8000,
    direct_confidence_quantile: float = 0.0,
    direct_min_cluster_size: int = 0,
    density_min_cluster_size: int = 20,
    density_min_samples: int | None = None,
    density_cluster_selection_epsilon: float = 0.0,
) -> OfscilSubdivisionResult:
    if target_num_clusters is not None and int(target_num_clusters) > 0 and k_min == k_max:
        selected_k = int(target_num_clusters)
        history = []
    elif target_num_clusters is not None and int(target_num_clusters) > 0 and target_k_strength >= 1.0:
        selected_k = int(target_num_clusters)
        history = []
    else:
        selected_k, history = estimate_k(
            features,
            known_anchor_features,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
            sample_size=auto_sample_size,
            assignment_margin=assignment_margin,
            known_reject_margin=known_reject_margin,
            backend=backend,
            target_num_clusters=target_num_clusters,
            target_k_strength=target_k_strength,
            uncertain_penalty=uncertain_penalty,
            stability_weight=stability_weight,
            db_weight=db_weight,
            ch_weight=ch_weight,
        )
    result = prototype_guided_clustering(
        features,
        known_anchor_features,
        num_clusters=selected_k,
        seed=seed,
        assignment_margin=assignment_margin,
        known_reject_margin=known_reject_margin,
        n_init=n_init,
        backend=backend,
        agg_sample_size=agg_sample_size,
        merge_similarity_threshold=merge_similarity_threshold,
        direct_confidence_quantile=direct_confidence_quantile,
        direct_min_cluster_size=direct_min_cluster_size,
        density_min_cluster_size=density_min_cluster_size,
        density_min_samples=density_min_samples,
        density_cluster_selection_epsilon=density_cluster_selection_epsilon,
    )
    result.k_search_history.extend(history)
    return result


def encode_string_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    as_str = np.asarray(labels).astype(str)
    names = sorted(set(as_str.tolist()))
    mapping = {name: idx for idx, name in enumerate(names)}
    encoded = np.asarray([mapping[name] for name in as_str], dtype=np.int64)
    return encoded, names


def purity_score(true_encoded: np.ndarray, pred_clusters: np.ndarray) -> float:
    if len(true_encoded) == 0:
        return 0.0
    total = 0
    for cluster in np.unique(pred_clusters):
        mask = pred_clusters == cluster
        if np.any(mask):
            counts = np.bincount(true_encoded[mask])
            total += int(counts.max())
    return float(total / len(true_encoded))


def hungarian_cluster_accuracy(true_encoded: np.ndarray, pred_clusters: np.ndarray) -> float:
    if len(true_encoded) == 0:
        return 0.0
    true_ids = np.unique(true_encoded)
    cluster_ids = np.unique(pred_clusters)
    matrix = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_map = {int(label): idx for idx, label in enumerate(true_ids)}
    cluster_map = {int(label): idx for idx, label in enumerate(cluster_ids)}
    for true_label, cluster_label in zip(true_encoded, pred_clusters):
        matrix[true_map[int(true_label)], cluster_map[int(cluster_label)]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return float(matrix[row_ind, col_ind].sum() / len(true_encoded))


def evaluate_unknown_subdivision(true_names: np.ndarray, pred_clusters: np.ndarray) -> dict[str, Any]:
    if len(true_names) == 0:
        return {
            "num_evaluated_unknown": 0,
            "num_true_unknown_classes": 0,
            "num_predicted_clusters": 0,
            "nmi": 0.0,
            "ari": 0.0,
            "purity": 0.0,
            "hungarian_accuracy": 0.0,
        }
    true_encoded, class_names = encode_string_labels(true_names)
    return {
        "num_evaluated_unknown": int(len(true_names)),
        "num_true_unknown_classes": int(len(class_names)),
        "num_predicted_clusters": int(len(np.unique(pred_clusters))),
        "nmi": float(normalized_mutual_info_score(true_encoded, pred_clusters)),
        "ari": float(adjusted_rand_score(true_encoded, pred_clusters)),
        "purity": purity_score(true_encoded, pred_clusters),
        "hungarian_accuracy": hungarian_cluster_accuracy(true_encoded, pred_clusters),
    }


