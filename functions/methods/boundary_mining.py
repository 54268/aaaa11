from __future__ import annotations

from math import ceil
from typing import Dict

import numpy as np

from .prototype_utils import squared_euclidean


def _normalize_per_class(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        v = values[mask]
        v_min = float(v.min())
        v_max = float(v.max())
        if abs(v_max - v_min) < 1e-8:
            out[mask] = 0.0
        else:
            out[mask] = (v - v_min) / (v_max - v_min)
    return out


def _avg_same_class_knn_distance(
    embeddings: np.ndarray,
    labels: np.ndarray,
    index: int,
    k: int,
) -> float:
    cls = labels[index]
    cls_indices = np.where(labels == cls)[0]
    cls_indices = cls_indices[cls_indices != index]
    if len(cls_indices) == 0:
        return 1.0
    dists = np.linalg.norm(embeddings[cls_indices] - embeddings[index], axis=1)
    use_k = min(k, len(dists))
    return float(np.partition(dists, use_k - 1)[:use_k].mean())


def _same_class_knn_scale(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    local_sparsity = np.ones(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) <= 1:
            continue

        cls_embeddings = embeddings[cls_indices]
        use_k = min(k, len(cls_indices) - 1)
        cls_sparsity = np.empty(len(cls_indices), dtype=np.float32)
        for start in range(0, len(cls_indices), chunk_size):
            end = min(start + chunk_size, len(cls_indices))
            distances = squared_euclidean(cls_embeddings[start:end], cls_embeddings)
            row_ids = np.arange(end - start)
            self_ids = np.arange(start, end)
            distances[row_ids, self_ids] = np.inf
            nearest = np.partition(distances, use_k - 1, axis=1)[:, :use_k]
            cls_sparsity[start:end] = nearest.mean(axis=1)

        local_sparsity[cls_indices] = np.maximum(cls_sparsity, 1e-12)
    local_scale = np.sqrt(local_sparsity).astype(np.float32)
    return local_sparsity, local_scale


def mine_boundary_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    k: int,
    beta: float,
    alpha: float,
    top_m: int,
    ordinary_edge_ratio: float,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    num_samples = len(labels)
    labels = labels.astype(np.int64, copy=False)
    local_sparsity, local_scale = _same_class_knn_scale(embeddings, labels, k)

    prototype_distances_sq = squared_euclidean(embeddings, prototypes)
    prototype_distances = np.sqrt(prototype_distances_sq)
    row_ids = np.arange(num_samples)
    prototype_deviation = prototype_distances_sq[row_ids, labels].astype(np.float32)
    own_distance = prototype_distances[row_ids, labels]
    local_marginality = (prototype_deviation + float(beta) * local_sparsity).astype(np.float32)

    foreign_distances = prototype_distances.copy()
    foreign_distances[row_ids, labels] = np.inf
    nearest_foreign = foreign_distances.argmin(axis=1).astype(np.int64)
    nearest_distance = prototype_distances[row_ids, nearest_foreign]
    gap = (nearest_distance - own_distance).astype(np.float32)

    nearest_distance_sq = prototype_distances_sq[row_ids, nearest_foreign]
    prototype_separation = np.linalg.norm(
        prototypes[nearest_foreign] - prototypes[labels],
        axis=1,
    )
    competition_distance = (
        (nearest_distance_sq - prototype_deviation)
        / np.maximum(2.0 * prototype_separation, eps)
    ).astype(np.float32)

    marginality_norm = _normalize_per_class(local_marginality, labels)
    competition_norm = _normalize_per_class(competition_distance, labels)
    scores = alpha * marginality_norm + (1.0 - alpha) * (1.0 - competition_norm)

    critical_mask = np.zeros(num_samples, dtype=bool)
    ordinary_mask = np.zeros(num_samples, dtype=bool)
    marginal_mask = np.zeros(num_samples, dtype=bool)
    noise_mask = competition_distance < 0.0
    summary = {}

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        valid_indices = cls_indices[~noise_mask[cls_indices]]
        ordinary_count = min(
            max(1, ceil(len(cls_indices) * ordinary_edge_ratio)),
            max(len(valid_indices) - min(top_m, len(valid_indices)), 0),
        )
        m = min(top_m, max(len(valid_indices) - ordinary_count, 0))
        marginal_count = min(len(valid_indices), m + ordinary_count)

        if marginal_count:
            marginal_order = np.argsort(-local_marginality[valid_indices])[:marginal_count]
            marginal_indices = valid_indices[marginal_order]
            marginal_mask[marginal_indices] = True
            marginal_threshold = float(local_marginality[marginal_indices].min())
        else:
            marginal_indices = np.asarray([], dtype=np.int64)
            marginal_threshold = float("nan")

        if m:
            critical_order = np.argsort(competition_distance[marginal_indices])[:m]
            critical_indices = marginal_indices[critical_order]
            competition_threshold = float(competition_distance[critical_indices].max())
        else:
            critical_indices = np.asarray([], dtype=np.int64)
            competition_threshold = float("nan")
        critical_mask[critical_indices] = True

        critical_set = set(int(idx) for idx in critical_indices)
        ordinary_candidates = [
            int(idx)
            for idx in marginal_indices[np.argsort(-local_marginality[marginal_indices])]
            if int(idx) not in critical_set
        ]
        ordinary_indices = ordinary_candidates[:ordinary_count]
        ordinary_mask[ordinary_indices] = True

        summary[int(cls)] = {
            "num_samples": int(len(cls_indices)),
            "num_noise_excluded": int(noise_mask[cls_indices].sum()),
            "num_marginal": int(len(marginal_indices)),
            "num_critical": int(len(critical_indices)),
            "num_ordinary_edge": int(len(ordinary_indices)),
            "marginal_threshold": marginal_threshold,
            "competition_threshold": competition_threshold,
            "beta": float(beta),
            "critical_indices": critical_indices.tolist(),
            "ordinary_indices": [int(idx) for idx in ordinary_indices],
        }

    return {
        "scores": scores,
        "local_edge": local_marginality,
        "prototype_deviation": prototype_deviation,
        "local_sparsity": local_sparsity,
        "local_marginality": local_marginality,
        "gap": gap,
        "competition_distance": competition_distance,
        "local_scale": local_scale,
        "nearest_foreign": nearest_foreign,
        "marginal_mask": marginal_mask,
        "critical_mask": critical_mask,
        "ordinary_edge_mask": ordinary_mask,
        "noise_mask": noise_mask,
        "summary": summary,
    }



