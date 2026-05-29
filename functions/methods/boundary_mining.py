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
) -> np.ndarray:
    local_scale = np.ones(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) <= 1:
            continue

        cls_embeddings = embeddings[cls_indices]
        use_k = min(k, len(cls_indices) - 1)
        cls_scale = np.empty(len(cls_indices), dtype=np.float32)
        for start in range(0, len(cls_indices), chunk_size):
            end = min(start + chunk_size, len(cls_indices))
            distances = squared_euclidean(cls_embeddings[start:end], cls_embeddings)
            row_ids = np.arange(end - start)
            self_ids = np.arange(start, end)
            distances[row_ids, self_ids] = np.inf
            nearest = np.partition(distances, use_k - 1, axis=1)[:, :use_k]
            cls_scale[start:end] = np.sqrt(nearest).mean(axis=1)

        local_scale[cls_indices] = np.maximum(cls_scale, 1e-6)
    return local_scale


def mine_boundary_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    k: int,
    alpha: float,
    top_m: int,
    ordinary_edge_ratio: float,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    num_samples = len(labels)
    labels = labels.astype(np.int64, copy=False)
    local_scale = _same_class_knn_scale(embeddings, labels, k)

    prototype_distances = np.sqrt(squared_euclidean(embeddings, prototypes))
    row_ids = np.arange(num_samples)
    own_distance = prototype_distances[row_ids, labels]
    local_edge = (own_distance / (local_scale + eps)).astype(np.float32)

    foreign_distances = prototype_distances.copy()
    foreign_distances[row_ids, labels] = np.inf
    nearest_foreign = foreign_distances.argmin(axis=1).astype(np.int64)
    nearest_distance = prototype_distances[row_ids, nearest_foreign]
    gap = (nearest_distance - own_distance).astype(np.float32)

    local_edge_norm = _normalize_per_class(local_edge, labels)
    gap_norm = _normalize_per_class(gap, labels)
    scores = alpha * local_edge_norm + (1.0 - alpha) * (1.0 - gap_norm)

    critical_mask = np.zeros(num_samples, dtype=bool)
    ordinary_mask = np.zeros(num_samples, dtype=bool)
    summary = {}

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        cls_scores = scores[cls_indices]
        cls_local = local_edge[cls_indices]

        m = min(top_m, len(cls_indices))
        critical_order = np.argsort(-cls_scores)[:m]
        critical_indices = cls_indices[critical_order]
        critical_mask[critical_indices] = True

        ordinary_count = min(
            max(1, ceil(len(cls_indices) * ordinary_edge_ratio)),
            max(len(cls_indices) - m, 1),
        )
        local_order = np.argsort(-cls_local)
        critical_set = set(int(idx) for idx in critical_indices)
        ordinary_candidates = [idx for idx in cls_indices[local_order] if int(idx) not in critical_set]
        ordinary_indices = ordinary_candidates[:ordinary_count]
        ordinary_mask[ordinary_indices] = True

        summary[int(cls)] = {
            "num_samples": int(len(cls_indices)),
            "num_critical": int(len(critical_indices)),
            "num_ordinary_edge": int(len(ordinary_indices)),
            "critical_indices": critical_indices.tolist(),
            "ordinary_indices": [int(idx) for idx in ordinary_indices],
        }

    return {
        "scores": scores,
        "local_edge": local_edge,
        "gap": gap,
        "local_scale": local_scale,
        "nearest_foreign": nearest_foreign,
        "critical_mask": critical_mask,
        "ordinary_edge_mask": ordinary_mask,
        "summary": summary,
    }



