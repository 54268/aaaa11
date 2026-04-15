from __future__ import annotations

from math import ceil
from typing import Dict

import numpy as np


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
    local_edge = np.zeros(num_samples, dtype=np.float32)
    gap = np.zeros(num_samples, dtype=np.float32)
    nearest_foreign = np.zeros(num_samples, dtype=np.int64)
    local_scale = np.zeros(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        cls = int(labels[idx])
        cls_proto = prototypes[cls]
        scale = _avg_same_class_knn_distance(embeddings, labels, idx, k)
        local_scale[idx] = scale
        local_edge[idx] = float(np.linalg.norm(embeddings[idx] - cls_proto) / (scale + eps))

        foreign_ids = [j for j in range(len(prototypes)) if j != cls]
        foreign_protos = prototypes[foreign_ids]
        foreign_dist = np.linalg.norm(foreign_protos - embeddings[idx], axis=1)
        nearest_idx = int(np.argmin(foreign_dist))
        nearest_cls = foreign_ids[nearest_idx]
        nearest_foreign[idx] = nearest_cls
        gap[idx] = float(
            np.linalg.norm(embeddings[idx] - prototypes[nearest_cls])
            - np.linalg.norm(embeddings[idx] - cls_proto)
        )

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
        ordinary_candidates = [idx for idx in cls_indices[local_order] if idx not in critical_indices]
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
