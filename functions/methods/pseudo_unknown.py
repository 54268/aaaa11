from __future__ import annotations

from typing import Dict

import numpy as np


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def generate_hybrid_pseudo_unknown(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    boundary_result: Dict[str, np.ndarray],
    ordinary_eta: float,
    critical_eta: float,
    critical_beta: float,
    ordinary_variations: int,
    critical_variations: int,
    jitter: float,
    enable_conflict_protection: bool = True,
    use_critical_boundary: bool = True,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pseudo_embeddings = []
    pseudo_labels = []
    source_indices = []
    source_classes = []
    pseudo_kind = []

    def _append(
        index: int,
        kind: str,
        direction: np.ndarray,
        base_eta: float,
        variations: int,
        eta_multiplier: float = 1.0,
        variation_bonus: int = 0,
    ) -> None:
        scale = float(boundary_result["local_scale"][index])
        delta = base_eta * eta_multiplier * scale
        total_variations = max(1, int(variations + variation_bonus))
        for _ in range(total_variations):
            jitter_scale = 1.0 + rng.uniform(-jitter, jitter)
            pseudo_embeddings.append(embeddings[index] + delta * jitter_scale * direction)
            pseudo_labels.append(-1)
            source_indices.append(index)
            source_classes.append(int(labels[index]))
            pseudo_kind.append(kind)

    critical_indices = np.where(boundary_result["critical_mask"])[0]
    ordinary_indices = np.where(boundary_result["ordinary_edge_mask"])[0]
    if not use_critical_boundary:
        marginal_mask = boundary_result.get("marginal_mask")
        if marginal_mask is not None:
            ordinary_indices = np.where(np.asarray(marginal_mask, dtype=bool))[0]
        else:
            ordinary_indices = np.unique(np.concatenate([ordinary_indices, critical_indices]))
        critical_indices = np.asarray([], dtype=np.int64)

    for index in ordinary_indices:
        cls = int(labels[index])
        direction = _normalize(embeddings[index] - prototypes[cls])
        score = float(boundary_result["scores"][index])
        eta_multiplier = 0.9 + 0.35 * score
        variation_bonus = 1 if score >= 0.85 else 0
        _append(
            index,
            "ordinary_edge",
            direction,
            ordinary_eta,
            ordinary_variations,
            eta_multiplier=eta_multiplier,
            variation_bonus=variation_bonus,
        )

    for index in critical_indices:
        cls = int(labels[index])
        foreign_cls = int(boundary_result["nearest_foreign"][index])
        normal_dir = _normalize(embeddings[index] - prototypes[cls])
        repel_dir = _normalize(embeddings[index] - prototypes[foreign_cls])
        if enable_conflict_protection and float(np.dot(normal_dir, repel_dir)) < 0.0:
            repel_dir = repel_dir - float(np.dot(normal_dir, repel_dir)) * normal_dir
            repel_dir = _normalize(repel_dir)
        direction = _normalize(critical_beta * normal_dir + (1.0 - critical_beta) * repel_dir)
        score = float(boundary_result["scores"][index])
        eta_multiplier = 1.0 + 0.6 * score
        variation_bonus = int(score >= 0.75) + int(score >= 0.9)
        _append(
            index,
            "critical_boundary",
            direction,
            critical_eta,
            critical_variations,
            eta_multiplier=eta_multiplier,
            variation_bonus=variation_bonus,
        )

    pseudo_embeddings = np.asarray(pseudo_embeddings, dtype=np.float32).reshape(
        -1,
        embeddings.shape[1],
    )
    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    source_classes = np.asarray(source_classes, dtype=np.int64)
    pseudo_kind = np.asarray(pseudo_kind)

    summary = {
        "num_ordinary_pseudo": int(np.sum(pseudo_kind == "ordinary_edge")),
        "num_critical_pseudo": int(np.sum(pseudo_kind == "critical_boundary")),
        "num_total_pseudo": int(len(pseudo_kind)),
        "use_critical_boundary": bool(use_critical_boundary),
    }
    return {
        "pseudo_embeddings": pseudo_embeddings,
        "pseudo_labels": pseudo_labels,
        "source_indices": source_indices,
        "source_classes": source_classes,
        "pseudo_kind": pseudo_kind,
        "summary": summary,
    }



