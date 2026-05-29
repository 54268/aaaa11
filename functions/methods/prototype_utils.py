from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    dim = embeddings.shape[1]
    prototypes = np.zeros((num_classes, dim), dtype=np.float32)
    for cls in range(num_classes):
        mask = labels == cls
        if not np.any(mask):
            continue
        prototypes[cls] = embeddings[mask].mean(axis=0)
    return prototypes


def squared_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.sum(a * a, axis=1, keepdims=True)
    b_norm = np.sum(b * b, axis=1, keepdims=True).T
    distances = a_norm + b_norm - 2.0 * (a @ b.T)
    return np.maximum(distances, 0.0)


def predict_with_prototypes(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = squared_euclidean(embeddings, prototypes)
    logits = -distances / max(temperature, 1e-6)
    pred = logits.argmax(axis=1)
    return pred, logits, distances


def activations_from_distances(distances: np.ndarray) -> np.ndarray:
    raw = -distances
    shift = raw.min(axis=1, keepdims=True)
    return raw - shift + 1e-6


def collect_distance_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    pred, _, distances = predict_with_prototypes(embeddings, prototypes, temperature)
    pred_dist = distances[np.arange(len(labels)), pred]

    mu = np.zeros(len(prototypes), dtype=np.float32)
    sigma = np.ones(len(prototypes), dtype=np.float32)
    for cls in range(len(prototypes)):
        mask = (labels == cls) & (pred == cls)
        cls_dist = distances[mask, cls]
        if len(cls_dist) == 0:
            cls_dist = distances[labels == cls, cls]
        mu[cls] = float(cls_dist.mean()) if len(cls_dist) else 0.0
        sigma[cls] = float(cls_dist.std()) if len(cls_dist) else 1.0
    return {
        "pred": pred,
        "pred_distance": pred_dist,
        "mu": mu,
        "sigma": np.maximum(sigma, 1e-6),
        "distances": distances,
    }



