from __future__ import annotations

import torch


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1).clamp_min(1e-12)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    cosine_sim = (x * y).sum(dim=-1) / (_safe_norm(x) * _safe_norm(y))
    return 1.0 - cosine_sim.clamp(-1.0, 1.0)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).norm(dim=-1)


def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    cosine_sim = x @ y.T / torch.outer(_safe_norm(x), _safe_norm(y))
    return 1.0 - cosine_sim.clamp(-1.0, 1.0)


def pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y)


def compute_distance(
    samples: torch.Tensor,
    centroid: torch.Tensor,
    distance_type: str = "eucl",
    euclid_weight: float = 1.0 / 200.0,
) -> torch.Tensor:
    if distance_type == "eucos":
        return (
            euclid_weight * euclidean_distance(samples, centroid)
            + cosine_distance(samples, centroid)
        )
    if distance_type == "eucl":
        return euclidean_distance(samples, centroid)
    if distance_type == "cos":
        return cosine_distance(samples, centroid)
    raise ValueError(
        "distance_type must be one of {'eucl', 'cos', 'eucos'}, "
        f"got {distance_type!r}"
    )


def compute_pairwise_distance(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    distance_type: str = "eucl",
    euclid_weight: float = 1.0 / 200.0,
) -> torch.Tensor:
    if distance_type == "eucos":
        return (
            euclid_weight * pairwise_euclidean_distance(samples, centroids)
            + pairwise_cosine_distance(samples, centroids)
        )
    if distance_type == "eucl":
        return pairwise_euclidean_distance(samples, centroids)
    if distance_type == "cos":
        return pairwise_cosine_distance(samples, centroids)
    raise ValueError(
        "distance_type must be one of {'eucl', 'cos', 'eucos'}, "
        f"got {distance_type!r}"
    )
