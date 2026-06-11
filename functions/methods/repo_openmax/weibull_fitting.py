from __future__ import annotations

import numpy as np
import scipy.stats
import torch

from .distances import compute_distance


class WeibullFitting:
    translation = 10000.0
    shape_a = 1.0
    location = 0.0

    def __init__(self, tailsize: int = 10, num_classes: int = 31) -> None:
        self.tailsize = int(tailsize)
        self.num_classes = int(num_classes)
        self.weibull_models: list[dict[str, float]] | None = None

    def fit_high(self, values: torch.Tensor | np.ndarray) -> dict[str, float]:
        values_np = np.asarray(values, dtype=np.float64)
        if values_np.size == 0:
            raise ValueError("Cannot fit a Weibull model without samples.")
        tail_size = min(self.tailsize, values_np.size)
        tail = np.sort(values_np)[-tail_size:]
        min_val = float(tail[0])
        shifted = tail + self.translation - min_val
        params = scipy.stats.exponweib.fit(shifted, floc=0, f0=1)
        return {
            "c": float(params[1]),
            "scale": float(params[3]),
            "min_val": min_val,
        }

    def fit_all_models(
        self,
        distances: list[torch.Tensor | np.ndarray],
    ) -> list[dict[str, float]]:
        if len(distances) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class distance arrays, got {len(distances)}."
            )
        self.weibull_models = [self.fit_high(values) for values in distances]
        return self.weibull_models

    @classmethod
    def w_score(
        cls,
        distances: torch.Tensor | np.ndarray,
        weibull_model: dict[str, float],
    ) -> np.ndarray:
        values = (
            np.asarray(distances, dtype=np.float64)
            + cls.translation
            - float(weibull_model["min_val"])
        )
        return scipy.stats.exponweib.cdf(
            values,
            a=cls.shape_a,
            c=float(weibull_model["c"]),
            loc=cls.location,
            scale=float(weibull_model["scale"]),
        )

    def compute_centroids_and_distances(
        self,
        samples: list[torch.Tensor],
        distance_type: str = "eucl",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(samples) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class sample tensors, got {len(samples)}."
            )
        distances: list[torch.Tensor] = []
        centroids: list[torch.Tensor] = []
        for class_samples in samples:
            if len(class_samples) == 0:
                raise ValueError("Each class must contain at least one sample.")
            centroid = class_samples.mean(dim=0)
            distances.append(
                compute_distance(class_samples, centroid, distance_type=distance_type)
            )
            centroids.append(centroid)
        return distances, centroids
