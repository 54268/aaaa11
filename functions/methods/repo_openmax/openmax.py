from __future__ import annotations

import numpy as np
import torch

from .distances import compute_pairwise_distance
from .weibull_fitting import WeibullFitting


class OpenMax:
    def __init__(
        self,
        centroids: list[torch.Tensor],
        weibull_models: list[dict[str, float]],
        alpha: int = 10,
        distance_type: str = "eucl",
    ) -> None:
        if distance_type not in {"eucl", "cos", "eucos"}:
            raise ValueError(
                "distance_type must be one of {'eucl', 'cos', 'eucos'}, "
                f"got {distance_type!r}"
            )
        self.centroids = torch.stack(centroids).detach().cpu()
        self.weibull_models = weibull_models
        self.alpha = min(int(alpha), len(centroids))
        self.distance_type = distance_type
        self.num_classes = int(self.centroids.shape[0])

    def recalibrate_logits(
        self,
        logits: torch.Tensor,
        embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embs is None:
            embs = logits
        logits = logits.detach().cpu()
        embs = embs.detach().cpu()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if embs.ndim == 1:
            embs = embs.unsqueeze(0)

        top_preds = logits.argsort(dim=-1, descending=True)[:, : self.alpha]
        alpha_coeffs = torch.zeros_like(logits)
        for rank in range(self.alpha):
            coefficient = float(self.alpha - rank) / float(self.alpha)
            alpha_coeffs.scatter_(
                1,
                top_preds[:, rank : rank + 1],
                coefficient,
            )

        distances = compute_pairwise_distance(
            embs,
            self.centroids,
            distance_type=self.distance_type,
        )
        weibull_probs = np.zeros(distances.shape, dtype=np.float32)
        distances_np = distances.detach().cpu().numpy()
        for class_index in range(self.num_classes):
            weibull_probs[:, class_index] = WeibullFitting.w_score(
                distances_np[:, class_index],
                self.weibull_models[class_index],
            )
        weibull_probs_tensor = torch.from_numpy(weibull_probs).to(logits.dtype)

        recalibrated_known = logits * (1.0 - alpha_coeffs * weibull_probs_tensor)
        unknown_logit = (logits - recalibrated_known).sum(dim=-1, keepdim=True)
        return torch.cat([unknown_logit, recalibrated_known], dim=-1)

    @staticmethod
    def compute_probs(logits_hat: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(logits_hat, dim=-1)
