from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _register_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RepoOpenMaxAdapter:
    def __init__(self, alpha_rank: int = 3, tail_size: int = 20, distance_type: str = "eucl") -> None:
        repo_root = _project_root() / "third_party" / "openmax"
        _register_path(repo_root)
        from openmax.openmax import OpenMax  # type: ignore
        from openmax.weibull_fitting import WeibullFitting  # type: ignore

        self._OpenMax = OpenMax
        self._WeibullFitting = WeibullFitting
        self.alpha_rank = alpha_rank
        self.tail_size = tail_size
        self.distance_type = distance_type
        self.centroids = None
        self.weibull_models = None
        self._model = None

    def fit(self, activations: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> None:
        import torch

        classes = sorted(np.unique(labels).tolist())
        samples = []
        for cls in classes:
            mask = (labels == cls) & (predictions == cls)
            cls_acts = activations[mask]
            if len(cls_acts) == 0:
                cls_acts = activations[labels == cls]
            samples.append(torch.from_numpy(cls_acts.astype(np.float32)))

        fitter = self._WeibullFitting(tailsize=self.tail_size, num_classes=len(classes))
        distances, centroids = fitter.compute_centroids_and_distances(samples, distance_type=self.distance_type)
        weibull_models = fitter.fit_all_models(distances)

        self.centroids = [c.detach().cpu() for c in centroids]
        self.weibull_models = weibull_models
        self._model = self._OpenMax(
            centroids=self.centroids,
            weibull_models=self.weibull_models,
            alpha=self.alpha_rank,
            distance_type=self.distance_type,
        )

    def predict(self, activations: np.ndarray) -> Dict[str, np.ndarray]:
        import torch

        if self._model is None:
            raise RuntimeError("RepoOpenMaxAdapter is not fitted.")
        logits_hat = self._model.recalibrate_logits(torch.from_numpy(activations.astype(np.float32)))
        probs = self._model.compute_probs(logits_hat).detach().cpu().numpy()
        return {
            "known_probs": probs[:, 1:],
            "unknown_prob": probs[:, 0],
            "pred": probs[:, 1:].argmax(axis=1),
            "revised_activations": logits_hat[:, 1:].detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "alpha_rank": self.alpha_rank,
            "tail_size": self.tail_size,
            "distance_type": self.distance_type,
            "centroids": [c.numpy() for c in self.centroids] if self.centroids is not None else None,
            "weibull_models": self.weibull_models,
            "backend": "repo_openmax",
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "RepoOpenMaxAdapter":
        obj = cls(
            alpha_rank=int(state["alpha_rank"]),
            tail_size=int(state["tail_size"]),
            distance_type=str(state["distance_type"]),
        )
        obj.centroids = [np.asarray(c, dtype=np.float32) for c in list(state["centroids"])]
        obj.weibull_models = state["weibull_models"]
        import torch

        obj._model = obj._OpenMax(
            centroids=[torch.from_numpy(c) for c in obj.centroids],
            weibull_models=obj.weibull_models,
            alpha=obj.alpha_rank,
            distance_type=obj.distance_type,
        )
        return obj
