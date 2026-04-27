from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import softmax
from scipy.stats import weibull_min

from .openmax_backends import RepoOpenMaxAdapter


@dataclass
class WeibullTail:
    shape: float
    loc: float
    scale: float


class OpenMaxCalibrator:
    def __init__(self, alpha_rank: int = 3, tail_size: int = 20, backend: str = "native", distance_type: str = "eucl", euclid_weight: float = 1.0) -> None:
        self.alpha_rank = alpha_rank
        self.tail_size = tail_size
        self.backend = backend
        self.distance_type = distance_type
        self.euclid_weight = euclid_weight
        self.mavs: np.ndarray | None = None
        self.tails: Dict[int, WeibullTail] = {}
        self.adapter = None

    def fit(self, activations: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> None:
        if self.backend == "repo_openmax":
            self.adapter = RepoOpenMaxAdapter(
                alpha_rank=self.alpha_rank,
                tail_size=self.tail_size,
                distance_type=self.distance_type,
            )
            self.adapter.fit(activations, labels, predictions)
            return
        num_classes = activations.shape[1]
        self.mavs = np.zeros((num_classes, activations.shape[1]), dtype=np.float32)
        self.tails = {}
        for cls in range(num_classes):
            mask = (labels == cls) & (predictions == cls)
            cls_acts = activations[mask]
            if len(cls_acts) == 0:
                mask = labels == cls
                cls_acts = activations[mask]
            if len(cls_acts) == 0:
                self.mavs[cls] = np.zeros(activations.shape[1], dtype=np.float32)
                self.tails[cls] = WeibullTail(shape=1.0, loc=0.0, scale=1.0)
                continue
            self.mavs[cls] = cls_acts.mean(axis=0)
            dists = np.linalg.norm(cls_acts - self.mavs[cls], axis=1)
            dists = np.sort(dists)
            tail = dists[-min(self.tail_size, len(dists)) :]
            if len(tail) == 0:
                tail = np.array([1.0], dtype=np.float32)
            shape, loc, scale = weibull_min.fit(tail, floc=0.0)
            self.tails[cls] = WeibullTail(float(shape), float(loc), float(scale))

    def _wscore(self, cls: int, activation: np.ndarray) -> float:
        if self.mavs is None:
            raise RuntimeError("OpenMax calibrator is not fitted.")
        tail = self.tails[cls]
        dist = np.linalg.norm(activation - self.mavs[cls])
        return float(weibull_min.cdf(dist, tail.shape, loc=tail.loc, scale=tail.scale))

    def predict(self, activations: np.ndarray) -> Dict[str, np.ndarray]:
        if self.backend == "repo_openmax":
            if self.adapter is None:
                raise RuntimeError("OpenMax adapter is not fitted.")
            return self.adapter.predict(activations)
        if self.mavs is None:
            raise RuntimeError("OpenMax calibrator is not fitted.")

        known_probs = []
        unknown_probs = []
        revised_acts = []
        for act in activations:
            top_ranked = np.argsort(act)[::-1][: self.alpha_rank]
            revised = act.copy()
            unknown_activation = 0.0
            for rank, cls in enumerate(top_ranked):
                omega = float(self.alpha_rank - rank) / float(max(self.alpha_rank, 1))
                wscore = self._wscore(int(cls), act)
                reduced = revised[cls] * (1.0 - omega * wscore)
                unknown_activation += max(revised[cls] - reduced, 0.0)
                revised[cls] = reduced
            logits = np.concatenate([revised, np.asarray([unknown_activation], dtype=np.float32)])
            probs = softmax(logits)
            known_probs.append(probs[:-1])
            unknown_probs.append(probs[-1])
            revised_acts.append(revised)

        known_probs = np.asarray(known_probs, dtype=np.float32)
        unknown_probs = np.asarray(unknown_probs, dtype=np.float32)
        revised_acts = np.asarray(revised_acts, dtype=np.float32)
        return {
            "known_probs": known_probs,
            "unknown_prob": unknown_probs,
            "pred": known_probs.argmax(axis=1),
            "revised_activations": revised_acts,
        }

    def state_dict(self) -> Dict[str, object]:
        if self.backend == "repo_openmax" and self.adapter is not None:
            return self.adapter.state_dict()
        return {
            "alpha_rank": self.alpha_rank,
            "tail_size": self.tail_size,
            "backend": self.backend,
            "distance_type": self.distance_type,
            "euclid_weight": self.euclid_weight,
            "mavs": self.mavs,
            "tails": {cls: tail.__dict__ for cls, tail in self.tails.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "OpenMaxCalibrator":
        backend = str(state.get("backend", "native"))
        if backend == "repo_openmax":
            obj = cls(alpha_rank=int(state["alpha_rank"]), tail_size=int(state["tail_size"]), backend=backend, distance_type=str(state.get("distance_type", "eucl")))
            obj.adapter = RepoOpenMaxAdapter.from_state_dict(state)
            return obj
        obj = cls(
            alpha_rank=int(state["alpha_rank"]),
            tail_size=int(state["tail_size"]),
            backend=backend,
            distance_type=str(state.get("distance_type", "eucl")),
            euclid_weight=float(state.get("euclid_weight", 1.0)),
        )
        obj.mavs = np.asarray(state["mavs"], dtype=np.float32)
        obj.tails = {
            int(cls): WeibullTail(**tail_state)
            for cls, tail_state in dict(state["tails"]).items()
        }
        return obj
