from __future__ import annotations

import sys
from pathlib import Path

import torch


def _register_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class EpisodicProtoRegularizer:
    """
    Thin adapter around the cloned Orobix ProtoNet loss.
    We keep it optional because your main method is not episodic few-shot learning.
    """

    def __init__(self, n_support: int = 2) -> None:
        repo_src = Path(__file__).resolve().parents[2] / "third_party" / "prototypical-networks" / "src"
        _register_path(repo_src)
        from prototypical_loss import prototypical_loss  # type: ignore

        self._loss_fn = prototypical_loss
        self.n_support = n_support

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes, counts = torch.unique(labels, return_counts=True)
        if len(classes) < 2:
            return embeddings.new_tensor(0.0)
        if int(counts.min().item()) <= self.n_support:
            return embeddings.new_tensor(0.0)
        loss, _ = self._loss_fn(embeddings, labels, self.n_support)
        return loss.to(embeddings.device)
