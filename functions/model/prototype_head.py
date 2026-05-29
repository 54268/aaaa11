from __future__ import annotations

import torch
from torch import nn


class PrototypeClassifierHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        temperature: float = 1.0,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embedding_dim))

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = torch.cdist(embeddings, self.prototypes, p=2.0) ** 2
        logits = -distances / max(self.temperature, 1e-6)
        return logits, distances

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        self.prototypes.copy_(prototypes)

    @torch.no_grad()
    def update_ema(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        for cls in labels.unique():
            mask = labels == cls
            cls_mean = embeddings[mask].mean(dim=0)
            self.prototypes[int(cls)] = (
                self.momentum * self.prototypes[int(cls)] + (1.0 - self.momentum) * cls_mean
            )



