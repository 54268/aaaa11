from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from functions.model.cvcnn_iq import CVCNNBackbone
from functions.model.prototype_head import PrototypeClassifierHead
from functions.methods.prototype_utils import compute_prototypes
from functions.common.io import ensure_dir


def _choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


@dataclass
class TrainingArtifacts:
    checkpoint_path: str
    best_val_acc: float


class ClosedSetTrainer:
    def __init__(self, config: Dict, num_classes: int, signal_length: int) -> None:
        self.config = config
        self.device = _choose_device(config["train"]["device"])
        embedding_dim = int(config["model"]["embedding_dim"])
        self.backbone = CVCNNBackbone(
            signal_length=signal_length,
            embedding_dim=embedding_dim,
            hidden_dim=int(config["model"].get("hidden_dim", 32)),
            dropout=float(config["model"].get("dropout", 0.2)),
        ).to(self.device)
        self.head = PrototypeClassifierHead(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            temperature=float(config["model"].get("temperature", 1.0)),
            momentum=float(config["train"].get("prototype_momentum", 0.9)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"].get("weight_decay", 0.0)),
        )
        self._last_embeddings = None

    def _scheme_regularization_loss(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config["loss"]
        basic = F.cross_entropy(logits, labels)

        norm_embeddings = F.normalize(embeddings, dim=1)
        norm_prototypes = F.normalize(self.head.prototypes, dim=1)
        cosine = torch.matmul(norm_embeddings, norm_prototypes.T)
        pos = cosine[torch.arange(len(labels), device=labels.device), labels]
        neg = cosine.masked_fill(
            F.one_hot(labels, num_classes=self.head.num_classes).bool(),
            float("-inf"),
        ).max(dim=1).values
        angle_margin = float(loss_cfg.get("angle_margin", 0.2))
        angle = F.relu(angle_margin - (pos - neg)).mean()
        target_prototypes = self.head.prototypes[labels]
        prototype = F.mse_loss(embeddings, target_prototypes)

        total = (
            float(loss_cfg.get("lambda_basic", 1.0)) * basic
            + float(loss_cfg.get("lambda_angle", 0.0)) * angle
            + float(loss_cfg.get("lambda_prototype", 0.0)) * prototype
        )
        return {
            "total": total,
            "basic": basic,
            "angle": angle,
            "prototype": prototype,
        }

    @torch.no_grad()
    def refresh_prototypes(self, train_loader) -> None:
        self.backbone.eval()
        embeddings = []
        labels = []
        for x, y in train_loader:
            x = x.to(self.device)
            emb = self.backbone(x).cpu().numpy()
            embeddings.append(emb)
            labels.append(y.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        prototypes = compute_prototypes(embeddings, labels, self.head.num_classes)
        self.head.set_prototypes(torch.from_numpy(prototypes).to(self.device))

    @torch.no_grad()
    def extract_embeddings(self, loader) -> Dict[str, np.ndarray]:
        self.backbone.eval()
        self.head.eval()
        embeddings = []
        labels = []
        logits_all = []
        distances_all = []
        pred_all = []
        for x, y in loader:
            x = x.to(self.device)
            emb = self.backbone(x)
            logits, distances = self.head(emb)
            pred = logits.argmax(dim=1)
            embeddings.append(emb.cpu().numpy())
            labels.append(y.numpy())
            logits_all.append(logits.cpu().numpy())
            distances_all.append(distances.cpu().numpy())
            pred_all.append(pred.cpu().numpy())
        return {
            "embeddings": np.concatenate(embeddings, axis=0),
            "labels": np.concatenate(labels, axis=0),
            "logits": np.concatenate(logits_all, axis=0),
            "distances": np.concatenate(distances_all, axis=0),
            "pred": np.concatenate(pred_all, axis=0),
            "prototypes": self.head.prototypes.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def evaluate_known_accuracy(self, loader) -> float:
        payload = self.extract_embeddings(loader)
        return float((payload["pred"] == payload["labels"]).mean())

    def fit(self, train_loader, val_loader, output_dir: str | Path) -> TrainingArtifacts:
        output_dir = ensure_dir(output_dir)
        checkpoint_path = output_dir / "best_closed_set.pt"

        self.refresh_prototypes(train_loader)
        best_val_acc = -1.0
        epochs = int(self.config["train"]["epochs"])

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            running_loss = 0.0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                embeddings = self.backbone(x)
                self._last_embeddings = embeddings
                logits, distances = self.head(embeddings)
                loss_dict = self._scheme_regularization_loss(embeddings, logits, y)

                self.optimizer.zero_grad()
                loss_dict["total"].backward()
                nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5.0)
                self.optimizer.step()
                self.head.update_ema(embeddings.detach(), y.detach())
                running_loss += float(loss_dict["total"].item())

            self.refresh_prototypes(train_loader)
            val_acc = self.evaluate_known_accuracy(val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.state_dict(),
                        "config": self.config,
                        "best_val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={running_loss / max(len(train_loader), 1):.4f} "
                f"val_acc={val_acc:.4f}"
            )

        return TrainingArtifacts(checkpoint_path=str(checkpoint_path), best_val_acc=best_val_acc)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.head.load_state_dict(checkpoint["head"])



