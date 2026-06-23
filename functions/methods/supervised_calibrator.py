from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from functions.common.metrics import evaluate_open_set
from functions.methods.fusion import apply_unknown_rejection


def build_calibrator_features(
    q_om: np.ndarray,
    q_pd: np.ndarray,
    fusion_lambda: float,
) -> np.ndarray:
    q_om = np.asarray(q_om, dtype=np.float32)
    q_pd = np.asarray(q_pd, dtype=np.float32)
    if q_om.shape != q_pd.shape:
        raise ValueError("q_om and q_pd must have equal shapes.")
    fused = float(fusion_lambda) * q_om + (1.0 - float(fusion_lambda)) * q_pd
    return np.column_stack([q_om, q_pd, fused]).astype(np.float32)


class _CalibratorNet(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


class UnknownScoreCalibrator:
    def __init__(
        self,
        hidden_dim: int = 8,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.net = _CalibratorNet(self.hidden_dim)
        self.feature_mean = np.zeros(3, dtype=np.float32) if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.ones(3, dtype=np.float32) if feature_std is None else np.asarray(feature_std, dtype=np.float32)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        return (
            np.asarray(features, dtype=np.float32) - self.feature_mean
        ) / np.maximum(self.feature_std, 1e-6)

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.net.eval()
        tensor = torch.from_numpy(self._normalize(features))
        return torch.sigmoid(self.net(tensor)).cpu().numpy()


@dataclass
class CalibratorTrainingResult:
    model: UnknownScoreCalibrator
    loss_history: list[float]
    num_known: int
    num_pseudo: int
    seed: int
    epochs: int
    lr: float


def train_calibrator(
    known_features: np.ndarray,
    pseudo_features: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int = 8,
) -> CalibratorTrainingResult:
    known_features = np.asarray(known_features, dtype=np.float32)
    pseudo_features = np.asarray(pseudo_features, dtype=np.float32)
    if known_features.ndim != 2 or known_features.shape[1] != 3:
        raise ValueError("known_features must have shape [N, 3].")
    if pseudo_features.ndim != 2 or pseudo_features.shape[1] != 3:
        raise ValueError("pseudo_features must have shape [N, 3].")
    if len(known_features) == 0 or len(pseudo_features) == 0:
        raise ValueError("known and pseudo feature sets must be non-empty.")

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    count = min(len(known_features), len(pseudo_features))
    known_idx = rng.choice(len(known_features), size=count, replace=False)
    pseudo_idx = rng.choice(len(pseudo_features), size=count, replace=False)
    x = np.concatenate([known_features[known_idx], pseudo_features[pseudo_idx]])
    y = np.concatenate(
        [
            np.zeros(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
        ]
    )
    order = rng.permutation(len(x))
    x = x[order]
    y = y[order]
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0) + 1e-6

    model = UnknownScoreCalibrator(
        hidden_dim=hidden_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    optimizer = torch.optim.Adam(model.net.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(model._normalize(x))
    y_tensor = torch.from_numpy(y)
    history = []
    for _ in range(int(epochs)):
        model.net.train()
        logits = model.net(x_tensor)
        loss = loss_fn(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return CalibratorTrainingResult(
        model=model,
        loss_history=history,
        num_known=count,
        num_pseudo=count,
        seed=int(seed),
        epochs=int(epochs),
        lr=float(lr),
    )


def save_calibrator(path: str | Path, result: CalibratorTrainingResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_dim": result.model.hidden_dim,
            "state_dict": result.model.net.state_dict(),
            "feature_mean": result.model.feature_mean,
            "feature_std": result.model.feature_std,
            "loss_history": result.loss_history,
            "num_known": result.num_known,
            "num_pseudo": result.num_pseudo,
            "seed": result.seed,
            "epochs": result.epochs,
            "lr": result.lr,
        },
        path,
    )


def load_calibrator(path: str | Path) -> UnknownScoreCalibrator:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model = UnknownScoreCalibrator(
        hidden_dim=int(payload["hidden_dim"]),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
    )
    model.net.load_state_dict(payload["state_dict"])
    model.net.eval()
    return model


def apply_supervised_calibrator_score(
    *,
    q_om: np.ndarray,
    q_pd: np.ndarray,
    fusion_lambda: float,
    calibrator_path: str | Path,
) -> np.ndarray:
    features = build_calibrator_features(q_om, q_pd, fusion_lambda)
    return load_calibrator(calibrator_path).predict_proba(features)


def thresholds_from_quantile(
    known_scores: np.ndarray,
    known_pred: np.ndarray,
    num_classes: int,
    quantile: float,
) -> list[float]:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    known_pred = np.asarray(known_pred, dtype=np.int64)
    thresholds = []
    for cls in range(num_classes):
        cls_scores = known_scores[known_pred == cls]
        if len(cls_scores) == 0:
            cls_scores = known_scores
        thresholds.append(
            float(np.quantile(cls_scores, float(quantile), method="lower"))
        )
    return thresholds


def choose_thresholds_for_known_accuracy(
    *,
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    known_scores: np.ndarray,
    num_classes: int,
    start_quantile: float,
    min_known_accuracy: float,
    quantile_grid: list[float],
) -> dict[str, Any]:
    candidates = sorted(
        {
            float(value)
            for value in quantile_grid
            if float(value) >= float(start_quantile)
        }
        | {float(start_quantile)}
    )
    best = None
    for quantile in candidates:
        thresholds = thresholds_from_quantile(
            known_scores,
            known_pred,
            num_classes,
            quantile,
        )
        y_pred = apply_unknown_rejection(
            np.asarray(known_pred, dtype=np.int64),
            np.asarray(known_scores, dtype=np.float64),
            num_classes,
            thresholds_per_class=thresholds,
        )
        accuracy = float(
            np.mean(y_pred == np.asarray(known_labels, dtype=np.int64))
        )
        result = {
            "quantile": quantile,
            "thresholds": thresholds,
            "known_accuracy": accuracy,
        }
        best = result
        if accuracy >= float(min_known_accuracy):
            return result
    if best is None:
        raise RuntimeError("No threshold quantile candidate was generated.")
    return best


def choose_classwise_thresholds_with_pseudo_guard(
    *,
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    known_scores: np.ndarray,
    pseudo_pred: np.ndarray,
    pseudo_scores: np.ndarray,
    num_classes: int,
    start_quantile: float,
    min_known_accuracy: float,
    quantile_grid: list[float],
) -> dict[str, Any]:
    known_labels = np.asarray(known_labels, dtype=np.int64)
    known_pred = np.asarray(known_pred, dtype=np.int64)
    known_scores = np.asarray(known_scores, dtype=np.float64)
    pseudo_pred = np.asarray(pseudo_pred, dtype=np.int64)
    pseudo_scores = np.asarray(pseudo_scores, dtype=np.float64)
    quantiles = sorted(
        {
            float(value)
            for value in quantile_grid
            if float(value) >= float(start_quantile)
        }
        | {float(start_quantile)}
    )
    target_correct = int(np.ceil(float(min_known_accuracy) * len(known_labels)))

    class_options: list[list[dict[str, Any]]] = []
    for cls in range(num_classes):
        cls_known_scores = known_scores[known_pred == cls]
        if len(cls_known_scores) == 0:
            cls_known_scores = known_scores
        options = []
        seen_thresholds = set()
        for quantile in quantiles:
            threshold = float(
                np.quantile(cls_known_scores, quantile, method="lower")
            )
            if threshold in seen_thresholds:
                continue
            seen_thresholds.add(threshold)
            correct = int(
                np.sum(
                    (known_labels == cls)
                    & (known_pred == cls)
                    & (known_scores <= threshold)
                )
            )
            pseudo_accepted = int(
                np.sum((pseudo_pred == cls) & (pseudo_scores <= threshold))
            )
            options.append(
                {
                    "threshold": threshold,
                    "quantile": quantile,
                    "correct": correct,
                    "pseudo_accepted": pseudo_accepted,
                }
            )
        class_options.append(options)

    states: dict[int, tuple[int, float, list[dict[str, Any]]]] = {
        0: (0, 0.0, [])
    }
    for options in class_options:
        next_states: dict[int, tuple[int, float, list[dict[str, Any]]]] = {}
        for current_correct, (current_cost, current_threshold_sum, chosen) in states.items():
            for option in options:
                new_correct = current_correct + int(option["correct"])
                candidate = (
                    current_cost + int(option["pseudo_accepted"]),
                    current_threshold_sum + float(option["threshold"]),
                    chosen + [option],
                )
                existing = next_states.get(new_correct)
                if existing is None or candidate[:2] < existing[:2]:
                    next_states[new_correct] = candidate
        states = next_states

    feasible = [
        (correct, state)
        for correct, state in states.items()
        if correct >= target_correct
    ]
    pool = feasible or list(states.items())
    correct, (pseudo_accepted, _, chosen) = min(
        pool,
        key=lambda item: (
            item[1][0],
            max(item[0] - target_correct, 0),
            item[1][1],
        ),
    )
    return {
        "thresholds": [float(option["threshold"]) for option in chosen],
        "quantiles_per_class": [float(option["quantile"]) for option in chosen],
        "known_accuracy": float(correct / max(len(known_labels), 1)),
        "pseudo_accepted": int(pseudo_accepted),
        "pseudo_recall": float(
            1.0 - pseudo_accepted / max(len(pseudo_scores), 1)
        ),
    }


def evaluate_calibrator_candidate(
    *,
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    known_scores: np.ndarray,
    heldout_pred: np.ndarray,
    heldout_scores: np.ndarray,
    thresholds: list[float],
    unknown_label: int,
    min_known_accuracy: float,
) -> dict[str, Any]:
    known_y_pred = apply_unknown_rejection(
        known_pred,
        known_scores,
        unknown_label,
        thresholds_per_class=thresholds,
    )
    heldout_y_pred = apply_unknown_rejection(
        heldout_pred,
        heldout_scores,
        unknown_label,
        thresholds_per_class=thresholds,
    )
    y_true = np.concatenate(
        [
            np.asarray(known_labels, dtype=np.int64),
            np.full(len(heldout_pred), unknown_label, dtype=np.int64),
        ]
    )
    y_pred = np.concatenate([known_y_pred, heldout_y_pred])
    all_scores = np.concatenate([known_scores, heldout_scores])
    metrics = evaluate_open_set(y_true, y_pred, all_scores, unknown_label)
    metrics["heldout_unknown_recall"] = float(
        np.mean(heldout_y_pred == unknown_label)
    )
    metrics["selection_score"] = float(
        0.45 * metrics["known_accuracy"]
        + 0.35 * metrics["heldout_unknown_recall"]
        + 0.15 * metrics["macro_f1"]
        + 0.05 * metrics["auroc"]
    )
    metrics["feasible"] = bool(
        metrics["known_accuracy"] >= float(min_known_accuracy)
    )
    return metrics
