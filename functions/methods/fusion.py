from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from functions.common.metrics import evaluate_open_set


def prototype_distance_unknown_score(
    distances: np.ndarray,
    pred: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    d_min = distances[np.arange(len(pred)), pred]
    normalized = (d_min - mu[pred]) / (sigma[pred] + 1e-6)
    return 1.0 / (1.0 + np.exp(-normalized))


@dataclass
class FusionResult:
    fusion_lambda: float
    threshold: float | None
    thresholds_per_class: list[float] | None
    threshold_mode: str
    threshold_quantile: float | None
    metrics: Dict[str, float]


def fuse_unknown_score(
    q_om: np.ndarray,
    q_pd: np.ndarray,
    fusion_lambda: float,
    mode: str = "linear",
) -> np.ndarray:
    if mode == "linear":
        return fusion_lambda * q_om + (1.0 - fusion_lambda) * q_pd
    if mode == "geometric_mean":
        return np.sqrt(np.clip(q_om * q_pd, 1e-8, 1.0))
    if mode == "harmonic_mean":
        return 2.0 * q_om * q_pd / (q_om + q_pd + 1e-8)
    if mode == "geometric_blend":
        arithmetic = 0.5 * (q_om + q_pd)
        geometric = np.sqrt(np.clip(q_om * q_pd, 1e-8, 1.0))
        return fusion_lambda * arithmetic + (1.0 - fusion_lambda) * geometric
    raise ValueError(f"Unsupported fusion mode: {mode}")


def apply_unknown_rejection(
    known_pred: np.ndarray,
    q_u: np.ndarray,
    unknown_label: int,
    threshold: float | None = None,
    thresholds_per_class: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    y_pred = known_pred.copy()
    if thresholds_per_class is not None:
        thresholds = np.asarray(thresholds_per_class, dtype=np.float64)
        y_pred[q_u > thresholds[known_pred]] = unknown_label
        return y_pred
    if threshold is None:
        raise ValueError("Either threshold or thresholds_per_class must be provided.")
    y_pred[q_u > float(threshold)] = unknown_label
    return y_pred


def fusion_selection_score(metrics: Dict[str, float], selection_weights: Dict[str, float] | None = None) -> float:
    if selection_weights:
        score = 0.0
        for key, weight in selection_weights.items():
            score += float(weight) * float(metrics.get(key, 0.0))
        return score
    return float(metrics["macro_f1"] + 0.2 * metrics["auroc"])


def search_fusion_params(
    y_true: np.ndarray,
    known_pred: np.ndarray,
    q_om: np.ndarray,
    q_pd: np.ndarray,
    unknown_label: int,
    lambda_grid: Iterable[float],
    threshold_grid: Iterable[float],
    selection_weights: Dict[str, float] | None = None,
    known_quantile_floor: float | None = None,
    min_known_accuracy: float | None = None,
    threshold_mode: str = "global",
    classwise_quantile_grid: Iterable[float] | None = None,
    classwise_known_weight: float = 0.55,
    classwise_unknown_weight: float = 0.45,
    classwise_min_known_accept: float | None = None,
    fusion_mode: str = "linear",
) -> FusionResult:
    best_result = None
    best_relaxed = None
    known_mask = y_true != unknown_label
    for fusion_lambda in lambda_grid:
        q_u = fuse_unknown_score(q_om, q_pd, fusion_lambda, mode=fusion_mode)
        if threshold_mode == "classwise_quantile":
            quantiles = sorted({float(q) for q in (classwise_quantile_grid or [0.95])})
            if known_quantile_floor is not None:
                quantiles.append(float(known_quantile_floor))
                quantiles = sorted(set(quantiles))
            for quantile in quantiles:
                thresholds = []
                for cls in range(unknown_label):
                    cls_mask = known_mask & (known_pred == cls)
                    if np.any(cls_mask):
                        thresholds.append(float(np.quantile(q_u[cls_mask], quantile)))
                    else:
                        thresholds.append(float(np.quantile(q_u[known_mask], quantile)))
                y_pred = apply_unknown_rejection(
                    known_pred=known_pred,
                    q_u=q_u,
                    unknown_label=unknown_label,
                    thresholds_per_class=thresholds,
                )
                metrics = evaluate_open_set(y_true, y_pred, q_u, unknown_label)
                score = fusion_selection_score(metrics, selection_weights)
                candidate = (
                    score,
                    FusionResult(
                        fusion_lambda=float(fusion_lambda),
                        threshold=None,
                        thresholds_per_class=thresholds,
                        threshold_mode="classwise_quantile",
                        threshold_quantile=float(quantile),
                        metrics=metrics,
                    ),
                )
                if best_relaxed is None or score > best_relaxed[0]:
                    best_relaxed = candidate
                if min_known_accuracy is not None and float(metrics.get("known_accuracy", 0.0)) < min_known_accuracy:
                    continue
                if best_result is None or score > best_result[0]:
                    best_result = candidate
        elif threshold_mode == "classwise_balanced":
            for fusion_lambda in [fusion_lambda]:
                q_u = fuse_unknown_score(q_om, q_pd, fusion_lambda, mode=fusion_mode)
                thresholds = []
                for cls in range(unknown_label):
                    cls_known = q_u[known_mask & (known_pred == cls)]
                    cls_unknown = q_u[(~known_mask) & (known_pred == cls)]

                    if len(cls_known) == 0:
                        fallback = float(np.quantile(q_u[known_mask], known_quantile_floor or 0.95))
                        thresholds.append(fallback)
                        continue

                    candidates = set(float(t) for t in threshold_grid)
                    for quantile in np.linspace(0.80, 0.995, 24):
                        candidates.add(float(np.quantile(cls_known, quantile)))
                    if len(cls_unknown) > 0:
                        for quantile in np.linspace(0.05, 0.95, 19):
                            candidates.add(float(np.quantile(cls_unknown, quantile)))
                    candidates = sorted(candidates)

                    best_cls = None
                    relaxed_cls = None
                    for threshold in candidates:
                        keep_known = float((cls_known <= threshold).mean())
                        reject_unknown = float((cls_unknown > threshold).mean()) if len(cls_unknown) > 0 else 0.0
                        cls_score = classwise_known_weight * keep_known + classwise_unknown_weight * reject_unknown
                        candidate_cls = (cls_score, float(threshold))
                        if relaxed_cls is None or cls_score > relaxed_cls[0]:
                            relaxed_cls = candidate_cls
                        if classwise_min_known_accept is not None and keep_known < classwise_min_known_accept:
                            continue
                        if best_cls is None or cls_score > best_cls[0]:
                            best_cls = candidate_cls
                    thresholds.append((best_cls or relaxed_cls)[1])

                y_pred = apply_unknown_rejection(
                    known_pred=known_pred,
                    q_u=q_u,
                    unknown_label=unknown_label,
                    thresholds_per_class=thresholds,
                )
                metrics = evaluate_open_set(y_true, y_pred, q_u, unknown_label)
                score = fusion_selection_score(metrics, selection_weights)
                candidate = (
                    score,
                    FusionResult(
                        fusion_lambda=float(fusion_lambda),
                        threshold=None,
                        thresholds_per_class=thresholds,
                        threshold_mode="classwise_balanced",
                        threshold_quantile=None,
                        metrics=metrics,
                    ),
                )
                if best_relaxed is None or score > best_relaxed[0]:
                    best_relaxed = candidate
                if min_known_accuracy is not None and float(metrics.get("known_accuracy", 0.0)) < min_known_accuracy:
                    continue
                if best_result is None or score > best_result[0]:
                    best_result = candidate
        else:
            thresholds = sorted({float(threshold) for threshold in threshold_grid})
            if known_quantile_floor is not None and np.any(known_mask):
                thresholds.append(float(np.quantile(q_u[known_mask], known_quantile_floor)))
                thresholds = sorted(set(thresholds))
            for threshold in thresholds:
                y_pred = apply_unknown_rejection(
                    known_pred=known_pred,
                    q_u=q_u,
                    unknown_label=unknown_label,
                    threshold=float(threshold),
                )
                metrics = evaluate_open_set(y_true, y_pred, q_u, unknown_label)
                score = fusion_selection_score(metrics, selection_weights)
                candidate = (
                    score,
                    FusionResult(
                        fusion_lambda=float(fusion_lambda),
                        threshold=float(threshold),
                        thresholds_per_class=None,
                        threshold_mode="global",
                        threshold_quantile=None,
                        metrics=metrics,
                    ),
                )
                if best_relaxed is None or score > best_relaxed[0]:
                    best_relaxed = candidate
                if min_known_accuracy is not None and float(metrics.get("known_accuracy", 0.0)) < min_known_accuracy:
                    continue
                if best_result is None or score > best_result[0]:
                    best_result = candidate
    chosen = best_result if best_result is not None else best_relaxed
    return chosen[1]



