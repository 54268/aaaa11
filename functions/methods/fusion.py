from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

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
    score_calibration: dict[str, Any] | None = None


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


def fit_score_calibration(
    q_u: np.ndarray,
    known_pred: np.ndarray,
    num_classes: int,
    mode: str | None,
) -> dict[str, Any] | None:
    if mode in (None, "", "none"):
        return None
    if mode != "classwise_z":
        raise ValueError(f"Unsupported score calibration mode: {mode}")

    global_mean = float(np.mean(q_u))
    global_std = float(np.std(q_u) + 1e-6)
    means = []
    sigmas = []
    counts = []
    for cls in range(num_classes):
        cls_scores = q_u[known_pred == cls]
        counts.append(int(len(cls_scores)))
        if len(cls_scores) < 2:
            means.append(global_mean)
            sigmas.append(global_std)
            continue
        means.append(float(np.mean(cls_scores)))
        sigmas.append(float(np.std(cls_scores) + 1e-6))
    return {
        "mode": "classwise_z",
        "means": means,
        "sigmas": sigmas,
        "counts": counts,
        "fallback_mean": global_mean,
        "fallback_sigma": global_std,
    }


def apply_score_calibration(
    q_u: np.ndarray,
    known_pred: np.ndarray,
    calibration: dict[str, Any] | None,
) -> np.ndarray:
    if not calibration or calibration.get("mode") in (None, "", "none"):
        return q_u
    mode = str(calibration.get("mode"))
    if mode != "classwise_z":
        raise ValueError(f"Unsupported score calibration mode: {mode}")

    means = np.asarray(calibration["means"], dtype=np.float64)
    sigmas = np.asarray(calibration["sigmas"], dtype=np.float64)
    fallback_mean = float(calibration.get("fallback_mean", float(np.mean(q_u))))
    fallback_sigma = float(calibration.get("fallback_sigma", float(np.std(q_u) + 1e-6)))

    calibrated = np.empty_like(q_u, dtype=np.float64)
    valid = (known_pred >= 0) & (known_pred < len(means))
    z = np.empty_like(q_u, dtype=np.float64)
    z[valid] = (q_u[valid] - means[known_pred[valid]]) / (sigmas[known_pred[valid]] + 1e-6)
    z[~valid] = (q_u[~valid] - fallback_mean) / (fallback_sigma + 1e-6)
    z = np.clip(z, -50.0, 50.0)
    calibrated[:] = 1.0 / (1.0 + np.exp(-z))
    return calibrated


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


def apply_known_rescue(
    y_pred: np.ndarray,
    known_pred: np.ndarray,
    q_u: np.ndarray,
    distances: np.ndarray,
    unknown_label: int,
    rescue_config: dict[str, Any] | None,
) -> np.ndarray:
    if not rescue_config or not bool(rescue_config.get("enabled", False)):
        return y_pred
    rules = rescue_config.get("rules_per_class") or []
    if not rules:
        return y_pred

    sorted_distances = np.sort(distances, axis=1)
    d_min = sorted_distances[:, 0]
    if sorted_distances.shape[1] > 1:
        d_margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    else:
        d_margin = np.zeros_like(d_min)

    rescued = y_pred.copy()
    rejected_mask = y_pred == unknown_label
    for rule in rules:
        cls = int(rule["class_index"])
        max_score = float(rule.get("max_score", np.inf))
        max_distance = float(rule.get("max_distance", np.inf))
        min_margin = float(rule.get("min_margin", -np.inf))
        mask = (
            rejected_mask
            & (known_pred == cls)
            & (q_u <= max_score)
            & (d_min <= max_distance)
            & (d_margin >= min_margin)
        )
        rescued[mask] = known_pred[mask]
    return rescued


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
    score_calibration_mode: str | None = None,
) -> FusionResult:
    best_result = None
    best_relaxed = None
    known_mask = y_true != unknown_label
    for fusion_lambda in lambda_grid:
        q_u_raw = fuse_unknown_score(q_om, q_pd, fusion_lambda, mode=fusion_mode)
        score_calibration = None
        if np.any(known_mask):
            score_calibration = fit_score_calibration(
                q_u_raw[known_mask],
                known_pred[known_mask],
                unknown_label,
                score_calibration_mode,
            )
        q_u = apply_score_calibration(q_u_raw, known_pred, score_calibration)
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
                        score_calibration=score_calibration,
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
                        score_calibration=score_calibration,
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
                        score_calibration=score_calibration,
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



