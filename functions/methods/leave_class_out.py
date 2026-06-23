from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from functions.common.metrics import evaluate_open_set
from functions.methods.fusion import (
    apply_score_calibration,
    fit_score_calibration,
    fuse_unknown_score,
)


@dataclass(frozen=True)
class LeaveClassOutFold:
    fold_index: int
    known_classes: tuple[int, ...]
    held_out_classes: tuple[int, ...]


def build_leave_class_out_folds(num_classes: int, num_folds: int) -> list[LeaveClassOutFold]:
    if num_classes <= 0 or num_folds <= 0:
        raise ValueError("num_classes and num_folds must be positive.")
    if num_classes % num_folds != 0:
        raise ValueError("num_classes must be divisible by num_folds.")

    classes = tuple(range(num_classes))
    held_out_per_fold = num_classes // num_folds
    folds = []
    for fold_index in range(num_folds):
        start = fold_index * held_out_per_fold
        held_out = classes[start : start + held_out_per_fold]
        known = tuple(cls for cls in classes if cls not in held_out)
        folds.append(
            LeaveClassOutFold(
                fold_index=fold_index,
                known_classes=known,
                held_out_classes=held_out,
            )
        )
    return folds


def subset_and_remap(
    x: np.ndarray,
    y: np.ndarray,
    included_classes: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    included = tuple(int(cls) for cls in included_classes)
    if not included:
        raise ValueError("included_classes must not be empty.")
    mask = np.isin(y, included)
    if not np.any(mask):
        raise ValueError("No samples match included_classes.")

    global_y = np.asarray(y[mask], dtype=np.int64)
    mapping = {global_class: local_class for local_class, global_class in enumerate(included)}
    local_y = np.asarray([mapping[int(label)] for label in global_y], dtype=np.int64)
    return np.asarray(x[mask]), local_y, global_y


def threshold_to_known_quantile(scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("scores must not be empty.")
    return float(np.mean(scores <= float(threshold)))


def threshold_from_known_quantile(scores: np.ndarray, quantile: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("scores must not be empty.")
    return float(np.quantile(scores, np.clip(float(quantile), 0.0, 1.0), method="lower"))


def aggregate_class_quantiles(
    rows: list[dict[str, Any]],
    num_classes: int,
) -> list[float]:
    result = []
    for global_class in range(num_classes):
        values = [
            float(row["quantile"])
            for row in rows
            if int(row["global_class"]) == global_class
        ]
        if not values:
            raise ValueError(f"No threshold quantiles found for class {global_class}.")
        result.append(float(np.median(values)))
    return result


def aggregate_transfer_quantiles(
    *,
    rows: list[dict[str, Any]],
    target_global_classes: Iterable[int],
    excluded_fold_index: int | None,
    mode: str,
) -> list[float]:
    usable = [
        row
        for row in rows
        if excluded_fold_index is None
        or int(row["fold_index"]) != int(excluded_fold_index)
    ]
    if not usable:
        raise ValueError("No quantile rows remain after fold exclusion.")

    if mode.endswith("lower_quartile"):
        reducer = lambda values: float(np.quantile(values, 0.25))
    elif mode.endswith("median"):
        reducer = lambda values: float(np.median(values))
    else:
        raise ValueError(f"Unsupported transfer aggregation mode: {mode}")

    global_mode = mode.startswith("global_")
    global_values = [float(row["quantile"]) for row in usable]
    result = []
    for global_class in target_global_classes:
        values = global_values if global_mode else [
            float(row["quantile"])
            for row in usable
            if int(row["global_class"]) == int(global_class)
        ]
        if not values:
            values = global_values
        result.append(reducer(values))
    return result


def restore_class_thresholds(
    *,
    known_scores: np.ndarray,
    known_pred: np.ndarray,
    class_quantiles: Iterable[float],
    num_classes: int,
) -> list[float]:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    known_pred = np.asarray(known_pred, dtype=np.int64)
    quantiles = [float(value) for value in class_quantiles]
    if len(quantiles) != num_classes:
        raise ValueError("class_quantiles length must equal num_classes.")
    if len(known_scores) != len(known_pred):
        raise ValueError("known_scores and known_pred must have equal length.")

    thresholds = []
    for cls in range(num_classes):
        cls_scores = known_scores[known_pred == cls]
        if len(cls_scores) == 0:
            cls_scores = known_scores
        thresholds.append(threshold_from_known_quantile(cls_scores, quantiles[cls]))
    return thresholds


def fold_class_coverage(
    folds: Iterable[LeaveClassOutFold],
    num_classes: int,
) -> dict[str, dict[str, int]]:
    held_out_counts = {str(cls): 0 for cls in range(num_classes)}
    known_counts = {str(cls): 0 for cls in range(num_classes)}
    for fold in folds:
        for cls in fold.held_out_classes:
            held_out_counts[str(cls)] += 1
        for cls in fold.known_classes:
            known_counts[str(cls)] += 1
    return {
        "held_out_counts": held_out_counts,
        "known_counts": known_counts,
    }


def balanced_pseudo_indices(
    source_labels: np.ndarray,
    pseudo_kind: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    source_labels = np.asarray(source_labels)
    pseudo_kind = np.asarray(pseudo_kind)
    if len(source_labels) != len(pseudo_kind):
        raise ValueError("source_labels and pseudo_kind must have equal length.")
    if max_samples <= 0 or len(source_labels) == 0:
        return np.empty(0, dtype=np.int64)
    if len(source_labels) <= max_samples:
        return np.arange(len(source_labels), dtype=np.int64)

    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, str], list[int]] = {}
    for index, (source, kind) in enumerate(zip(source_labels, pseudo_kind)):
        groups.setdefault((int(source), str(kind)), []).append(index)
    ordered_groups = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(indices)
        ordered_groups.append(indices.tolist())

    selected: list[int] = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for indices in ordered_groups:
            if offset < len(indices):
                selected.append(indices[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1
    return np.asarray(selected, dtype=np.int64)


def _reject(
    known_pred: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    unknown_label: int,
) -> np.ndarray:
    result = np.asarray(known_pred, dtype=np.int64).copy()
    result[np.asarray(scores) > thresholds[result]] = int(unknown_label)
    return result


def evaluate_leave_class_out_candidate(
    *,
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    heldout_pred: np.ndarray,
    pseudo_pred: np.ndarray,
    known_scores: np.ndarray,
    heldout_scores: np.ndarray,
    pseudo_scores: np.ndarray,
    thresholds: np.ndarray,
    unknown_label: int,
    selection_weights: dict[str, float],
) -> dict[str, float]:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    known_y_pred = _reject(known_pred, known_scores, thresholds, unknown_label)
    heldout_y_pred = _reject(heldout_pred, heldout_scores, thresholds, unknown_label)
    pseudo_y_pred = _reject(pseudo_pred, pseudo_scores, thresholds, unknown_label)

    unknown_count = len(heldout_pred) + len(pseudo_pred)
    y_true = np.concatenate(
        [
            np.asarray(known_labels, dtype=np.int64),
            np.full(unknown_count, unknown_label, dtype=np.int64),
        ]
    )
    y_pred = np.concatenate([known_y_pred, heldout_y_pred, pseudo_y_pred])
    all_scores = np.concatenate([known_scores, heldout_scores, pseudo_scores])
    metrics = evaluate_open_set(y_true, y_pred, all_scores, unknown_label)
    metrics["heldout_unknown_recall"] = (
        float(np.mean(heldout_y_pred == unknown_label)) if len(heldout_y_pred) else 0.0
    )
    metrics["pseudo_unknown_recall"] = (
        float(np.mean(pseudo_y_pred == unknown_label)) if len(pseudo_y_pred) else 0.0
    )
    metrics["selection_score"] = float(
        sum(float(weight) * float(metrics.get(key, 0.0)) for key, weight in selection_weights.items())
    )
    return metrics


def _candidate_thresholds(scores: np.ndarray, threshold_grid: Iterable[float]) -> list[float]:
    candidates = {float(value) for value in threshold_grid}
    if len(scores):
        for quantile in np.linspace(0.02, 0.995, 60):
            candidates.add(float(np.quantile(scores, quantile)))
        candidates.add(float(np.max(scores) + 1e-9))
    return sorted(candidates)


def search_leave_class_out_candidates(
    *,
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    known_q_om: np.ndarray,
    known_q_pd: np.ndarray,
    heldout_pred: np.ndarray,
    heldout_q_om: np.ndarray,
    heldout_q_pd: np.ndarray,
    pseudo_pred: np.ndarray,
    pseudo_q_om: np.ndarray,
    pseudo_q_pd: np.ndarray,
    num_classes: int,
    lambda_grid: Iterable[float],
    threshold_grid: Iterable[float],
    known_penalty_grid: Iterable[float],
    min_known_accuracy: float,
    selection_weights: dict[str, float],
    score_calibration_mode: str = "classwise_z",
    fusion_mode: str = "linear",
) -> list[dict[str, Any]]:
    known_labels = np.asarray(known_labels, dtype=np.int64)
    known_pred = np.asarray(known_pred, dtype=np.int64)
    heldout_pred = np.asarray(heldout_pred, dtype=np.int64)
    pseudo_pred = np.asarray(pseudo_pred, dtype=np.int64)

    total_known = max(len(known_labels), 1)
    total_heldout = max(len(heldout_pred), 1)
    total_pseudo = max(len(pseudo_pred), 1)
    known_weight = float(selection_weights.get("known_accuracy", 0.40))
    heldout_weight = float(selection_weights.get("heldout_unknown_recall", 0.35))
    pseudo_weight = float(selection_weights.get("pseudo_unknown_recall", 0.10))
    results = []

    for fusion_lambda in lambda_grid:
        known_raw = fuse_unknown_score(known_q_om, known_q_pd, fusion_lambda, mode=fusion_mode)
        heldout_raw = fuse_unknown_score(heldout_q_om, heldout_q_pd, fusion_lambda, mode=fusion_mode)
        pseudo_raw = fuse_unknown_score(pseudo_q_om, pseudo_q_pd, fusion_lambda, mode=fusion_mode)
        calibration = fit_score_calibration(
            known_raw,
            known_pred,
            num_classes,
            score_calibration_mode,
        )
        known_scores = apply_score_calibration(known_raw, known_pred, calibration)
        heldout_scores = apply_score_calibration(heldout_raw, heldout_pred, calibration)
        pseudo_scores = apply_score_calibration(pseudo_raw, pseudo_pred, calibration)

        best_feasible: dict[str, Any] | None = None
        best_relaxed: dict[str, Any] | None = None
        for penalty in known_penalty_grid:
            thresholds = []
            for cls in range(num_classes):
                cls_known_correct = (known_pred == cls) & (known_labels == cls)
                cls_heldout = heldout_pred == cls
                cls_pseudo = pseudo_pred == cls
                cls_scores = np.concatenate(
                    [
                        known_scores[known_pred == cls],
                        heldout_scores[cls_heldout],
                        pseudo_scores[cls_pseudo],
                    ]
                )
                best_class: tuple[float, float] | None = None
                for threshold in _candidate_thresholds(cls_scores, threshold_grid):
                    utility = (
                        (known_weight + float(penalty))
                        * float(np.sum(cls_known_correct & (known_scores <= threshold)))
                        / total_known
                        + heldout_weight
                        * float(np.sum(cls_heldout & (heldout_scores > threshold)))
                        / total_heldout
                        + pseudo_weight
                        * float(np.sum(cls_pseudo & (pseudo_scores > threshold)))
                        / total_pseudo
                    )
                    class_candidate = (utility, float(threshold))
                    if best_class is None or class_candidate > best_class:
                        best_class = class_candidate
                if best_class is None:
                    raise RuntimeError(f"No threshold candidate generated for class {cls}.")
                thresholds.append(best_class[1])

            metrics = evaluate_leave_class_out_candidate(
                known_labels=known_labels,
                known_pred=known_pred,
                heldout_pred=heldout_pred,
                pseudo_pred=pseudo_pred,
                known_scores=known_scores,
                heldout_scores=heldout_scores,
                pseudo_scores=pseudo_scores,
                thresholds=np.asarray(thresholds),
                unknown_label=num_classes,
                selection_weights=selection_weights,
            )
            quantiles = []
            for cls, threshold in enumerate(thresholds):
                cls_known_scores = known_scores[known_pred == cls]
                if len(cls_known_scores) == 0:
                    cls_known_scores = known_scores
                quantiles.append(threshold_to_known_quantile(cls_known_scores, threshold))
            candidate = {
                "fusion_lambda": float(fusion_lambda),
                "thresholds_per_class": [float(value) for value in thresholds],
                "threshold_quantiles_per_class": quantiles,
                "known_penalty": float(penalty),
                "score_calibration": calibration or {"mode": "none"},
                "metrics": metrics,
                "feasible": bool(metrics["known_accuracy"] >= float(min_known_accuracy)),
            }
            if best_relaxed is None or metrics["selection_score"] > best_relaxed["metrics"]["selection_score"]:
                best_relaxed = candidate
            if candidate["feasible"] and (
                best_feasible is None
                or metrics["selection_score"] > best_feasible["metrics"]["selection_score"]
            ):
                best_feasible = candidate

        chosen = best_feasible or best_relaxed
        if chosen is None:
            raise RuntimeError(f"No calibration candidate generated for lambda={fusion_lambda}.")
        results.append(chosen)
    return results


def choose_cross_fold_lambda(
    fold_candidate_lists: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    if not fold_candidate_lists:
        raise ValueError("fold_candidate_lists must not be empty.")

    lambdas = sorted(
        set.intersection(
            *[
                {float(candidate["fusion_lambda"]) for candidate in candidates}
                for candidates in fold_candidate_lists
            ]
        )
    )
    if not lambdas:
        raise ValueError("No fusion lambda is shared by every fold.")

    summaries = []
    for fusion_lambda in lambdas:
        selected = []
        for candidates in fold_candidate_lists:
            selected.append(
                next(
                    candidate
                    for candidate in candidates
                    if float(candidate["fusion_lambda"]) == fusion_lambda
                )
            )
        metric_keys = selected[0]["metrics"].keys()
        mean_metrics = {
            key: float(np.mean([candidate["metrics"][key] for candidate in selected]))
            for key in metric_keys
        }
        summaries.append(
            {
                "fusion_lambda": fusion_lambda,
                "metrics": mean_metrics,
                "all_folds_feasible": all(bool(candidate["feasible"]) for candidate in selected),
                "feasible_fold_count": int(sum(bool(candidate["feasible"]) for candidate in selected)),
                "fold_candidates": selected,
            }
        )

    feasible = [summary for summary in summaries if summary["all_folds_feasible"]]
    pool = feasible or summaries
    return max(
        pool,
        key=lambda item: (
            item["feasible_fold_count"],
            item["metrics"]["selection_score"],
            item["metrics"]["known_accuracy"],
            -item["fusion_lambda"],
        ),
    )
