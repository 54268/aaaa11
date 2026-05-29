from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def known_accuracy(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    mask = y_true != unknown_label
    if not np.any(mask):
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def unknown_recall(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    mask = y_true == unknown_label
    if not np.any(mask):
        return 0.0
    return float((y_pred[mask] == unknown_label).mean())


def unknown_precision(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    mask = y_pred == unknown_label
    if not np.any(mask):
        return 0.0
    return float((y_true[mask] == unknown_label).mean())


def fpr95(y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> float:
    target = (y_true == unknown_label).astype(np.int64)
    pos = unknown_score[target == 1]
    neg = unknown_score[target == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    threshold = np.quantile(pos, 0.05)
    return float((neg >= threshold).mean())


def auroc(y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> float:
    target = (y_true == unknown_label).astype(np.int64)
    if len(np.unique(target)) < 2:
        return 0.0
    return float(roc_auc_score(target, unknown_score))


def oscr_score(y_true: np.ndarray, y_pred: np.ndarray, known_confidence: np.ndarray, unknown_label: int) -> float:
    known_mask = y_true != unknown_label
    unknown_mask = y_true == unknown_label
    if not np.any(known_mask) or not np.any(unknown_mask):
        return 0.0

    known_correct = ((y_pred == y_true) & known_mask).astype(np.float64)
    unknown_accept = unknown_mask.astype(np.float64)
    order = np.argsort(-known_confidence)
    sorted_known_correct = known_correct[order]
    sorted_unknown_accept = unknown_accept[order]
    sorted_confidence = known_confidence[order]

    ccr = [0.0]
    fpr = [0.0]
    num_known = max(int(known_mask.sum()), 1)
    num_unknown = max(int(unknown_mask.sum()), 1)
    for threshold in sorted_confidence:
        accept = known_confidence >= threshold
        accept_ordered = accept[order]
        ccr.append(float(sorted_known_correct[accept_ordered].sum() / num_known))
        fpr.append(float(sorted_unknown_accept[accept_ordered].sum() / num_unknown))
    ccr.append(1.0)
    fpr.append(1.0)

    ccr_arr = np.asarray(ccr, dtype=np.float64)
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    idx = np.argsort(fpr_arr)
    return float(np.trapz(ccr_arr[idx], fpr_arr[idx]))


def evaluate_open_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
) -> Dict[str, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {unknown_label})
    known_confidence = 1.0 - unknown_score
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "known_accuracy": known_accuracy(y_true, y_pred, unknown_label),
        "unknown_precision": unknown_precision(y_true, y_pred, unknown_label),
        "unknown_recall": unknown_recall(y_true, y_pred, unknown_label),
        "known_fpr_as_unknown": float(((y_true != unknown_label) & (y_pred == unknown_label)).sum() / max((y_true != unknown_label).sum(), 1)),
        "unknown_false_accept_rate": float(((y_true == unknown_label) & (y_pred != unknown_label)).sum() / max((y_true == unknown_label).sum(), 1)),
        "auroc": auroc(y_true, unknown_score, unknown_label),
        "fpr95": fpr95(y_true, unknown_score, unknown_label),
        "oscr": oscr_score(y_true, y_pred, known_confidence, unknown_label),
    }


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        if not np.any(mask):
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += int(counts.max())
    return float(total / max(len(y_true), 1))


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_labels = {label: idx for idx, label in enumerate(np.unique(y_true))}
    pred_labels = {label: idx for idx, label in enumerate(np.unique(y_pred))}
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[true_labels[t], pred_labels[p]] += 1
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[row_ind, col_ind].sum() / max(len(y_true), 1))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "purity": purity_score(y_true, y_pred),
        "hungarian_accuracy": hungarian_accuracy(y_true, y_pred),
    }
