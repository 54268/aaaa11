from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from functions.common.io import ensure_dir


def macro_f1_with_unknown(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {unknown_label})
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


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


def known_fpr_as_unknown(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    known_mask = y_true != unknown_label
    if not np.any(known_mask):
        return 0.0
    return float((y_pred[known_mask] == unknown_label).mean())


def unknown_false_accept_rate(y_true: np.ndarray, y_pred: np.ndarray, unknown_label: int) -> float:
    unknown_mask = y_true == unknown_label
    if not np.any(unknown_mask):
        return 0.0
    return float((y_pred[unknown_mask] != unknown_label).mean())


def auroc_known_vs_unknown(y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> float:
    binary_target = (y_true == unknown_label).astype(np.int32)
    if len(np.unique(binary_target)) < 2:
        return 0.0
    return float(roc_auc_score(binary_target, unknown_score))


def fpr95_known_vs_unknown(y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> float:
    binary_target = (y_true == unknown_label).astype(np.int32)
    pos_scores = unknown_score[binary_target == 1]
    neg_scores = unknown_score[binary_target == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    threshold = np.quantile(pos_scores, 0.05)
    return float((neg_scores >= threshold).mean())


def oscr_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    known_confidence: np.ndarray,
    unknown_label: int,
) -> float:
    known_mask = y_true != unknown_label
    unknown_mask = y_true == unknown_label
    if not np.any(known_mask) or not np.any(unknown_mask):
        return 0.0

    known_correct = ((y_pred == y_true) & known_mask).astype(np.float64)
    unknown_accept = unknown_mask.astype(np.float64)

    order = np.argsort(-known_confidence)
    sorted_score = known_confidence[order]
    sorted_known_correct = known_correct[order]
    sorted_unknown_accept = unknown_accept[order]

    ccr = []
    fpr = []
    num_known = known_mask.sum()
    num_unknown = unknown_mask.sum()
    for threshold in sorted_score:
        accept = known_confidence >= threshold
        ccr.append(float(sorted_known_correct[accept[order]].sum() / max(num_known, 1)))
        fpr.append(float(sorted_unknown_accept[accept[order]].sum() / max(num_unknown, 1)))

    ccr = np.asarray([0.0] + ccr + [1.0])
    fpr = np.asarray([0.0] + fpr + [1.0])
    order = np.argsort(fpr)
    return float(np.trapz(ccr[order], fpr[order]))


def evaluate_open_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
) -> Dict[str, float]:
    known_confidence = 1.0 - unknown_score
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {unknown_label})
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": macro_f1_with_unknown(y_true, y_pred, unknown_label),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "known_accuracy": known_accuracy(y_true, y_pred, unknown_label),
        "unknown_precision": unknown_precision(y_true, y_pred, unknown_label),
        "unknown_recall": unknown_recall(y_true, y_pred, unknown_label),
        "known_fpr_as_unknown": known_fpr_as_unknown(y_true, y_pred, unknown_label),
        "unknown_false_accept_rate": unknown_false_accept_rate(y_true, y_pred, unknown_label),
        "auroc": auroc_known_vs_unknown(y_true, unknown_score, unknown_label),
        "fpr95": fpr95_known_vs_unknown(y_true, unknown_score, unknown_label),
        "oscr": oscr_score(y_true, y_pred, known_confidence, unknown_label),
    }


def save_confusion_matrix(path: str | Path, y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(path, matrix, delimiter=",", fmt="%d")


def save_prediction_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    q_om: np.ndarray,
    q_pd: np.ndarray,
    d_min: np.ndarray,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "unknown_score", "q_om", "q_pd", "d_min"])
        for row in zip(y_true, y_pred, unknown_score, q_om, q_pd, d_min):
            writer.writerow(row)



