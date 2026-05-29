from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cluster purity: for each cluster, count its dominant true class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    total = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        if np.any(mask):
            _, counts = np.unique(y_true[mask], return_counts=True)
            total += int(counts.max())
    return float(total / len(y_true))


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best one-to-one cluster accuracy found with Hungarian assignment."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0

    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)
    true_map = {value: idx for idx, value in enumerate(true_ids)}
    pred_map = {value: idx for idx, value in enumerate(pred_ids)}
    matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[true_map[t], pred_map[p]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return float(matrix[row_ind, col_ind].sum() / len(y_true))


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "purity": purity_score(y_true, y_pred),
        "hungarian_accuracy": hungarian_accuracy(y_true, y_pred),
    }


def confusion_after_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_ids = np.unique(y_true)
    pred_ids = np.unique(y_pred)
    true_map = {value: idx for idx, value in enumerate(true_ids)}
    pred_map = {value: idx for idx, value in enumerate(pred_ids)}
    matrix = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[true_map[t], pred_map[p]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    ordered_cols = list(col_ind)
    remaining = [idx for idx in range(len(pred_ids)) if idx not in set(ordered_cols)]
    ordered_cols.extend(remaining)
    aligned = matrix[:, ordered_cols]
    return aligned, [str(x) for x in true_ids], [str(pred_ids[idx]) for idx in ordered_cols]

