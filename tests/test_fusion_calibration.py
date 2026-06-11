from __future__ import annotations

import numpy as np

from functions.common.metrics import evaluate_open_set
from functions.methods.fusion import (
    apply_score_calibration,
    apply_unknown_rejection,
    fuse_unknown_score,
    search_fusion_params,
)


def test_search_fusion_params_applies_score_calibration() -> None:
    unknown_label = 2
    y_true = np.asarray([0, 0, 1, 1, unknown_label, unknown_label], dtype=np.int64)
    known_pred = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    raw_scores = np.asarray([0.10, 0.12, 0.80, 0.82, 0.50, 1.20], dtype=np.float64)

    result = search_fusion_params(
        y_true=y_true,
        known_pred=known_pred,
        q_om=raw_scores,
        q_pd=raw_scores,
        unknown_label=unknown_label,
        lambda_grid=[1.0],
        threshold_grid=[0.2, 0.6, 0.9],
        threshold_mode="global",
        score_calibration_mode="classwise_z",
    )

    assert result.score_calibration is not None
    assert result.score_calibration["mode"] == "classwise_z"
    assert result.metrics["overall_accuracy"] == 1.0


def test_classwise_balanced_thresholds_use_calibrated_score_space() -> None:
    unknown_label = 2
    y_true = np.asarray([0, 0, 1, 1, unknown_label, unknown_label], dtype=np.int64)
    known_pred = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    raw_scores = np.asarray([0.10, 0.12, 0.80, 0.82, 0.50, 1.20], dtype=np.float64)

    result = search_fusion_params(
        y_true=y_true,
        known_pred=known_pred,
        q_om=raw_scores,
        q_pd=raw_scores,
        unknown_label=unknown_label,
        lambda_grid=[1.0],
        threshold_grid=[0.2, 0.6, 0.9],
        threshold_mode="classwise_balanced",
        classwise_min_known_accept=1.0,
        score_calibration_mode="classwise_z",
    )

    q_u = fuse_unknown_score(raw_scores, raw_scores, result.fusion_lambda)
    q_u = apply_score_calibration(q_u, known_pred, result.score_calibration)
    y_pred = apply_unknown_rejection(
        known_pred=known_pred,
        q_u=q_u,
        unknown_label=unknown_label,
        thresholds_per_class=result.thresholds_per_class,
    )
    replayed_metrics = evaluate_open_set(y_true, y_pred, q_u, unknown_label)

    assert replayed_metrics["overall_accuracy"] == result.metrics["overall_accuracy"]
