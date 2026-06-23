from __future__ import annotations

import numpy as np

from functions.methods.supervised_calibrator import (
    apply_supervised_calibrator_score,
    build_calibrator_features,
    choose_thresholds_for_known_accuracy,
    choose_classwise_thresholds_with_pseudo_guard,
    evaluate_calibrator_candidate,
    load_calibrator,
    save_calibrator,
    thresholds_from_quantile,
    train_calibrator,
)


def test_build_calibrator_features_preserves_linear_fusion() -> None:
    q_om = np.array([0.2, 0.8])
    q_pd = np.array([0.6, 0.4])

    features = build_calibrator_features(q_om, q_pd, fusion_lambda=0.25)

    assert features.shape == (2, 3)
    assert np.allclose(features[:, 2], 0.25 * q_om + 0.75 * q_pd)


def test_training_learns_separated_pseudo_unknowns_and_round_trips(tmp_path) -> None:
    rng = np.random.default_rng(42)
    known = rng.normal(0.10, 0.02, size=(100, 3)).astype(np.float32)
    pseudo = rng.normal(0.90, 0.02, size=(100, 3)).astype(np.float32)

    result = train_calibrator(known, pseudo, seed=42, epochs=150, lr=0.02)

    assert result.model.predict_proba(pseudo).mean() > 0.9
    assert result.model.predict_proba(known).mean() < 0.1
    path = tmp_path / "calibrator.pt"
    save_calibrator(path, result)
    restored = load_calibrator(path)
    assert np.allclose(
        restored.predict_proba(pseudo),
        result.model.predict_proba(pseudo),
    )


def test_classwise_thresholds_keep_requested_known_fraction() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    pred = np.array([0, 0, 1, 1])

    thresholds = thresholds_from_quantile(
        scores,
        pred,
        num_classes=2,
        quantile=0.5,
    )

    assert thresholds == [0.1, 0.3]


def test_candidate_evaluation_reports_feasibility_from_measured_accuracy() -> None:
    metrics = evaluate_calibrator_candidate(
        known_labels=np.array([0, 0, 1, 1]),
        known_pred=np.array([0, 0, 1, 1]),
        known_scores=np.array([0.1, 0.2, 0.3, 0.9]),
        heldout_pred=np.array([0, 1]),
        heldout_scores=np.array([0.8, 0.8]),
        thresholds=[0.5, 0.5],
        unknown_label=2,
        min_known_accuracy=0.95,
    )

    assert metrics["known_accuracy"] == 0.75
    assert metrics["heldout_unknown_recall"] == 1.0
    assert metrics["feasible"] is False


def test_apply_supervised_calibrator_score_uses_saved_model(tmp_path) -> None:
    known = np.tile([0.1, 0.1, 0.1], (50, 1)).astype(np.float32)
    pseudo = np.tile([0.9, 0.9, 0.9], (50, 1)).astype(np.float32)
    result = train_calibrator(known, pseudo, seed=42, epochs=100, lr=0.02)
    path = tmp_path / "calibrator.pt"
    save_calibrator(path, result)

    scores = apply_supervised_calibrator_score(
        q_om=np.array([0.1, 0.9]),
        q_pd=np.array([0.1, 0.9]),
        fusion_lambda=0.5,
        calibrator_path=path,
    )

    assert scores[0] < 0.1
    assert scores[1] > 0.9


def test_choose_thresholds_raises_quantile_until_known_accuracy_is_safe() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = labels.copy()
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3, 0.9])

    result = choose_thresholds_for_known_accuracy(
        known_labels=labels,
        known_pred=pred,
        known_scores=scores,
        num_classes=2,
        start_quantile=0.5,
        min_known_accuracy=0.75,
        quantile_grid=[0.5, 0.75, 1.0],
    )

    assert result["quantile"] == 0.75
    assert result["known_accuracy"] == 0.75


def test_classwise_safety_relaxes_class_with_lower_pseudo_cost() -> None:
    known_labels = np.array([0, 0, 1, 1])
    known_pred = known_labels.copy()
    known_scores = np.array([0.1, 0.8, 0.1, 0.8])
    pseudo_pred = np.array([0, 0, 1, 1])
    pseudo_scores = np.array([0.2, 0.3, 0.9, 0.95])

    result = choose_classwise_thresholds_with_pseudo_guard(
        known_labels=known_labels,
        known_pred=known_pred,
        known_scores=known_scores,
        pseudo_pred=pseudo_pred,
        pseudo_scores=pseudo_scores,
        num_classes=2,
        start_quantile=0.5,
        min_known_accuracy=0.75,
        quantile_grid=[0.5, 1.0],
    )

    assert result["known_accuracy"] == 0.75
    assert result["thresholds"][0] == 0.1
    assert result["thresholds"][1] == 0.8
