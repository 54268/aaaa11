from __future__ import annotations

import numpy as np

from functions.methods.leave_class_out import (
    aggregate_class_quantiles,
    aggregate_transfer_quantiles,
    balanced_pseudo_indices,
    build_leave_class_out_folds,
    choose_cross_fold_lambda,
    evaluate_leave_class_out_candidate,
    fold_class_coverage,
    restore_class_thresholds,
    search_leave_class_out_candidates,
    subset_and_remap,
    threshold_from_known_quantile,
    threshold_to_known_quantile,
)


def test_build_leave_class_out_folds_covers_each_class_once() -> None:
    folds = build_leave_class_out_folds(num_classes=10, num_folds=5)

    assert [fold.held_out_classes for fold in folds] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]
    assert sorted(cls for fold in folds for cls in fold.held_out_classes) == list(range(10))
    for fold in folds:
        assert len(fold.known_classes) == 8
        assert set(fold.known_classes).isdisjoint(fold.held_out_classes)


def test_subset_and_remap_uses_contiguous_local_labels() -> None:
    x = np.arange(24).reshape(6, 1, 4)
    y = np.array([0, 1, 2, 3, 4, 5])

    subset_x, local_y, global_y = subset_and_remap(x, y, (1, 3, 5))

    assert subset_x[:, 0, 0].tolist() == [4, 12, 20]
    assert local_y.tolist() == [0, 1, 2]
    assert global_y.tolist() == [1, 3, 5]


def test_threshold_quantile_round_trip() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    quantile = threshold_to_known_quantile(scores, threshold=0.3)
    restored = threshold_from_known_quantile(scores, quantile)

    assert quantile == 0.75
    assert 0.29 <= restored <= 0.31


def test_aggregate_class_quantiles_uses_median() -> None:
    rows = [
        {"global_class": 0, "quantile": 0.94},
        {"global_class": 0, "quantile": 0.96},
        {"global_class": 0, "quantile": 0.95},
        {"global_class": 0, "quantile": 0.97},
    ]

    assert aggregate_class_quantiles(rows, num_classes=1) == [0.955]


def test_aggregate_transfer_quantiles_excludes_target_fold_and_supports_lower_quartile() -> None:
    rows = [
        {"fold_index": 0, "global_class": 0, "quantile": 0.99},
        {"fold_index": 1, "global_class": 0, "quantile": 0.80},
        {"fold_index": 2, "global_class": 0, "quantile": 0.90},
        {"fold_index": 3, "global_class": 0, "quantile": 1.00},
        {"fold_index": 1, "global_class": 1, "quantile": 0.70},
        {"fold_index": 2, "global_class": 1, "quantile": 0.80},
        {"fold_index": 3, "global_class": 1, "quantile": 0.90},
    ]

    result = aggregate_transfer_quantiles(
        rows=rows,
        target_global_classes=(0, 1),
        excluded_fold_index=0,
        mode="class_lower_quartile",
    )

    assert np.allclose(result, [0.85, 0.75])


def test_restore_class_thresholds_uses_each_predicted_class_distribution() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    predicted = np.array([0, 0, 0, 1, 1, 1])

    thresholds = restore_class_thresholds(
        known_scores=scores,
        known_pred=predicted,
        class_quantiles=[2 / 3, 2 / 3],
        num_classes=2,
    )

    assert thresholds == [0.2, 0.8]


def test_fold_class_coverage_reports_one_holdout_and_four_known_appearances() -> None:
    coverage = fold_class_coverage(build_leave_class_out_folds(10, 5), num_classes=10)

    assert coverage["held_out_counts"] == {str(cls): 1 for cls in range(10)}
    assert coverage["known_counts"] == {str(cls): 4 for cls in range(10)}


def test_balanced_pseudo_indices_are_deterministic_and_group_balanced() -> None:
    source_labels = np.array([0] * 8 + [1] * 2)
    pseudo_kind = np.array(["ordinary"] * 5 + ["critical"] * 3 + ["ordinary"] * 2)

    first = balanced_pseudo_indices(source_labels, pseudo_kind, max_samples=6, seed=42)
    second = balanced_pseudo_indices(source_labels, pseudo_kind, max_samples=6, seed=42)

    assert first.tolist() == second.tolist()
    assert len(first) == 6
    selected_groups = list(zip(source_labels[first].tolist(), pseudo_kind[first].tolist()))
    assert len(set(selected_groups)) == 3


def test_evaluate_candidate_reports_unknown_groups_separately() -> None:
    metrics = evaluate_leave_class_out_candidate(
        known_labels=np.array([0, 0, 1, 1]),
        known_pred=np.array([0, 0, 1, 1]),
        heldout_pred=np.array([0, 1]),
        pseudo_pred=np.array([0, 1]),
        known_scores=np.array([0.1, 0.7, 0.2, 0.3]),
        heldout_scores=np.array([0.8, 0.4]),
        pseudo_scores=np.array([0.9, 0.6]),
        thresholds=np.array([0.5, 0.5]),
        unknown_label=2,
        selection_weights={
            "known_accuracy": 0.40,
            "heldout_unknown_recall": 0.35,
            "pseudo_unknown_recall": 0.10,
            "macro_f1": 0.10,
            "auroc": 0.05,
        },
    )

    assert metrics["known_accuracy"] == 0.75
    assert metrics["heldout_unknown_recall"] == 0.5
    assert metrics["pseudo_unknown_recall"] == 1.0


def test_search_prefers_lambda_that_detects_heldout_unknowns() -> None:
    candidates = search_leave_class_out_candidates(
        known_labels=np.array([0, 0, 1, 1]),
        known_pred=np.array([0, 0, 1, 1]),
        known_q_om=np.array([0.1, 0.1, 0.1, 0.1]),
        known_q_pd=np.array([0.1, 0.1, 0.1, 0.1]),
        heldout_pred=np.array([0, 1]),
        heldout_q_om=np.array([0.9, 0.9]),
        heldout_q_pd=np.array([0.1, 0.1]),
        pseudo_pred=np.array([0, 1]),
        pseudo_q_om=np.array([0.8, 0.8]),
        pseudo_q_pd=np.array([0.1, 0.1]),
        num_classes=2,
        lambda_grid=[0.0, 1.0],
        threshold_grid=[0.3, 0.5, 0.7],
        known_penalty_grid=[0.0, 1.0],
        min_known_accuracy=1.0,
        selection_weights={
            "known_accuracy": 0.40,
            "heldout_unknown_recall": 0.35,
            "pseudo_unknown_recall": 0.10,
            "macro_f1": 0.10,
            "auroc": 0.05,
        },
        score_calibration_mode="none",
    )

    chosen = choose_cross_fold_lambda([candidates])

    assert chosen["fusion_lambda"] == 1.0
    assert chosen["metrics"]["heldout_unknown_recall"] == 1.0
