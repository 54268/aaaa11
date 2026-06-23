from __future__ import annotations

import numpy as np

from functions.methods.leave_class_out import LeaveClassOutFold
from run_oracle_leave_class_out import build_metric_comparison, materialize_fold_data


def test_materialize_fold_data_uses_only_train_and_validation_known(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train_x = np.arange(64, dtype=np.float32).reshape(8, 2, 4)
    train_y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    val_x = train_x + 100
    val_y = train_y.copy()
    np.savez_compressed(source / "train_known.npz", x=train_x, y=train_y)
    np.savez_compressed(source / "val_known.npz", x=val_x, y=val_y)
    np.savez_compressed(
        source / "test_unknown.npz",
        x=np.full((2, 2, 4), 999, dtype=np.float32),
        y=np.array([4, 5]),
    )
    target = tmp_path / "fold"
    fold = LeaveClassOutFold(
        fold_index=0,
        known_classes=(2, 3),
        held_out_classes=(0, 1),
    )

    manifest = materialize_fold_data(source, target, fold)

    train = np.load(target / "train_known.npz")
    val = np.load(target / "val_known.npz")
    simulated_unknown = np.load(target / "test_unknown.npz")
    assert train["y"].tolist() == [0, 0, 1, 1]
    assert val["y"].tolist() == [0, 0, 1, 1]
    assert simulated_unknown["y"].tolist() == [2, 2, 2, 2]
    assert not np.any(simulated_unknown["x"] == 999)
    assert manifest["uses_real_unknown_for_calibration"] is False


def test_build_metric_comparison_reports_percentage_point_changes() -> None:
    comparison = build_metric_comparison(
        old_manual={
            "known_accuracy": 0.96,
            "unknown_recall": 0.97,
            "macro_f1": 0.94,
            "auroc": 0.98,
        },
        current_auto={
            "known_accuracy": 0.95,
            "unknown_recall": 0.89,
            "macro_f1": 0.87,
            "auroc": 0.99,
        },
        leave_class_out={
            "known_accuracy": 0.955,
            "unknown_recall": 0.95,
            "macro_f1": 0.92,
            "auroc": 0.985,
        },
    )

    assert comparison["known_accuracy"]["vs_old_manual_pp"] == -0.5
    assert comparison["unknown_recall"]["vs_current_auto_pp"] == 6.0
    assert comparison["macro_f1"]["vs_current_auto_direction"] == "提高"
