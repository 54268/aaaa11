from __future__ import annotations

from run_oracle import build_config as build_oracle_config
from run_wisig import build_config as build_wisig_config


EXPECTED_LAMBDA_GRID = [index / 10 for index in range(11)]
EXPECTED_SELECTION_WEIGHTS = {
    "known_accuracy": 0.45,
    "unknown_recall": 0.35,
    "macro_f1": 0.15,
    "auroc": 0.05,
}


def _assert_shared_auto_search(config: dict) -> None:
    fusion = config["fusion"]

    assert fusion["lambda_grid"] == EXPECTED_LAMBDA_GRID
    assert fusion.get("manual_threshold") is None
    assert fusion.get("manual_thresholds_per_class") is None
    assert fusion["known_rescue"] == {"enabled": False}
    assert fusion["selection_weights"] == EXPECTED_SELECTION_WEIGHTS
    assert fusion["require_feasible"] is True


def test_oracle_uses_pseudo_unknown_auto_calibration() -> None:
    config = build_oracle_config()

    _assert_shared_auto_search(config)
    assert config["fusion"]["threshold_mode"] == "classwise_joint"
    assert config["fusion"]["score_calibration"] == "classwise_z"
    assert config["fusion"]["min_known_accuracy"] == 0.96


def test_oracle_defines_leakage_free_leave_class_out_calibration() -> None:
    config = build_oracle_config()
    leave_class_out = config["fusion"]["leave_class_out"]

    assert leave_class_out["enabled"] is True
    assert leave_class_out["num_folds"] == 5
    assert leave_class_out["base_seed"] == 42
    assert leave_class_out["pseudo_max_samples"] == 800
    assert leave_class_out["min_known_accuracy"] == 0.95
    assert leave_class_out["selection_weights"] == {
        "known_accuracy": 0.40,
        "heldout_unknown_recall": 0.35,
        "pseudo_unknown_recall": 0.10,
        "macro_f1": 0.10,
        "auroc": 0.05,
    }
    assert leave_class_out["uses_real_unknown_for_calibration"] is False


def test_oracle_defines_pseudo_unknown_supervised_calibrator() -> None:
    config = build_oracle_config()
    calibrator = config["fusion"]["supervised_calibrator"]

    assert calibrator["enabled"] is True
    assert calibrator["hidden_dim"] == 8
    assert calibrator["pseudo_scale_grid"] == [0.75, 1.0, 1.25, 1.5, 2.0]
    assert calibrator["lambda_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert calibrator["epoch_grid"] == [100, 250]
    assert calibrator["seed_grid"] == [42, 43, 44]
    assert calibrator["threshold_quantile_grid"] == [
        0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99
    ]
    assert calibrator["min_known_accuracy"] == 0.95
    assert calibrator["formal_known_accuracy_target"] == 0.93
    assert calibrator["uses_real_unknown_for_selection"] is False


def test_wisig_uses_pseudo_unknown_auto_calibration() -> None:
    config = build_wisig_config()

    _assert_shared_auto_search(config)
    assert config["fusion"]["threshold_mode"] == "global"
    assert config["fusion"].get("score_calibration", "none") == "none"
    assert config["fusion"]["min_known_accuracy"] == 0.985
