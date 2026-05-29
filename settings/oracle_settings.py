from __future__ import annotations

from pathlib import Path


def p(root: Path, relative_path: str) -> str:
    return str((root / relative_path).resolve())


def default_oracle_config(root: Path) -> dict:
    return {
        "_config_path": str(root / "run_oracle.py"),
        "_project_root": str(root),
        "project": {
            "name": "oracle_osr_main",
            "output_dir": p(root, "outputs/oracle_kri16_demod_known_first"),
        },
        "prep": {
            "kind": "oracle_sigmf",
            "raw_root": p(root, "data/raw/oracle/KRI-16IQImbalances-DemodulatedData"),
            "split_file": p(root, "data/splits/oracle/oracle_main_sorted_k10_u6.json"),
            "processed_root": p(root, "data/processed/oracle_kri16_demod"),
        },
        "data": {
            "mode": "separate_npz",
            "root": p(root, "data/processed/oracle_kri16_demod"),
            "batch_size": 128,
            "num_workers": 0,
            "signal_length": 256,
            "normalize": "per_sample",
        },
        "model": {
            "backbone": "cvcnn_iq",
            "embedding_dim": 128,
            "hidden_dim": 64,
            "dropout": 0.15,
            "temperature": 1.0,
        },
        "train": {
            "seed": 42,
            "device": "auto",
            "epochs": 20,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "prototype_momentum": 0.9,
        },
        "loss": {
            "lambda_basic": 1.0,
            "lambda_angle": 0.15,
            "lambda_prototype": 0.1,
            "angle_margin": 0.15,
        },
        "boundary": {
            "k": 5,
            "alpha": 0.5,
            "top_m": 50,
            "ordinary_edge_ratio": 0.15,
        },
        "pseudo_unknown": {
            "ordinary_eta": 1.0,
            "critical_eta": 1.15,
            "critical_beta": 0.7,
            "ordinary_variations": 2,
            "critical_variations": 3,
            "jitter": 0.1,
        },
        "openmax": {
            "backend": "repo_openmax",
            "alpha_rank": 3,
            "tail_size": 25,
            "distance_type": "eucl",
            "euclid_weight": 1.0,
        },
        "fusion": {
            "mode": "linear",
            "lambda_grid": [0.35],
            "manual_fusion_lambda": 0.35,
            "manual_thresholds_per_class": [
                0.9379194630872483,
                0.8998657718120805,
                0.9300097823143005,
                0.9062080536912751,
                0.9125503355704697,
                0.8744966442953019,
                0.8935234899328859,
                0.6357381489066517,
                0.3958303617830037,
                0.3330835997357088,
            ],
            "threshold_grid": [0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.88],
            "threshold_mode": "manual_classwise",
            "classwise_quantile_grid": [0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
            "classwise_known_weight": 0.45,
            "classwise_unknown_weight": 0.55,
            "classwise_min_known_accept": 0.88,
            "selection_weights": {
                "known_accuracy": 0.45,
                "unknown_recall": 0.35,
                "macro_f1": 0.15,
                "auroc": 0.05,
            },
            "known_quantile_floor": 0.95,
            "min_known_accuracy": 0.91,
        },
        "eval": {
            "save_predictions": True,
        },
        "reporting": {
            "write_root_summaries": False,
            "write_figures_inside_output_dir": True,
        },
        "unknown_subdivision": {
            "enabled": True,
            "method": "gmm_full_direct_with_known_anchor_guard",
            "output_subdir": "unknown_subdivision",
            "feature_mode": "embedding",
            "pca_dim": 32,
            "k_min": 5,
            "k_max": 5,
            "clustering_backend": "gmm_full_direct",
            "target_num_clusters": 5,
            "target_k_strength": 1.0,
            "auto_sample_size": 3000,
            "use_known_prototype_anchors": True,
            "known_anchor_fixed": True,
            "suspected_known_policy": "uncertain",
            "assignment_margin": 0.0,
            "known_reject_margin": -0.05,
            "n_init": 30,
            "agg_sample_size": 8000,
            "uncertain_penalty": 0.15,
            "stability_weight": 0.30,
            "db_weight": 0.25,
            "ch_weight": 0.30,
        },
    }
