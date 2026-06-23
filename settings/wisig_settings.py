from __future__ import annotations

from pathlib import Path


def p(root: Path, relative_path: str) -> str:
    return str((root / relative_path).resolve())


def default_wisig_config(root: Path) -> dict:
    return {
        "_config_path": str(root / "run_wisig.py"),
        "_project_root": str(root),
        "project": {
            "name": "wisig_singleday_osr_k16_u12",
            "output_dir": p(root, "outputs/wisig_singleday_osr_k16_u12"),
        },
        "prep": {
            "kind": "wisig_compact",
            "raw_path": p(root, "data/raw/wisig/SingleDay.pkl"),
            "split_file": p(root, "data/splits/wisig/single_day_rx1_eq0/wisig_single_day_rx1_eq0_k16_u12_seed42.json"),
            "processed_root": p(root, "data/processed/wisig_singleday_osr_k16_u12"),
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "deduplicate_exact": False,
        },
        "data": {
            "mode": "separate_npz",
            "root": p(root, "data/processed/wisig_singleday_osr_k16_u12"),
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
            "beta": 1.0,
            "alpha": 0.5,
            "top_m": 50,
            "ordinary_edge_ratio": 0.15,
        },
        "pseudo_unknown": {
            "ordinary_eta": 1.0,
            "critical_eta": 1.0,
            "critical_beta": 0.7,
            "ordinary_variations": 1,
            "critical_variations": 2,
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
            "lambda_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "score_calibration": "none",
            "manual_threshold": None,
            "manual_thresholds_per_class": None,
            "known_rescue": {"enabled": False},
            "threshold_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99],
            "threshold_mode": "global",
            "selection_weights": {
                "known_accuracy": 0.45,
                "unknown_recall": 0.35,
                "macro_f1": 0.15,
                "auroc": 0.05,
            },
            "known_quantile_floor": 0.95,
            "min_known_accuracy": 0.985,
            "require_feasible": True,
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
            "method": "embedding_stats_gmm_full_direct",
            "output_subdir": "unknown_subdivision",
            "feature_mode": "embedding_stats",
            "pca_dim": 96,
            "k_min": 12,
            "k_max": 12,
            "clustering_backend": "gmm_full_direct",
            "target_num_clusters": 12,
            "target_k_strength": 1.0,
            "auto_sample_size": 3000,
            "use_known_prototype_anchors": False,
            "known_anchor_fixed": True,
            "suspected_known_policy": "uncertain",
            "assignment_margin": 0.0,
            "known_reject_margin": -1.0,
            "overcluster_extra_clusters": 0,
            "overcluster_extra_candidates": [0, 1, 2, 3],
            "m_selection_mode": "offline_min_gain",
            "m_selection_min_quality_gain": 0.01,
            "direct_confidence_quantile": 0.10,
            "direct_min_cluster_size": 160,
            "n_init": 30,
            "agg_sample_size": 8000,
            "uncertain_penalty": 0.15,
            "stability_weight": 0.30,
            "db_weight": 0.25,
            "ch_weight": 0.30,
        },
    }
