from __future__ import annotations

from pathlib import Path

from functions.pipeline import checkpoint_path_for, format_metric_value, run_osr_pipeline
from settings import default_oracle_config


ROOT = Path(__file__).resolve().parent

# 是否重新预处理 Oracle 原始数据；数据已经处理好时保持 False。
RUN_DATA_PREPARE = False

# 是否重新训练闭集模型；平时调拒识阈值、OpenMax、细分策略时保持 False，避免覆盖已有 best_closed_set.pt。
RUN_TRAINING = False

# 是否在拒识结束后继续做未知类细分；只看开放集拒识时保持 False。
RUN_UNKNOWN_SUBDIVISION = True

# 指定已有模型权重路径；填 None 时默认使用输出目录里的 best_closed_set.pt。
CHECKPOINT_PATH = None

# 常改实验参数。
EXPERIMENT_NAME = "oracle_kri16_demod_known_first"
EPOCHS = 35
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# 常改拒识参数。
FUSION_LAMBDA_GRID = [0.35]
THRESHOLD_GRID = [0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.88]
MIN_KNOWN_ACCURACY = 0.91
CRITICAL_EDGE_COUNT = 50
ORDINARY_EDGE_RATIO = 0.15

# 常改未知类细分参数。
SUBDIVISION_FEATURE_MODE = "embedding_stats"
SUBDIVISION_PCA_DIM = 96
SUBDIVISION_K_MIN = 6
SUBDIVISION_K_MAX = 6
SUBDIVISION_CLUSTERING_BACKEND = "gmm_full_direct"
SUBDIVISION_TARGET_NUM_CLUSTERS = 6
SUBDIVISION_TARGET_K_STRENGTH = 1.0
SUBDIVISION_KNOWN_REJECT_MARGIN = -1.0
SUBDIVISION_OVERCLUSTER_EXTRA = 1
SUBDIVISION_OVERCLUSTER_CANDIDATES = [0, 1, 2, 3]
SUBDIVISION_M_SELECTION_MODE = "offline_min_gain"
SUBDIVISION_M_MIN_QUALITY_GAIN = 0.01
SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE = 0.10
SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE = 800


def build_config() -> dict:
    config = default_oracle_config(ROOT)
    config["project"]["name"] = EXPERIMENT_NAME
    config["project"]["output_dir"] = str((ROOT / "outputs" / EXPERIMENT_NAME).resolve())
    config["train"]["epochs"] = EPOCHS
    config["train"]["lr"] = LEARNING_RATE
    config["data"]["batch_size"] = BATCH_SIZE
    config["fusion"]["lambda_grid"] = FUSION_LAMBDA_GRID
    if config["fusion"].get("manual_thresholds_per_class") is not None and FUSION_LAMBDA_GRID:
        config["fusion"]["manual_fusion_lambda"] = float(FUSION_LAMBDA_GRID[0])
    config["fusion"]["threshold_grid"] = THRESHOLD_GRID
    config["fusion"]["min_known_accuracy"] = MIN_KNOWN_ACCURACY
    config["boundary"]["top_m"] = CRITICAL_EDGE_COUNT
    config["boundary"]["ordinary_edge_ratio"] = ORDINARY_EDGE_RATIO
    config["unknown_subdivision"]["feature_mode"] = SUBDIVISION_FEATURE_MODE
    config["unknown_subdivision"]["pca_dim"] = SUBDIVISION_PCA_DIM
    config["unknown_subdivision"]["k_min"] = SUBDIVISION_K_MIN
    config["unknown_subdivision"]["k_max"] = SUBDIVISION_K_MAX
    config["unknown_subdivision"]["clustering_backend"] = SUBDIVISION_CLUSTERING_BACKEND
    config["unknown_subdivision"]["target_num_clusters"] = SUBDIVISION_TARGET_NUM_CLUSTERS
    config["unknown_subdivision"]["target_k_strength"] = SUBDIVISION_TARGET_K_STRENGTH
    config["unknown_subdivision"]["known_reject_margin"] = SUBDIVISION_KNOWN_REJECT_MARGIN
    config["unknown_subdivision"]["overcluster_extra_clusters"] = SUBDIVISION_OVERCLUSTER_EXTRA
    config["unknown_subdivision"]["overcluster_extra_candidates"] = SUBDIVISION_OVERCLUSTER_CANDIDATES
    config["unknown_subdivision"]["m_selection_mode"] = SUBDIVISION_M_SELECTION_MODE
    config["unknown_subdivision"]["m_selection_min_quality_gain"] = SUBDIVISION_M_MIN_QUALITY_GAIN
    config["unknown_subdivision"]["direct_confidence_quantile"] = SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE
    config["unknown_subdivision"]["direct_min_cluster_size"] = SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE
    return config


def main() -> None:
    config = build_config()
    if not RUN_TRAINING and not checkpoint_path_for(config, CHECKPOINT_PATH).exists():
        raise FileNotFoundError("未找到已有闭集模型；如需从头训练，请把 RUN_TRAINING 改成 True。")
    metrics = run_osr_pipeline(
        config,
        skip_prepare=not RUN_DATA_PREPARE,
        skip_training=not RUN_TRAINING,
        ckpt_path=CHECKPOINT_PATH,
        with_unknown_subdivision=RUN_UNKNOWN_SUBDIVISION,
    )
    print("\nOracle 最终开放集指标")
    for key, value in metrics.items():
        print(f"{key}: {format_metric_value(value)}")


if __name__ == "__main__":
    main()
