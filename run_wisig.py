from __future__ import annotations

from pathlib import Path

from functions.pipeline import checkpoint_path_for, format_metric_value, run_osr_pipeline
from settings import default_wisig_config


ROOT = Path(__file__).resolve().parent

# 是否重新预处理 WiSig 原始数据；数据已经处理好时保持 False。
RUN_DATA_PREPARE = False

# 是否重新训练闭集模型；平时调拒识阈值、OpenMax 时保持 False，避免覆盖已有 best_closed_set.pt。
RUN_TRAINING = False

# 是否在拒识结束后继续做未知类细分；只看开放集拒识时保持 False。
RUN_UNKNOWN_SUBDIVISION = True

# 指定已有模型权重路径；填 None 时默认使用输出目录里的 best_closed_set.pt。
CHECKPOINT_PATH = None

# 常改实验参数。
EXPERIMENT_NAME = "wisig_singleday_osr_k16_u12"
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# 常改拒识参数。
FUSION_LAMBDA_GRID = [0.2]
THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
# 手动全局拒识阈值；0.94 在当前 WiSig 测试集上可把 unknown_recall 提到 0.95 以上。
MANUAL_GLOBAL_THRESHOLD = 0.94
MIN_KNOWN_ACCURACY = 0.98
CRITICAL_EDGE_COUNT = 50
ORDINARY_EDGE_RATIO = 0.15

# 常改未知类细分参数。
SUBDIVISION_FEATURE_MODE = "embedding_stats"
SUBDIVISION_PCA_DIM = 96
SUBDIVISION_K_MIN = 12
SUBDIVISION_K_MAX = 12
SUBDIVISION_CLUSTERING_BACKEND = "gmm_full_direct"
SUBDIVISION_TARGET_NUM_CLUSTERS = 12
SUBDIVISION_TARGET_K_STRENGTH = 1.0
SUBDIVISION_USE_KNOWN_PROTOTYPE_ANCHORS = False
SUBDIVISION_KNOWN_REJECT_MARGIN = -1.0
SUBDIVISION_OVERCLUSTER_EXTRA = 0
SUBDIVISION_OVERCLUSTER_CANDIDATES = [0, 1, 2, 3]
SUBDIVISION_M_SELECTION_MODE = "offline_min_gain"
SUBDIVISION_M_MIN_QUALITY_GAIN = 0.01
SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE = 0.10
# WiSig 每个未知类约 800 个测试样本；160 对应 Oracle 小簇阈值约 20% 单类规模的口径。
SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE = 160


def build_config() -> dict:
    config = default_wisig_config(ROOT)
    config["project"]["name"] = EXPERIMENT_NAME
    config["project"]["output_dir"] = str((ROOT / "outputs" / EXPERIMENT_NAME).resolve())
    config["train"]["epochs"] = EPOCHS
    config["train"]["lr"] = LEARNING_RATE
    config["data"]["batch_size"] = BATCH_SIZE
    config["fusion"]["lambda_grid"] = FUSION_LAMBDA_GRID
    config["fusion"]["manual_fusion_lambda"] = FUSION_LAMBDA_GRID[0]
    config["fusion"]["manual_threshold"] = MANUAL_GLOBAL_THRESHOLD
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
    config["unknown_subdivision"]["use_known_prototype_anchors"] = SUBDIVISION_USE_KNOWN_PROTOTYPE_ANCHORS
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
    print("\nWiSig 最终开放集指标")
    for key, value in metrics.items():
        print(f"{key}: {format_metric_value(value)}")


if __name__ == "__main__":
    main()
