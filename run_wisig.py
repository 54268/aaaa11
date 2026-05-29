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
FUSION_LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
MIN_KNOWN_ACCURACY = 0.99
CRITICAL_EDGE_COUNT = 50
ORDINARY_EDGE_RATIO = 0.15

# 常改未知类细分参数。
SUBDIVISION_FEATURE_MODE = "score_distance"
SUBDIVISION_PCA_DIM = 13
SUBDIVISION_K_MIN = 12
SUBDIVISION_K_MAX = 12
SUBDIVISION_CLUSTERING_BACKEND = "gmm"
SUBDIVISION_TARGET_NUM_CLUSTERS = 12
SUBDIVISION_TARGET_K_STRENGTH = 1.0


def build_config() -> dict:
    config = default_wisig_config(ROOT)
    config["project"]["name"] = EXPERIMENT_NAME
    config["project"]["output_dir"] = str((ROOT / "outputs" / EXPERIMENT_NAME).resolve())
    config["train"]["epochs"] = EPOCHS
    config["train"]["lr"] = LEARNING_RATE
    config["data"]["batch_size"] = BATCH_SIZE
    config["fusion"]["lambda_grid"] = FUSION_LAMBDA_GRID
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
