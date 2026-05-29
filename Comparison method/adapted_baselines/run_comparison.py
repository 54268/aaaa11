from __future__ import annotations

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(SRC_DIR))

# 避免 Windows + MKL 下 KMeans/Joblib 线程警告刷屏。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.data_io import DatasetProtocol, load_protocol, protocol_summary
from src.models import HydraClassifier, HyperRSIClassifier, OpenRFIStyleClassifier
from src.reporting import (
    OPEN_SET_METRIC_KEYS,
    OPEN_SET_PER_SEED_FIELDS,
    OPEN_SET_SUMMARY_FIELDS,
    SUBDIVISION_METRIC_KEYS,
    SUBDIVISION_PER_SEED_FIELDS,
    SUBDIVISION_SUMMARY_FIELDS,
    save_csv,
    save_json,
    split_task_rows,
    summarize,
    write_markdown_report,
)
from src.trainers import choose_device, run_open_set_baseline, run_openrfi_subdivision


# =========================
# 常用开关：平时主要改这里
# =========================

# 运行模式：
# "smoke" 只抽少量样本、跑 1 个 epoch，用来检查流程是否能跑通；
# "formal" 使用完整数据和更多 epoch，用来生成论文对比结果。
RUN_MODE = "formal"

# 要运行的数据集。可选："wisig", "oracle"。
DATASETS_TO_RUN = ["wisig", "oracle"]

# 要运行的方法。可选："hyperrsi", "hydra", "openrfi"。
METHODS_TO_RUN = ["hyperrsi", "hydra", "openrfi"]

# 随机种子。正式实验建议用多个种子；smoke 模式保持一个种子即可。
SEEDS = [42]

# 拒识阈值使用验证集已知类置信度的低分位数。
# 0.05 表示大约允许 5% 验证已知类被拒识，用于不使用真实未知类调阈值。
KNOWN_REJECT_QUANTILE = 0.05


# =========================
# 训练参数：需要看性能时再改
# =========================

SMOKE_EPOCHS = 1
FORMAL_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "auto"

# smoke 抽样数量。formal 模式会自动设为 None，使用完整数据。
SMOKE_TRAIN_SAMPLES_PER_CLASS = 24
SMOKE_EVAL_SAMPLES_PER_CLASS = 24

# OpenRFI 风格原型分组数量。正式实验可以试 50、80、100。
OPENRFI_NUM_PROTOTYPES = 50

# 如果对应 checkpoint 已经存在，优先加载已有权重重新评估并整理指标；
# 设为 False 时会重新训练并覆盖该方法本次 seed 的 checkpoint。
REUSE_EXISTING_CHECKPOINTS = True


COMPARISON_ROOT = CURRENT_DIR.parents[0]
PROJECT_ROOT = CURRENT_DIR.parents[1]
RESULT_ROOT = COMPARISON_ROOT / "adapted_results"


def _ensure_inside_comparison(path: Path) -> None:
    resolved = path.resolve()
    root = COMPARISON_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"禁止写入 Comparison method 之外的路径: {resolved}") from exc


def _protocols() -> dict[str, DatasetProtocol]:
    return {
        "wisig": DatasetProtocol(
            name="wisig_singleday_osr_k16_u12",
            processed_root=PROJECT_ROOT / "data" / "processed" / "wisig_singleday_osr_k16_u12",
            normalize="per_sample",
        ),
        "oracle": DatasetProtocol(
            name="oracle_kri16_demod",
            processed_root=PROJECT_ROOT / "data" / "processed" / "oracle_kri16_demod",
            normalize="per_sample",
        ),
    }


def main() -> None:
    _ensure_inside_comparison(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    if RUN_MODE not in {"smoke", "formal"}:
        raise ValueError('RUN_MODE 只能是 "smoke" 或 "formal"。')
    epochs = SMOKE_EPOCHS if RUN_MODE == "smoke" else FORMAL_EPOCHS
    max_train = SMOKE_TRAIN_SAMPLES_PER_CLASS if RUN_MODE == "smoke" else None
    max_eval = SMOKE_EVAL_SAMPLES_PER_CLASS if RUN_MODE == "smoke" else None

    device = choose_device(DEVICE)
    print(f"当前设备: {device}")
    print(f"运行模式: {RUN_MODE}")
    print(f"所有结果将写入: {RESULT_ROOT}")

    protocol_map = _protocols()
    loaded = [load_protocol(protocol_map[name]) for name in DATASETS_TO_RUN]
    save_json(RESULT_ROOT / "dataset_protocol_summary.json", protocol_summary(loaded))

    rows: list[dict] = []
    for protocol in loaded:
        print(f"\n=== 数据集: {protocol.name} ===")
        for seed in SEEDS:
            if "hyperrsi" in METHODS_TO_RUN:
                model = HyperRSIClassifier(num_classes=protocol.num_known_classes)
                rows.append(
                    run_open_set_baseline(
                        method_name="HyperRSI",
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / "hyperrsi",
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_eval_per_class=max_eval,
                        score_mode="center_cosine",
                        known_quantile=KNOWN_REJECT_QUANTILE,
                        use_margin_labels=True,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

            if "hydra" in METHODS_TO_RUN:
                model = HydraClassifier(
                    num_classes=protocol.num_known_classes,
                    signal_length=protocol.signal_length,
                )
                rows.append(
                    run_open_set_baseline(
                        method_name="HyDRA",
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / "hydra",
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_eval_per_class=max_eval,
                        score_mode="softmax",
                        known_quantile=KNOWN_REJECT_QUANTILE,
                        use_margin_labels=False,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

            if "openrfi" in METHODS_TO_RUN:
                model = OpenRFIStyleClassifier(
                    num_classes=protocol.num_known_classes,
                    signal_length=protocol.signal_length,
                )
                rows.append(
                    run_openrfi_subdivision(
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / "openrfi",
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_unknown_per_class=max_eval,
                        n_prototypes=OPENRFI_NUM_PROTOTYPES,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

    open_set_rows, subdivision_rows = split_task_rows(rows)
    open_set_summary = summarize(open_set_rows, OPEN_SET_METRIC_KEYS)
    subdivision_summary = summarize(subdivision_rows, SUBDIVISION_METRIC_KEYS)

    result_dir = RESULT_ROOT / RUN_MODE
    save_csv(result_dir / "open_set_per_seed_results.csv", open_set_rows, OPEN_SET_PER_SEED_FIELDS)
    save_csv(result_dir / "open_set_summary_results.csv", open_set_summary, OPEN_SET_SUMMARY_FIELDS)
    save_csv(
        result_dir / "unknown_subdivision_per_seed_results.csv",
        subdivision_rows,
        SUBDIVISION_PER_SEED_FIELDS,
    )
    save_csv(
        result_dir / "unknown_subdivision_summary_results.csv",
        subdivision_summary,
        SUBDIVISION_SUMMARY_FIELDS,
    )
    write_markdown_report(
        result_dir / "comparison_report.md",
        open_set_summary=open_set_summary,
        subdivision_summary=subdivision_summary,
        open_set_rows=open_set_rows,
        subdivision_rows=subdivision_rows,
    )

    print("\n=== 对比方法汇总 ===")
    for row in open_set_summary + subdivision_summary:
        print(row)
    print(f"\n报告路径: {RESULT_ROOT / RUN_MODE / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
