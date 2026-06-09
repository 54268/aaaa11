from __future__ import annotations

import os
import sys
import csv
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(SRC_DIR))

# 避免 Windows + MKL 下 KMeans/Joblib 线程警告刷屏。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.data_io import DatasetProtocol, load_protocol, protocol_summary
from src.models import (
    ARPLClassifier,
    ARPLConfusingSampleDiscriminator1D,
    ARPLConfusingSampleGenerator1D,
    CompactHyperRSIClassifier,
    HydraClassifier,
    OpenRFIStyleClassifier,
    SoftmaxCNNClassifier,
)
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
from src.trainers import choose_device, run_open_set_baseline


# =========================
# 常用开关：平时主要改这里
# =========================

# 运行模式：
# "smoke" 只抽少量样本、跑 1 个 epoch，用来检查流程是否能跑通；
# "formal" 使用完整数据和更多 epoch，用来生成论文对比结果。
RUN_MODE = "formal"

# 要运行的数据集。可选："wisig", "oracle"。
DATASETS_TO_RUN = ["wisig", "oracle"]

# 要运行的方法。可选："softmax", "openmax", "hyperrsi", "hydra", "openrfi", "arpl"。
METHODS_TO_RUN = ["softmax", "openmax", "hyperrsi", "hydra", "openrfi", "arpl"]

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

# OpenRFI 风格原型分组数量。Oracle 和 WiSig 的最优分组粒度不同，因此按数据集单独设置。
OPENRFI_NUM_PROTOTYPES_BY_DATASET = {
    "oracle_kri16_demod": 150,
    "wisig_singleday_osr_k16_u12": 300,
}
OPENRFI_SUBDIVISION_BACKEND_BY_DATASET = {
    "oracle_kri16_demod": "openrfi_world_kplusm",
    "wisig_singleday_osr_k16_u12": "openrfi_prototype_grouping",
}
OPENRFI_WORLD_SUBDIVISION_N_PROTOTYPES_BY_DATASET = {
    "oracle_kri16_demod": 75,
}
OPENRFI_WORLD_SUBDIVISION_NEIGHBORS_BY_DATASET = {
    "oracle_kri16_demod": 1,
}
OPENRFI_WORLD_SUBDIVISION_CONFIDENCE_THRESHOLD_BY_DATASET = {
    "oracle_kri16_demod": 0.9966560805154376,
}

# OpenRFI 的原始思想偏表示学习和原型分组，不同数据集上的 softmax 置信度稳定性差异较大。
# Oracle 上 softmax unknown score 接近失效，因此使用 OpenMax 校准；WiSig 受控协议下 softmax 已足够稳定。
OPENRFI_SCORE_MODE_BY_DATASET = {
    "oracle_kri16_demod": "openmax",
    "wisig_singleday_osr_k16_u12": "softmax",
}

# 各方法的拒识分位数按验证集已知类校准，不使用测试未知标签调阈。
# OpenMax 在 Oracle 上取 0.20 是 known accuracy 与 unknown recall 的折中；
# OpenRFI 补齐表示学习模块后取 0.10，可同时保住已知类和未知类召回。
SOFTMAX_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.05,
    "wisig_singleday_osr_k16_u12": 0.05,
}
OPENMAX_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.20,
    "wisig_singleday_osr_k16_u12": 0.10,
}
HYPERRSI_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.10,
    "wisig_singleday_osr_k16_u12": 0.10,
}
HYDRA_SCORE_MODE_BY_DATASET = {
    "oracle_kri16_demod": "center_cosine",
    "wisig_singleday_osr_k16_u12": "center_cosine",
}
HYDRA_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.10,
    "wisig_singleday_osr_k16_u12": 0.10,
}
OPENRFI_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.10,
    "wisig_singleday_osr_k16_u12": 0.10,
}
ARPL_KNOWN_REJECT_QUANTILE_BY_DATASET = {
    "oracle_kri16_demod": 0.10,
    "wisig_singleday_osr_k16_u12": 0.10,
}

# ARPL 保留 reciprocal-point 训练；正式表使用 EVT/OpenMax 校准 ARPL logits 做拒识。
# confusing samples 分支已实现，但在当前 1D I/Q 任务上会明显牺牲已知类准确率，因此不作为默认正式结果。
ARPL_EMBEDDING_DIM = 64
ARPL_SCORE_MODE = "openmax"
ARPL_USE_CONFUSING_SAMPLES = False
ARPL_CONFUSING_NOISE_DIM = 64
ARPL_CONFUSING_BETA = 0.1
ARPL_CONFUSING_GAN_LR = 2e-4

# 如果对应 checkpoint 已经存在，优先加载已有权重重新评估并整理指标；
# 设为 False 时会重新训练并覆盖该方法本次 seed 的 checkpoint。
REUSE_EXISTING_CHECKPOINTS = True


COMPARISON_ROOT = CURRENT_DIR.parents[0]
PROJECT_ROOT = CURRENT_DIR.parents[1]
RESULT_ROOT = COMPARISON_ROOT / "adapted_results"

METHOD_ORDER = ["Softmax", "OpenMax", "HyperRSI", "HyDRA", "OpenRFI", "ARPL", "PCBM (ours)"]
FINAL_OPEN_SET_FIELDS = [
    "method",
    "overall_accuracy",
    "known_accuracy",
    "unknown_precision",
    "unknown_recall",
    "macro_f1",
    "auroc",
    "fpr95",
    "oscr",
]
FINAL_SUBDIVISION_FIELDS = [
    "method",
    "nmi",
    "ari",
    "purity",
    "hungarian_accuracy",
    "coverage_of_total_test_unknown",
    "unknown_cache_precision",
    "unknown_cache_recall",
    "resolved_num_clusters",
]

ROOT_FINAL_OPEN_SET_FIELDS = [
    "method",
    "overall_accuracy",
    "known_accuracy",
    "macro_f1",
    "auroc",
]

ROOT_FINAL_SUBDIVISION_FIELDS = [
    "method",
    "nmi",
    "ari",
    "purity",
    "coverage_of_total_test_unknown",
]

PCBM_OUTPUTS = {
    "wisig_singleday_osr_k16_u12": PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12",
    "oracle_kri16_demod": PROJECT_ROOT / "outputs" / "oracle_kri16_demod_known_first",
}


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.6f}"
        except ValueError:
            return value
    return str(value)


def _write_table_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_markdown_table(lines: list[str], rows: list[dict], headers: list[str]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field, "")) for field in headers) + " |")


def write_final_comparison_tables(
    result_dir: Path,
    open_set_summary: list[dict],
    subdivision_summary: list[dict],
) -> None:
    table_dir = result_dir / "final_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    open_map = {(row.get("dataset"), row.get("method")): row for row in open_set_summary}
    sub_map = {(row.get("dataset"), row.get("method")): row for row in subdivision_summary}

    all_lines = [
        "# 最终对比表",
        "",
        "本文件将开放集拒识和未知类细分拆开汇总。Softmax、OpenMax、HyperRSI 和 HyDRA 原生不具备未知类细分模块，因此只进入拒识对比表；未知类细分表只保留具备明确细分后处理的 OpenRFI 和本文 PCBM。",
        "",
    ]
    root_lines = [
        "# 最终对比表",
        "",
        "本文件为项目根目录下的精简版最终汇总，只保留对比方法最核心的拒识与细分指标。",
        "开放集拒识只保留 `overall_accuracy`、`known_accuracy`、`macro_f1`、`auroc`；未知类细分保留 `nmi`、`ari`、`purity` 与覆盖率 `coverage_of_total_test_unknown`。",
        "",
    ]
    # 只汇总本次实际运行的数据集，避免旧结果混入正式表。
    for dataset_name in sorted({row.get("dataset") for row in open_set_summary if row.get("dataset")}):
        open_rows = []
        for method in METHOD_ORDER[:-1]:
            open_row = open_map.get((dataset_name, method), {})
            merged = {"method": method}
            for field in FINAL_OPEN_SET_FIELDS[1:]:
                merged[field] = open_row.get(field, "")
            open_rows.append(merged)

        pcbm_root = PCBM_OUTPUTS.get(str(dataset_name))
        sub_rows = []
        openrfi_sub = sub_map.get((dataset_name, "OpenRFI"), {})
        if openrfi_sub:
            row = {"method": "OpenRFI"}
            for field in FINAL_SUBDIVISION_FIELDS[1:]:
                row[field] = openrfi_sub.get(field, "")
            sub_rows.append(row)

        if pcbm_root is not None:
            open_metrics = _load_json(pcbm_root / "open_set_metrics.json")
            sub_metrics = _load_json(pcbm_root / "unknown_subdivision" / "unknown_subdivision_metrics.json")
            open_pcbm_row = {"method": "PCBM (ours)"}
            for field in FINAL_OPEN_SET_FIELDS[1:]:
                open_pcbm_row[field] = open_metrics.get(field, "")
            open_rows.append(open_pcbm_row)

            sub_pcbm_row = {"method": "PCBM (ours)"}
            for field in FINAL_SUBDIVISION_FIELDS[1:]:
                sub_pcbm_row[field] = sub_metrics.get(field, "")
            sub_rows.append(sub_pcbm_row)

        _write_table_csv(table_dir / f"{dataset_name}_open_set_rejection_table.csv", open_rows, FINAL_OPEN_SET_FIELDS)
        _write_table_csv(table_dir / f"{dataset_name}_unknown_subdivision_table.csv", sub_rows, FINAL_SUBDIVISION_FIELDS)

        lines = [f"# {dataset_name} 开放集拒识对比表", ""]
        _append_markdown_table(lines, open_rows, FINAL_OPEN_SET_FIELDS)
        (table_dir / f"{dataset_name}_open_set_rejection_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        lines = [f"# {dataset_name} 未知类细分对比表", ""]
        _append_markdown_table(lines, sub_rows, FINAL_SUBDIVISION_FIELDS)
        (table_dir / f"{dataset_name}_unknown_subdivision_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        all_lines.extend([f"## {dataset_name} 开放集拒识", ""])
        _append_markdown_table(all_lines, open_rows, FINAL_OPEN_SET_FIELDS)
        all_lines.extend(["", f"## {dataset_name} 未知类细分", ""])
        _append_markdown_table(all_lines, sub_rows, FINAL_SUBDIVISION_FIELDS)
        all_lines.append("")

        root_lines.extend([f"## {dataset_name} 开放集拒识", ""])
        root_open_rows = []
        for row in open_rows:
            root_open_rows.append({key: row.get(key, "") for key in ROOT_FINAL_OPEN_SET_FIELDS})
        _append_markdown_table(root_lines, root_open_rows, ROOT_FINAL_OPEN_SET_FIELDS)
        root_lines.extend(["", f"## {dataset_name} 未知类细分", ""])
        root_sub_rows = []
        for row in sub_rows:
            root_sub_rows.append({key: row.get(key, "") for key in ROOT_FINAL_SUBDIVISION_FIELDS})
        _append_markdown_table(root_lines, root_sub_rows, ROOT_FINAL_SUBDIVISION_FIELDS)
        root_lines.append("")

    (table_dir / "final_comparison_tables.md").write_text("\n".join(all_lines), encoding="utf-8")
    (PROJECT_ROOT / "final_comparison_tables.md").write_text("\n".join(root_lines), encoding="utf-8")


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
            if "softmax" in METHODS_TO_RUN:
                model = SoftmaxCNNClassifier(num_classes=protocol.num_known_classes)
                rows.extend(
                    run_open_set_baseline(
                        method_name="Softmax",
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / "softmax",
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_eval_per_class=max_eval,
                        score_mode="softmax",
                        known_quantile=SOFTMAX_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=False,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

            if "openmax" in METHODS_TO_RUN:
                model = SoftmaxCNNClassifier(num_classes=protocol.num_known_classes)
                rows.extend(
                    run_open_set_baseline(
                        method_name="OpenMax",
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / "openmax",
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_eval_per_class=max_eval,
                        score_mode="openmax",
                        known_quantile=OPENMAX_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=False,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                        openmax_backend="native",
                        openmax_distance_type="eucl",
                    )
                )

            if "hyperrsi" in METHODS_TO_RUN:
                model = CompactHyperRSIClassifier(num_classes=protocol.num_known_classes)
                rows.extend(
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
                        known_quantile=HYPERRSI_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=True,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

            if "hydra" in METHODS_TO_RUN:
                model = HydraClassifier(
                    num_classes=protocol.num_known_classes,
                    signal_length=protocol.signal_length,
                )
                rows.extend(
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
                        score_mode=HYDRA_SCORE_MODE_BY_DATASET.get(protocol.name, "softmax"),
                        known_quantile=HYDRA_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=False,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                    )
                )

            if "openrfi" in METHODS_TO_RUN:
                model = OpenRFIStyleClassifier(
                    num_classes=protocol.num_known_classes,
                    signal_length=protocol.signal_length,
                    num_prototypes=OPENRFI_NUM_PROTOTYPES_BY_DATASET.get(protocol.name, 50),
                )
                rows.extend(
                    run_open_set_baseline(
                        method_name="OpenRFI",
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
                        max_eval_per_class=max_eval,
                        score_mode=OPENRFI_SCORE_MODE_BY_DATASET.get(protocol.name, "softmax"),
                        known_quantile=OPENRFI_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=True,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                        subdivision_backend=OPENRFI_SUBDIVISION_BACKEND_BY_DATASET.get(protocol.name, "openrfi_prototype_grouping"),
                        subdivision_n_prototypes=OPENRFI_WORLD_SUBDIVISION_N_PROTOTYPES_BY_DATASET.get(
                            protocol.name,
                            OPENRFI_NUM_PROTOTYPES_BY_DATASET.get(protocol.name, 50),
                        ),
                        subdivision_graph_neighbors=OPENRFI_WORLD_SUBDIVISION_NEIGHBORS_BY_DATASET.get(protocol.name, 3),
                        subdivision_graph_lambda=1.0,
                        subdivision_confidence_threshold=OPENRFI_WORLD_SUBDIVISION_CONFIDENCE_THRESHOLD_BY_DATASET.get(protocol.name),
                        n_prototypes=OPENRFI_NUM_PROTOTYPES_BY_DATASET.get(protocol.name, 50),
                    )
                )

            if "arpl" in METHODS_TO_RUN:
                model = ARPLClassifier(
                    num_classes=protocol.num_known_classes,
                    embedding_dim=ARPL_EMBEDDING_DIM,
                )
                generator = (
                    ARPLConfusingSampleGenerator1D(
                        noise_dim=ARPL_CONFUSING_NOISE_DIM,
                        signal_length=protocol.signal_length,
                    )
                    if ARPL_USE_CONFUSING_SAMPLES
                    else None
                )
                discriminator = (
                    ARPLConfusingSampleDiscriminator1D(signal_length=protocol.signal_length)
                    if ARPL_USE_CONFUSING_SAMPLES
                    else None
                )
                rows.extend(
                    run_open_set_baseline(
                        method_name="ARPL",
                        model=model,
                        protocol=protocol,
                        normalize="per_sample",
                        device=device,
                        output_dir=RESULT_ROOT / RUN_MODE / protocol.name / ("arpl_cs" if ARPL_USE_CONFUSING_SAMPLES else "arpl_evt"),
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        seed=seed,
                        max_train_per_class=max_train,
                        max_eval_per_class=max_eval,
                        score_mode=ARPL_SCORE_MODE,
                        known_quantile=ARPL_KNOWN_REJECT_QUANTILE_BY_DATASET.get(protocol.name, KNOWN_REJECT_QUANTILE),
                        use_margin_labels=False,
                        reuse_checkpoint=REUSE_EXISTING_CHECKPOINTS,
                        confusing_sample_generator=generator,
                        confusing_sample_discriminator=discriminator,
                        confusing_noise_dim=ARPL_CONFUSING_NOISE_DIM,
                        confusing_beta=ARPL_CONFUSING_BETA,
                        confusing_gan_lr=ARPL_CONFUSING_GAN_LR,
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
    write_final_comparison_tables(result_dir, open_set_summary, subdivision_summary)

    print("\n=== 对比方法汇总 ===")
    for row in open_set_summary + subdivision_summary:
        print(row)
    print(f"\n报告路径: {RESULT_ROOT / RUN_MODE / 'comparison_report.md'}")
    print(f"最终对比表路径: {PROJECT_ROOT / 'final_comparison_tables.md'}")


if __name__ == "__main__":
    main()
