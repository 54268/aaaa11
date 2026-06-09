from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


OPEN_SET_METRIC_KEYS = [
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "known_accuracy",
    "unknown_precision",
    "unknown_recall",
    "known_fpr_as_unknown",
    "unknown_false_accept_rate",
    "auroc",
    "fpr95",
    "oscr",
]

OPEN_SET_METADATA_KEYS = [
    "threshold_strategy_used",
    "threshold_mode",
    "threshold",
    "threshold_quantile",
    "number_of_tx",
    "number_of_rx_used",
    "rx_mode",
    "rx_used",
    "known_tx_list",
    "unknown_tx_list",
    "known_classes",
    "unknown_classes",
    "train_sample_count",
    "val_sample_count",
    "test_known_sample_count",
    "test_unknown_sample_count",
    "split_file",
    "output_dir",
    "best_val_acc",
    "best_epoch",
    "training_status",
]

SUBDIVISION_METRIC_KEYS = [
    "nmi",
    "ari",
    "purity",
    "hungarian_accuracy",
    "coverage_of_total_test_unknown",
]

SUBDIVISION_METADATA_KEYS = [
    "method_detail",
    "clustering_backend",
    "feature_mode",
    "resolved_num_clusters",
    "selected_unknown_cache_size",
    "uncertain_size",
    "uncertain_ratio",
    "cluster_size_min",
    "cluster_size_max",
    "cluster_size_mean",
    "nearest_known_proto_distance_mean",
    "nearest_known_proto_distance_min",
    "unknown_cache_precision",
    "unknown_cache_recall",
    "coverage_of_selected_true_unknown",
    "coverage_of_total_test_unknown",
    "suspected_known_noise_size",
    "num_evaluated_unknown",
    "num_true_unknown_classes",
    "num_predicted_clusters",
    "confidence_threshold",
    "known_classes",
    "unknown_classes",
    "train_sample_count",
    "val_sample_count",
    "test_known_sample_count",
    "test_unknown_sample_count",
    "split_file",
    "output_dir",
    "best_val_acc",
    "best_epoch",
    "training_status",
]

OPEN_SET_PER_SEED_FIELDS = ["dataset", "method", "task", "seed", *OPEN_SET_METRIC_KEYS, *OPEN_SET_METADATA_KEYS]
SUBDIVISION_PER_SEED_FIELDS = [
    "dataset",
    "method",
    "task",
    "seed",
    *SUBDIVISION_METRIC_KEYS,
    *SUBDIVISION_METADATA_KEYS,
]


def _summary_fields(metric_keys: list[str]) -> list[str]:
    return ["dataset", "method", "task", "seed", *metric_keys]


OPEN_SET_SUMMARY_FIELDS = _summary_fields(OPEN_SET_METRIC_KEYS)
SUBDIVISION_SUMMARY_FIELDS = _summary_fields(SUBDIVISION_METRIC_KEYS)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    if fieldnames is None:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def summarize(rows: Iterable[dict], metric_keys: list[str]) -> list[dict]:
    summary = []
    for row in rows:
        out = {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "task": row.get("task", ""),
            "seed": row.get("seed", ""),
        }
        for metric in metric_keys:
            out[metric] = row.get(metric, "")
        summary.append(out)
    return sorted(summary, key=lambda item: (str(item["dataset"]), str(item["method"]), str(item["seed"])))


def split_task_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    open_set_rows = [row for row in rows if row.get("task") == "open_set_rejection"]
    subdivision_rows = [row for row in rows if row.get("task") == "unknown_subdivision"]
    return open_set_rows, subdivision_rows


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_table(lines: list[str], rows: list[dict], headers: list[str]) -> None:
    if not rows:
        lines.append("本次没有生成该任务的可汇总结果。")
        return
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(key, "")) for key in headers) + " |")


def write_markdown_report(
    path: Path,
    *,
    open_set_summary: list[dict],
    subdivision_summary: list[dict],
    open_set_rows: list[dict],
    subdivision_rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 对比方法适配运行报告",
        "",
        "本报告由 `Comparison method/adapted_baselines/run_comparison.py` 生成。指标已经拆分为“开放集拒识”和“未知类细分”两张口径一致的表，指标名与本方法保持一致，不使用 mean/std 统计列。",
        "",
        "## 开放集拒识指标",
        "",
        "本表字段与本方法 `open_set_metrics.json` 的核心指标保持一致。",
        "",
    ]
    _append_table(lines, open_set_summary, OPEN_SET_SUMMARY_FIELDS)
    lines.extend(
        [
            "",
            "## 未知类细分指标",
            "",
            "本表字段与本方法 `unknown_subdivision_metrics.json` 的主要评价指标保持一致。",
            "",
        ]
    )
    _append_table(lines, subdivision_summary, SUBDIVISION_SUMMARY_FIELDS)

    lines.extend(
        [
            "",
            "## 逐次结果文件",
            "",
            "- 开放集拒识逐次指标：`open_set_per_seed_results.csv`",
            "- 开放集拒识主指标表：`open_set_summary_results.csv`",
            "- 未知类细分逐次指标：`unknown_subdivision_per_seed_results.csv`",
            "- 未知类细分主指标表：`unknown_subdivision_summary_results.csv`",
            "",
            "## 本次运行规模",
            "",
            f"- 开放集拒识逐次记录：{len(open_set_rows)}",
            f"- 未知类细分逐次记录：{len(subdivision_rows)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
