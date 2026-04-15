from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

from .io import ensure_dir

METRIC_CN = {
    "overall_accuracy": "总体准确率",
    "known_accuracy": "已知类准确率",
    "macro_f1": "宏平均F1",
    "auroc": "AUROC",
    "fpr95": "FPR95",
    "unknown_recall": "未知类召回率",
    "weighted_f1": "加权F1",
    "unknown_precision": "未知类精确率",
}

CORE_METRICS = [
    "overall_accuracy",
    "known_accuracy",
    "macro_f1",
    "auroc",
    "fpr95",
    "unknown_recall",
]

DATASET_LABELS = {
    "wisig": "WiSig",
    "oracle": "Oracle",
}


def infer_dataset_key(dataset_name: str, output_dir: str | Path | None = None) -> str:
    joined = f"{dataset_name} {output_dir or ''}".lower()
    for key in DATASET_LABELS:
        if key in joined:
            return key
    slug = re.sub(r"[^a-z0-9]+", "_", joined).strip("_")
    return slug or "experiment"


def dataset_summary_path(root: str | Path, dataset_name: str, output_dir: str | Path | None = None) -> Path:
    key = infer_dataset_key(dataset_name, output_dir)
    return Path(root) / f"RESULT_SUMMARY_{key.upper()}.md"


def write_final_report(
    path: str | Path,
    metrics: Dict[str, float],
    config_path: str,
    output_dir: str,
    dataset_name: str,
    extra_notes: list[str] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    notes = extra_notes or []
    dataset_key = infer_dataset_key(dataset_name, output_dir)
    dataset_label = DATASET_LABELS.get(dataset_key, dataset_name)

    lines = [
        f"# {dataset_label} 开放集 SEI 结果汇总",
        "",
        f"- 数据集：`{dataset_name}`",
        f"- 配置文件：`{config_path}`",
        f"- 输出目录：`{output_dir}`",
        "",
        "## 核心指标",
        "",
        "| 指标键 | 中文名 | 数值 |",
        "| --- | --- | ---: |",
    ]

    for key in CORE_METRICS:
        if key in metrics:
            lines.append(f"| {key} | {METRIC_CN.get(key, key)} | {metrics[key]:.6f} |")

    optional = [key for key in ["weighted_f1", "unknown_precision"] if key in metrics]
    if optional:
        lines.extend(["", "## 补充指标", "", "| 指标键 | 中文名 | 数值 |", "| --- | --- | ---: |"])
        for key in optional:
            lines.append(f"| {key} | {METRIC_CN.get(key, key)} | {metrics[key]:.6f} |")

    lines.extend(
        [
            "",
            "## 原始结果文件",
            "",
            "- `open_set_metrics.json`：完整指标结果",
            "- `confusion_matrix.csv`：混淆矩阵原始数值",
            "- `open_set_predictions.csv`：逐样本预测结果",
        ]
    )

    if notes:
        lines.extend(["", "## 备注", ""])
        lines.extend([f"- {note}" for note in notes])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_index(
    path: str | Path,
    entries: Iterable[dict[str, str]],
    latest_dataset: str,
    latest_output_dir: str,
    latest_config_path: str,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    normalized_entries = list(entries)
    lines = [
        "# 开放集 SEI 结果总览",
        "",
        "这个文件只作为总入口，不再保存某一次实验的单独结果。",
        "",
        "## 最近一次运行",
        "",
        f"- 数据集：`{latest_dataset}`",
        f"- 配置文件：`{latest_config_path}`",
        f"- 输出目录：`{latest_output_dir}`",
        "",
        "## 数据集汇总入口",
        "",
    ]

    for entry in normalized_entries:
        lines.append(f"- {entry['label']}：`{entry['path']}`")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `RESULT_SUMMARY_WISIG.md`：WiSig 最近一次运行结果汇总",
            "- `RESULT_SUMMARY_ORACLE.md`：Oracle 最近一次运行结果汇总",
            "- `outputs/<实验名>/final_report.md`：某次具体实验的独立汇总",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
