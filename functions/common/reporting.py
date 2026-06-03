from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

from functions.common.io import ensure_dir

METRIC_CN = {
    "overall_accuracy": "总体准确率",
    "macro_precision": "宏平均精确率",
    "macro_recall": "宏平均召回率",
    "known_accuracy": "已知类准确率",
    "macro_f1": "宏平均 F1",
    "auroc": "已知/未知区分 AUROC",
    "fpr95": "95% 未知召回下的已知误拒率",
    "unknown_recall": "未知类召回率",
    "weighted_f1": "加权 F1",
    "unknown_precision": "未知类精确率",
    "known_fpr_as_unknown": "已知类误拒率",
    "unknown_false_accept_rate": "未知类误接收率",
    "oscr": "开放集分类识别曲线面积",
}

METRIC_DESC = {
    "overall_accuracy": "已知类分对且未知类拒识正确的总比例，越高越好。",
    "macro_precision": "每个类别 precision 的平均值，越高越好。",
    "macro_recall": "每个类别 recall 的平均值，越高越好。",
    "known_accuracy": "只看真实已知类样本，被分到正确已知类别的比例，越高越好。",
    "macro_f1": "每个类别 F1 的平均值，更关注类别均衡表现，越高越好。",
    "auroc": "unknown score 区分已知与未知的整体能力，越接近 1 越好。",
    "fpr95": "未知召回约 95% 时，已知样本被误拒为 unknown 的比例，越低越好。",
    "unknown_recall": "真实未知类中被拒识为 unknown 的比例，越高越好。",
    "weighted_f1": "按类别样本数加权后的 F1，越高越好。",
    "unknown_precision": "被拒识为 unknown 的样本中真实未知类占比，越高越好。",
    "known_fpr_as_unknown": "真实已知类被错误拒识成 unknown 的比例，越低越好。",
    "unknown_false_accept_rate": "真实未知类被错误接受为某个已知类的比例，越低越好。",
    "oscr": "同时考虑已知类分类正确率和未知拒识能力的综合面积，越高越好。",
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
    return Path(root) / "outputs" / "summaries" / f"RESULT_SUMMARY_{key.upper()}.md"


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
        "| 指标键 | 中文名 | 数值 | 说明 |",
        "| --- | --- | ---: | --- |",
    ]

    for key in CORE_METRICS:
        if key in metrics:
            lines.append(f"| {key} | {METRIC_CN.get(key, key)} | {metrics[key]:.6f} | {METRIC_DESC.get(key, '')} |")

    optional = [
        key
        for key in [
            "macro_precision",
            "macro_recall",
            "weighted_f1",
            "unknown_precision",
            "known_fpr_as_unknown",
            "unknown_false_accept_rate",
            "oscr",
        ]
        if key in metrics
    ]
    if optional:
        lines.extend(["", "## 补充指标", "", "| 指标键 | 中文名 | 数值 | 说明 |", "| --- | --- | ---: | --- |"])
        for key in optional:
            lines.append(f"| {key} | {METRIC_CN.get(key, key)} | {metrics[key]:.6f} | {METRIC_DESC.get(key, '')} |")

    metadata_rows = [
        ("threshold_strategy_used", "阈值策略"),
        ("threshold_mode", "阈值模式"),
        ("score_calibration_mode", "分数校准方式"),
        ("known_rescue_enabled", "已知类救回"),
        ("number_of_tx", "Tx 总数"),
        ("number_of_rx_used", "使用 Rx 数"),
        ("rx_mode", "Rx 协议"),
        ("train_sample_count", "训练样本数"),
        ("val_sample_count", "验证样本数"),
        ("test_known_sample_count", "已知测试样本数"),
        ("test_unknown_sample_count", "未知测试样本数"),
    ]
    if any(key in metrics for key, _ in metadata_rows):
        lines.extend(["", "## 实验协议", "", "| 字段 | 中文名 | 数值 |", "| --- | --- | --- |"])
        for key, cn_name in metadata_rows:
            if key in metrics:
                lines.append(f"| {key} | {cn_name} | `{metrics[key]}` |")
        if metrics.get("known_tx_list") is not None:
            lines.append(f"| known_tx_list | 已知 Tx 列表 | `{', '.join(map(str, metrics['known_tx_list']))}` |")
        if metrics.get("unknown_tx_list") is not None:
            lines.append(f"| unknown_tx_list | 未知 Tx 列表 | `{', '.join(map(str, metrics['unknown_tx_list']))}` |")
        if metrics.get("rx_used") is not None:
            lines.append(f"| rx_used | Rx 列表 | `{', '.join(map(str, metrics['rx_used']))}` |")

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
            "- `outputs/summaries/RESULT_SUMMARY_WISIG.md`：WiSig 最近一次运行结果汇总",
            "- `outputs/summaries/RESULT_SUMMARY_ORACLE.md`：Oracle 最近一次运行结果汇总",
            "- `outputs/<实验名>/final_report.md`：某次具体实验的独立汇总",
            "- `README.md`：项目结构和指标阅读口径",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



