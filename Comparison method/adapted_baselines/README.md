# 对比方法适配说明

本目录用于适配 `Comparison method/raw/` 中的三种论文对比方法。这里的代码只读取主项目已有数据，不修改主项目代码、配置或输出。

## 目录说明

- `run_comparison.py`：唯一运行入口，常用参数在文件顶部直接修改。
- `src/data_io.py`：读取 WiSig 与 Oracle 的 `npz` 数据。
- `src/models.py`：HyperRSI、HyDRA、OpenRFI 风格模型。
- `src/trainers.py`：训练、特征提取、拒识和细分流程。
- `src/metrics.py`：开放集与聚类指标。
- `src/reporting.py`：保存 CSV、JSON、Markdown 报告。

## 输出位置

所有输出写入：

```text
Comparison method/adapted_results/
```

不会写入主项目的 `outputs/`、`functions/`、`settings/` 或根目录入口文件。

## 指标口径

对比方法的指标已经按本方法拆成两套表格：

- 开放集拒识：`overall_accuracy`、`macro_precision`、`macro_recall`、`macro_f1`、`weighted_f1`、`known_accuracy`、`unknown_precision`、`unknown_recall`、`known_fpr_as_unknown`、`unknown_false_accept_rate`、`auroc`、`fpr95`、`oscr`。
- 未知类细分：`nmi`、`ari`、`purity`、`hungarian_accuracy`，逐次结果中额外保留 `resolved_num_clusters`、`uncertain_size`、`unknown_cache_precision`、`unknown_cache_recall` 等与本方法报告对应的辅助字段。

主指标表直接使用本方法的指标名，不再添加 `_mean` / `_std`。如果以后需要多随机种子统计，可以另外单独生成统计表，不混进论文对比主表。

正式结果目录中主要看：

- `open_set_summary_results.csv`
- `open_set_per_seed_results.csv`
- `unknown_subdivision_summary_results.csv`
- `unknown_subdivision_per_seed_results.csv`
- `comparison_report.md`

## 运行说明

直接运行：

```text
D:\Anaconda3\envs\pytorch\python.exe "D:\learn_pytorch\笔记\方案\os_sei_code\Comparison method\adapted_baselines\run_comparison.py"
```

默认是 `formal` 模式，用于生成正式对比结果。调试流程时打开 `run_comparison.py`，把 `RUN_MODE = "formal"` 改为 `RUN_MODE = "smoke"`。

`REUSE_EXISTING_CHECKPOINTS = True` 时会优先复用已有权重重新评估并整理指标；如果需要重新训练，把它改为 `False`。
