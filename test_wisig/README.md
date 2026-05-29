# WiSig 天然可分性诊断实验

本目录完全隔离于主实验，用于诊断 WiSig 当前 unknown 类是否在原始 I/Q 或 FFT 弱特征下已经容易聚类。

运行入口：

```text
python test_wisig/run_wisig_separability_diagnostic.py
```

输入数据只读来自 `data/raw/wisig/`，当前项目配置和已有结果只读用于记录协议和旁注。所有新增脚本、缓存、日志、CSV、JSON、图片和 Markdown 报告均写入 `test_wisig/`。

主要输出：

- `config/detected_project_protocol.json`：自动识别到的当前 WiSig 主协议。
- `environment_check.txt`：依赖检查。
- `results/inventory/`：四个 official compact subsets 扫描结果。
- `results/singleday_current_protocol/`：当前 SingleDay unknown 协议直接聚类结果。
- `results/all_subsets_p1/`：四个 subset 的统一 6 类弱基线结果。
- `results/shared_tx_p2/`：共有 Tx 对齐比较。
- `results/condition_sensitivity_p3/`：receiver/day 条件敏感性诊断。
- `results/oracle_reference/`：当前已有 Oracle 细分结果旁注。
- `results/final_diagnostic_report.md`：总报告。
