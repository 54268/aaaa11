# 开放集 SEI 结果汇总

- 数据集：`wisig_compact`
- 配置文件：`E:\learn_pytorch\笔记\方案\os_sei_code\configs\wisig_singleday_osr_k16_u4.yaml`
- 输出目录：`E:\learn_pytorch\笔记\方案\os_sei_code\outputs\wisig_singleday_osr_k16_u4`

## 核心指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| overall_accuracy | 总体准确率 | 0.982465 |
| known_accuracy | 已知类准确率 | 0.960547 |
| macro_f1 | 宏平均F1 | 0.979960 |
| auroc | AUROC | 0.999218 |
| fpr95 | FPR95 | 0.001563 |
| unknown_recall | 未知类召回率 | 1.000000 |

## 补充指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| weighted_f1 | 加权F1 | 0.982337 |
| unknown_precision | 未知类精确率 | 0.969403 |

## 原始结果文件

- `open_set_metrics.json`：完整指标结果
- `confusion_matrix.csv`：混淆矩阵原始数值
- `open_set_predictions.csv`：逐样本预测结果

## 备注

- 图表目录：E:\learn_pytorch\笔记\方案\os_sei_code\figures\wisig_singleday_osr_k16_u4
- RESULT_SUMMARY.md 会被最近一次评估结果自动覆盖。
- 本次实验使用的 split 文件：E:\learn_pytorch\笔记\方案\os_sei_code\data\splits\wisig\single_day_rx1_eq0\wisig_single_day_rx1_eq0_k16_u4_seed42.json
- OpenMax 后端：repo_openmax
