# WiSig 开放集 SEI 结果汇总

- 数据集：`wisig_compact`
- 配置文件：`E:\learn_pytorch\笔记\方案\os_sei_code\configs\wisig_singleday_osr_k16_u12.yaml`
- 输出目录：`E:\learn_pytorch\笔记\方案\os_sei_code\outputs\wisig_singleday_osr_k16_u12`

## 核心指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| overall_accuracy | 总体准确率 | 0.987336 |
| known_accuracy | 已知类准确率 | 0.960547 |
| macro_f1 | 宏平均F1 | 0.972061 |
| auroc | AUROC | 0.998414 |
| fpr95 | FPR95 | 0.001563 |
| unknown_recall | 未知类召回率 | 0.994479 |

## 补充指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| weighted_f1 | 加权F1 | 0.987539 |
| unknown_precision | 未知类精确率 | 0.989532 |

## 原始结果文件

- `open_set_metrics.json`：完整指标结果
- `confusion_matrix.csv`：混淆矩阵原始数值
- `open_set_predictions.csv`：逐样本预测结果

## 备注

- 图表目录：E:\learn_pytorch\笔记\方案\os_sei_code\figures\wisig_singleday_osr_k16_u12
- 根目录结果汇总已改为按数据集分别保存，不会相互覆盖。
- 本次实验使用的 split 文件：E:\learn_pytorch\笔记\方案\os_sei_code\data\splits\wisig\single_day_rx1_eq0\wisig_single_day_rx1_eq0_k16_u12_seed42.json
- OpenMax 后端：repo_openmax
