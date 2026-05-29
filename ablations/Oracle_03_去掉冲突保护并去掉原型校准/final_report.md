# Oracle 开放集 SEI 结果汇总

- 数据集：`oracle_sigmf`
- 配置文件：`D:\learn_pytorch\笔记\方案\os_sei_code\experiments\ablations\Oracle_03_去掉冲突保护并去掉原型校准\config.yaml`
- 输出目录：`D:\learn_pytorch\笔记\方案\os_sei_code\experiments\ablations\Oracle_03_去掉冲突保护并去掉原型校准`

## 核心指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| overall_accuracy | 总体准确率 | 0.679750 |
| known_accuracy | 已知类准确率 | 0.943125 |
| macro_f1 | 宏平均F1 | 0.790817 |
| auroc | AUROC | 0.952017 |
| fpr95 | FPR95 | 0.157250 |
| unknown_recall | 未知类召回率 | 0.591958 |

## 补充指标

| 指标键 | 中文名 | 数值 |
| --- | --- | ---: |
| weighted_f1 | 加权F1 | 0.752850 |
| unknown_precision | 未知类精确率 | 0.981282 |

## 原始结果文件

- `open_set_metrics.json`：完整指标结果
- `confusion_matrix.csv`：混淆矩阵原始数值
- `open_set_predictions.csv`：逐样本预测结果

## 备注

- 图表目录：D:\learn_pytorch\笔记\方案\os_sei_code\experiments\ablations\Oracle_03_去掉冲突保护并去掉原型校准\figures
- 根目录结果汇总已改为按数据集分别保存，不会相互覆盖。
- 本次实验使用的 split 文件：E:\learn_pytorch\笔记\方案\os_sei_code\data\splits\oracle\oracle_main_sorted_k10_u6.json
- OpenMax 后端：repo_openmax

