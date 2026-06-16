# WiSig 开放集 SEI 结果汇总

- 数据集：`wisig_compact`
- 配置文件：`D:\learn_pytorch\笔记\方案\os_sei_code\run_wisig.py`
- 输出目录：`D:\learn_pytorch\笔记\方案\os_sei_code\ablations\01_模块消融\wisig\ordinary_mbs_only`

## 核心指标

| 指标键 | 中文名 | 数值 | 说明 |
| --- | --- | ---: | --- |
| overall_accuracy | 总体准确率 | 0.978372 | 已知类分对且未知类拒识正确的总比例，越高越好。 |
| known_accuracy | 已知类准确率 | 0.897266 | 只看真实已知类样本，被分到正确已知类别的比例，越高越好。 |
| macro_f1 | 宏平均 F1 | 0.947862 | 每个类别 F1 的平均值，更关注类别均衡表现，越高越好。 |
| auroc | 已知/未知区分 AUROC | 0.995949 | unknown score 区分已知与未知的整体能力，越接近 1 越好。 |
| fpr95 | 95% 未知召回下的已知误拒率 | 0.010937 | 未知召回约 95% 时，已知样本被误拒为 unknown 的比例，越低越好。 |
| unknown_recall | 未知类召回率 | 1.000000 | 真实未知类中被拒识为 unknown 的比例，越高越好。 |

## 补充指标

| 指标键 | 中文名 | 数值 | 说明 |
| --- | --- | ---: | --- |
| macro_precision | 宏平均精确率 | 0.998431 | 每个类别 precision 的平均值，越高越好。 |
| macro_recall | 宏平均召回率 | 0.903309 | 每个类别 recall 的平均值，越高越好。 |
| weighted_f1 | 加权 F1 | 0.977847 | 按类别样本数加权后的 F1，越高越好。 |
| unknown_precision | 未知类精确率 | 0.973335 | 被拒识为 unknown 的样本中真实未知类占比，越高越好。 |
| known_fpr_as_unknown | 已知类误拒率 | 0.102734 | 真实已知类被错误拒识成 unknown 的比例，越低越好。 |
| unknown_false_accept_rate | 未知类误接收率 | 0.000000 | 真实未知类被错误接受为某个已知类的比例，越低越好。 |
| oscr | 开放集分类识别曲线面积 | 0.897236 | 同时考虑已知类分类正确率和未知拒识能力的综合面积，越高越好。 |

## 实验协议

| 字段 | 中文名 | 数值 |
| --- | --- | --- |
| threshold_strategy_used | 阈值策略 | `classwise_balanced` |
| threshold_mode | 阈值模式 | `classwise_balanced` |
| score_calibration_mode | 分数校准方式 | `none` |
| known_rescue_enabled | 已知类救回 | `False` |
| number_of_tx | Tx 总数 | `28` |
| number_of_rx_used | 使用 Rx 数 | `1` |
| rx_mode | Rx 协议 | `fixed` |
| train_sample_count | 训练样本数 | `8960` |
| val_sample_count | 验证样本数 | `1280` |
| test_known_sample_count | 已知测试样本数 | `2560` |
| test_unknown_sample_count | 未知测试样本数 | `9600` |
| known_tx_list | 已知 Tx 列表 | `13-3, 20-7, 7-10, 6-15, 3-13, 11-4, 8-3, 7-11, 15-1, 3-18, 4-11, 14-7, 11-7, 5-5, 11-1, 1-11` |
| unknown_tx_list | 未知 Tx 列表 | `20-19, 6-1, 2-19, 16-16, 20-15, 8-20, 10-7, 11-17, 8-18, 10-11, 20-12, 14-10` |
| rx_used | Rx 列表 | `1-1` |

## 原始结果文件

- `open_set_metrics.json`：完整指标结果
- `confusion_matrix.csv`：混淆矩阵原始数值
- `open_set_predictions.csv`：逐样本预测结果

## 备注

- 图表目录：D:\learn_pytorch\笔记\方案\os_sei_code\ablations\01_模块消融\wisig\ordinary_mbs_only\figures
- 划分文件：D:\learn_pytorch\笔记\方案\os_sei_code\data\splits\wisig\single_day_rx1_eq0\wisig_single_day_rx1_eq0_k16_u12_seed42.json
- OpenMax 后端：repo_openmax
