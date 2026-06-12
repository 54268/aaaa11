# Oracle 开放集 SEI 结果汇总

- 数据集：`oracle_sigmf`
- 配置文件：`D:\learn_pytorch\笔记\方案\os_sei_code\run_oracle.py`
- 输出目录：`D:\learn_pytorch\笔记\方案\os_sei_code\ablations\04_细分流程消融\oracle\iq_descriptors_only`

## 核心指标

| 指标键 | 中文名 | 数值 | 说明 |
| --- | --- | ---: | --- |
| overall_accuracy | 总体准确率 | 0.966219 | 已知类分对且未知类拒识正确的总比例，越高越好。 |
| known_accuracy | 已知类准确率 | 0.963625 | 只看真实已知类样本，被分到正确已知类别的比例，越高越好。 |
| macro_f1 | 宏平均 F1 | 0.936956 | 每个类别 F1 的平均值，更关注类别均衡表现，越高越好。 |
| auroc | 已知/未知区分 AUROC | 0.986836 | unknown score 区分已知与未知的整体能力，越接近 1 越好。 |
| fpr95 | 95% 未知召回下的已知误拒率 | 0.035375 | 未知召回约 95% 时，已知样本被误拒为 unknown 的比例，越低越好。 |
| unknown_recall | 未知类召回率 | 0.967083 | 真实未知类中被拒识为 unknown 的比例，越高越好。 |

## 补充指标

| 指标键 | 中文名 | 数值 | 说明 |
| --- | --- | ---: | --- |
| macro_precision | 宏平均精确率 | 0.918591 | 每个类别 precision 的平均值，越高越好。 |
| macro_recall | 宏平均召回率 | 0.963939 | 每个类别 recall 的平均值，越高越好。 |
| weighted_f1 | 加权 F1 | 0.968003 | 按类别样本数加权后的 F1，越高越好。 |
| unknown_precision | 未知类精确率 | 0.992814 | 被拒识为 unknown 的样本中真实未知类占比，越高越好。 |
| known_fpr_as_unknown | 已知类误拒率 | 0.021000 | 真实已知类被错误拒识成 unknown 的比例，越低越好。 |
| unknown_false_accept_rate | 未知类误接收率 | 0.032917 | 真实未知类被错误接受为某个已知类的比例，越低越好。 |
| oscr | 开放集分类识别曲线面积 | 0.957482 | 同时考虑已知类分类正确率和未知拒识能力的综合面积，越高越好。 |

## 实验协议

| 字段 | 中文名 | 数值 |
| --- | --- | --- |
| threshold_strategy_used | 阈值策略 | `manual_classwise` |
| threshold_mode | 阈值模式 | `manual_classwise` |
| score_calibration_mode | 分数校准方式 | `classwise_z` |
| known_rescue_enabled | 已知类救回 | `True` |
| number_of_tx | Tx 总数 | `16` |
| number_of_rx_used | 使用 Rx 数 | `N/A` |
| rx_mode | Rx 协议 | `not_applicable` |
| train_sample_count | 训练样本数 | `28000` |
| val_sample_count | 验证样本数 | `4000` |
| test_known_sample_count | 已知测试样本数 | `8000` |
| test_unknown_sample_count | 未知测试样本数 | `24000` |
| known_tx_list | 已知 Tx 列表 | `1, 2, 3, 4, 5, 7, 9, 13, 14, 15` |
| unknown_tx_list | 未知 Tx 列表 | `17, 18, 19, 25, 26, 32` |
| rx_used | Rx 列表 | `N/A` |

## 原始结果文件

- `open_set_metrics.json`：完整指标结果
- `confusion_matrix.csv`：混淆矩阵原始数值
- `open_set_predictions.csv`：逐样本预测结果

## 备注

- 图表目录：D:\learn_pytorch\笔记\方案\os_sei_code\ablations\04_细分流程消融\oracle\iq_descriptors_only\figures
- 划分文件：D:\learn_pytorch\笔记\方案\os_sei_code\data\splits\oracle\oracle_main_sorted_k10_u6.json
- OpenMax 后端：repo_openmax
