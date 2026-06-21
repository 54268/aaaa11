# 模块消融表回调设计

## 目标

将模块消融恢复为“纯闭集分类 → OpenMax 校准 → OpenMax 与原型距离双重校准 → 完整 PCBM”的递进结构，使表格清楚体现开放集能力随模块加入而整体增强。

## 数据与实验来源

四行必须来自已有真实实验产物，不修改或人工构造指标：

1. `closed_set_only`：纯闭集原型分类，不执行未知拒识；最大 Softmax 不确定度只用于计算 AUROC。
2. `openmax_only`：读取统一协议正式对比实验中的 OpenMax 结果。
3. `ordinary_mbs_only`：使用相同闭集检查点，关闭原型竞争边界建模，保留 OpenMax 与原型距离校准。
4. `full_method`：读取正式 PCBM 结果。

## 展示字段

模块开关列保持不变：

- 原型竞争边界建模
- 原型距离校准
- OpenMax 校准

指标仅保留：

- `known_accuracy`：Known Acc.
- `unknown_recall`：Unknown Recall
- `macro_f1`：Macro F1
- `auroc`：AUROC

不再展示 Overall Acc.、Unknown Precision、Known FPR 和 OSCR，以降低表格密度并突出论文主指标。

## 展示口径

第一行纯闭集分类不会输出 unknown，因此 Unknown Recall 为 0；它展示闭集模型本身的已知类识别能力。后续三行逐步增加开放集模块。

表格用于说明“整体开放集表现逐步增强”，不声称四个单项指标在每一步都严格单调。Known Acc. 与 Unknown Recall 存在阈值权衡，完整 PCBM 的价值是同时保持较高已知类准确率、未知召回率、Macro F1 和 AUROC。

## 产物

- 更新 `ablations/ablation_suite.py` 的模块变体、字段和说明。
- 更新 `tests/test_ablation_support.py`，固定纯闭集定义和四指标字段。
- 重建 `ablations/消融结果汇总.md`、`.json`、`.csv`。
- 恢复并生成 `ablations/模块消融.png`，采用 Oracle 与 WiSig 上下排列的表格形式。
- 更新 `ablations/README.md`，说明纯闭集起点与四行数据来源。

## 验证标准

- 第一行 `threshold_mode` 为 `none`，Unknown Recall 为 0。
- 模块表仅包含三个开关列和四个指标列。
- Markdown 与 PNG 中四行顺序一致。
- PNG 可读、无裁切、数值与 JSON 原始结果一致。
- 消融支持测试与项目检查通过。
