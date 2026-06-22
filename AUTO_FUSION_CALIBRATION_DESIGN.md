# 伪未知驱动的自动融合校准设计

## 目标

最终开放集参数只能由已知验证集与特征层伪未知集确定，真实未知测试集不得参与融合权重、拒识阈值或已知救回规则的选择。

## 固定协议

- 融合权重网格：`lambda ∈ {0.0, 0.1, ..., 1.0}`。
- 选择目标：`0.45 * Known Acc. + 0.35 * Pseudo-unknown Recall + 0.15 * Macro F1 + 0.05 * AUROC`。
- 全局已知准确率约束：校准集 Known Acc. 不低于 `0.95`。
- Oracle：保留 `classwise_z` 分数标准化，使用 `classwise_balanced` 自动类别阈值，每类已知接收率不低于 `0.90`。
- WiSig：不做额外分数标准化，使用 `global` 自动阈值。
- 两套数据均取消 `manual_threshold`、`manual_thresholds_per_class` 和人工 `known_rescue`。
- 搜索完成后保存 `fusion.json`，再使用冻结参数运行一次 `test_known + test_unknown`。

## 评价

以修改前已保存的正式开放集结果为基线，比较 Overall Acc.、Known Acc.、Unknown Precision、Unknown Recall、Known FPR、Macro F1、AUROC、FPR95 和 OSCR。参数选择规则在读取新测试结果之前固定，测试结果不用于二次调参。

## t-SNE口径

t-SNE不使用OpenMax、原型距离融合或拒识阈值。编码器仅由已知类训练；冻结后，`test_known + test_unknown` 经编码器得到128维嵌入，再离线执行PCA和t-SNE。
