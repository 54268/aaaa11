# 消融实验说明

本目录保存 Oracle 和 WiSig 两个数据集上的四类消融结果，汇总文件位于本目录根部：

- `消融结果汇总.md`
- `消融结果汇总.csv`
- `消融结果汇总.json`

未知类细分主流程、标准化/PCA 解释见根目录 `未知类细分流程说明.md`；当前主结果的自动候选 K 搜索与簇均衡合并口径见根目录 `自动K搜索与未知类细分流程说明.md`。历史 `K+m` 表格仅作为诊断消融保留。

四类实验分别放在：

- `01_模块消融/`：原型竞争边界建模、原型距离校准、OpenMax 校准三项开关。
- `02_KM簇数消融/`：固定 `m=0,1,2,3` 与自动 `Auto` 的簇数选择结果。
- `03_损失函数消融/`：Classification、Angular、Prototype 三项闭集表征损失开关。
- `04_细分流程消融/`：embedding、I/Q 统计描述、特征融合和置信度过滤的流程贡献。

模块消融按逐步加入顺序展示：纯闭集原型分类、OpenMax、OpenMax/原型距离双重校准、完整 PCBM。第一行不执行未知拒识，因此未知类召回率为 0；最大 Softmax 不确定度只用于计算 OSCR。第二行使用统一协议下的正式 OpenMax 结果，第三行在相同闭集检查点上加入原型距离校准，完整方法行复用正式 PCBM 结果。Oracle 第三行采用验证集已知分数的更严格 classwise 阈值重评估，以体现已知类准确率与未知类召回率之间的阈值权衡。每一列用 `X/√` 表示原型竞争边界建模、原型距离校准和 OpenMax 校准是否启用，指标只展示已知类准确率、未知类召回率、Macro F1 和 OSCR。该表用于说明模块加入后整体开放集能力增强；已知类准确率与未知类召回率存在阈值权衡，不要求每个单项在每一步严格单调。

损失函数消融已按一致权重复核：`CE + Prototype` 和 `Full embedding learning` 中的 Prototype Loss 均使用 `lambda_prototype=0.10`。结果显示，Oracle 上 `CE + Angular` 的 Macro F1 和 AUROC 更高，加入 Prototype Loss 后反而下降；WiSig 上 Prototype Loss 只带来很小的 Known Acc./Macro F1 变化，且 Full 组合的 AUROC 下降。因此论文中不应把 Prototype Loss 写成必要贡献。更稳妥的表述是：原型距离分类头已经通过类别 logit

\[
\ell_c(z)=-\left\lVert z-p_c\right\rVert_2^2
\]

隐式提供原型约束，额外欧氏紧致项可能与角度间隔目标重叠，适合放在消融诊断或附录中说明。

K+M 自动选择会始终评估 `m=0,1,2,3`，只在调整后质量提升超过 1 个百分点时才接受更大的 `m`，避免为了很小收益引入冗余簇结构。

细分流程消融中，`Embedding only` 和 `I/Q descriptors only` 不启用低置信过滤或小簇过滤，只检验单一特征本身；`Feature fusion w/o filtering` 使用融合特征但关闭过滤；`Full subdivision` 才同时启用融合特征和过滤。WiSig single-day/fixed-RX 协议中融合特征已经接近饱和，因此 Full 与 w/o filtering 可能相同；Oracle 上过滤模块主要体现为剔除低置信样本、降低覆盖率，当前宽松设置下聚类质量提升幅度有限。

常用命令：

```powershell
python ablations\run_ablation.py --category all --dataset all
python ablations\run_ablation.py --category losses --dataset oracle --loss-variant ce_angular
python ablations\run_ablation.py --summary-only
```
