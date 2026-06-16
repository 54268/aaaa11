# 消融实验说明

本目录保存 Oracle 和 WiSig 两个数据集上的四类消融结果，汇总文件位于本目录根部：

- `消融结果汇总.md`
- `消融结果汇总.csv`
- `消融结果汇总.json`

未知类细分主流程、标准化/PCA 解释和 `K+m` 选择规则见根目录 `未知类细分流程说明.md`。

四类实验分别放在：

- `01_模块消融/`：原型竞争边界建模、原型距离校准、OpenMax 校准三项开关。
- `02_KM簇数消融/`：固定 `m=0,1,2,3` 与自动 `Auto` 的簇数选择结果。
- `03_损失函数消融/`：Classification、Angular、Prototype 三项闭集表征损失开关。
- `04_细分流程消融/`：embedding、I/Q 统计描述、特征融合和置信度过滤的流程贡献。

模块消融按逐步加入顺序展示：普通 MBS（基础拒识）、OpenMax、OpenMax/原型距离双重校准、完整 PCBM。前三行使用同一闭集检查点；第一行保留基于最大 softmax 的全局置信度拒识，阈值由验证已知集分位数确定，但不使用 OpenMax、原型距离校准或 PCBM。完整方法行复用正式 PCBM 结果。每一列用 `X/√` 表示 PCBM、原型距离校准和 OpenMax 校准是否启用，并展示总体准确率、已知类准确率、未知类召回率、未知类精确率、已知类误拒率、Macro F1、OSCR 和 AUROC。原型距离校准重点观察 AUROC/OSCR，PCBM 重点观察已知类误拒率、未知类精确率和 OSCR。Oracle 的原型距离校准行在验证集上搜索 `λ∈{0.1,0.3,0.5,0.7,0.9}`，并要求每类验证已知接收率不低于 90%，测试标签不参与参数选择。

损失函数消融已按一致权重复核：`CE + Prototype` 和 `Full embedding learning` 中的 Prototype Loss 均使用 `lambda_prototype=0.10`。结果显示，Oracle 上 `CE + Angular` 的 Macro F1 和 AUROC 更高，加入 Prototype Loss 后反而下降；WiSig 上 Prototype Loss 只带来很小的 Known Acc./Macro F1 变化，且 Full 组合的 AUROC 下降。因此论文中不应把 Prototype Loss 写成必要贡献。更稳妥的表述是：原型距离分类头已经通过类别 logit

\[
\ell_c(z)=-\left\lVert z-p_c\right\rVert_2^2
\]

隐式提供原型约束，额外欧氏紧致项可能与角度间隔目标重叠，适合放在消融诊断或附录中说明。

K+M 自动选择会始终评估 `m=0,1,2,3`，只在调整后质量提升超过 1 个百分点时才接受更大的 `m`，避免为了很小收益引入冗余簇结构。

细分流程消融中，`Embedding only` 和 `I/Q descriptors only` 不启用低置信过滤或小簇过滤，只检验单一特征本身；`Feature fusion w/o filtering` 使用融合特征但关闭过滤；`Full subdivision` 才同时启用融合特征和过滤。WiSig single-day/fixed-RX 协议中融合特征已经接近饱和，因此 Full 与 w/o filtering 可能相同；Oracle 上过滤模块仍能显著提升聚类质量，但会降低覆盖率。

常用命令：

```powershell
python ablations\run_ablation.py --category all --dataset all
python ablations\run_ablation.py --category losses --dataset oracle --loss-variant ce_angular
python ablations\run_ablation.py --summary-only
```
