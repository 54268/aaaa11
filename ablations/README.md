# 消融实验说明

本目录保存 Oracle 和 WiSig 两个数据集上的四类消融结果，汇总文件位于本目录根部：

- `消融结果汇总.md`
- `消融结果汇总.csv`
- `消融结果汇总.json`

四类实验分别放在：

- `01_模块消融/`：原型竞争边界建模、原型距离校准、OpenMax 校准三项开关。
- `02_KM簇数消融/`：固定 `m=0,1,2,3` 与自动 `Auto` 的簇数选择结果。
- `03_损失函数消融/`：Classification、Angular、Prototype 三项闭集表征损失开关。
- `04_细分流程消融/`：embedding、I/Q 统计描述、特征融合和置信度过滤的流程贡献。

模块消融按逐步加入顺序展示：闭集原型分类、加入 OpenMax、加入原型距离校准、完整 PCBM。OpenMax 行复用正式对比实验结果，完整方法行复用正式 PCBM 结果，中间行在相同闭集检查点上关闭 PCBS 后重新运行。每一列用 `X/√` 表示模块是否启用，并直接拼接 `overall_accuracy`、`known_accuracy`、`unknown_recall`、`macro_f1` 和 `auroc` 等指标。

损失函数消融中，Prototype Loss 已按原型距离空间改为欧氏紧致损失。实测结果显示 Angular Loss 是 Oracle 上更稳定的主要收益来源，Prototype Loss 在 WiSig 上有正向增益，但在 Oracle 联合开启时不是严格单调提升，因此论文中不宜把它单独表述为每加必涨的强贡献。

K+M 自动选择会始终评估 `m=0,1,2,3`，只在调整后质量提升超过 1 个百分点时才接受更大的 `m`，避免为了很小收益引入冗余簇结构。

常用命令：

```powershell
python ablations\run_ablation.py --category all --dataset all
python ablations\run_ablation.py --category losses --dataset oracle --loss-variant ce_angular
python ablations\run_ablation.py --summary-only
```
