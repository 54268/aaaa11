# 消融实验说明

本目录保存 Oracle 和 WiSig 两个数据集上的当前消融结果，汇总文件位于本目录根部：

- `消融结果汇总.md`
- `消融结果汇总.csv`
- `消融结果汇总.json`

当前保留三类消融：

- `01_模块消融/`：原型竞争边界建模、原型距离校准和 OpenMax 校准的模块开关。
- `03_损失函数消融/`：Classification、Angular、Prototype 三项闭集表征损失开关。
- `04_细分流程消融/`：Embedding only、I/Q descriptors only、Feature fusion 三种未知类细分特征设置。

历史诊断性簇数敏感性实验已删除；当前方法不再把真实未知类数作为给定 K，也不再把额外缓冲分量作为主消融口径。新的自动 K 分析直接列出每个数据集在当前 Feature fusion 主流程下自动选择的 `fit_K`、最终有效簇数、coverage、NMI、ARI 和 Hungarian Acc.。

细分流程消融只用于回答“使用哪类细分特征更有效”：`Embedding only` 和 `I/Q descriptors only` 检验单一特征，`Feature fusion` 使用 embedding 与 I/Q 统计描述拼接后进入标准化、PCA 和 auto-K GMM 流程。GMM 后的低置信/小簇处理不再作为单独模块写入消融表，而是视为 auto-K GMM 拟合、筛选和合并过程的一部分。

常用命令：

```powershell
python ablations\run_ablation.py --category all --dataset all
python ablations\run_ablation.py --category losses --dataset oracle --loss-variant ce_angular
python ablations\run_ablation.py --summary-only
```
