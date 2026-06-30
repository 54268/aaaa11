# 自动K搜索与未知类细分流程说明

本文档专门说明当前 PCBM 未知类细分阶段如何自动搜索候选 K，以及整体细分流程相对旧版 `K+m` 口径发生了什么变化。

## 1. 核心口径

当前细分阶段不把真实未知类数提前告诉模型。自动搜索的对象是 GMM 候选分量数 `fit_K`，不是最终真实类别数。

最终表中的 `effective_K` 来自无标签后处理：

```text
fit_K
  -> GMM 候选分量
  -> 低置信/小簇过滤
  -> 簇均衡自动合并
  -> effective_K
```

因此 `fit_K` 可以大于最终有效簇数。Oracle 当前为 `fit_K=7 -> effective_K=6`；WiSig 当前为 `fit_K=13 -> effective_K=12`。

## 2. 为什么旧 auto-K 不够

旧版 sample-only auto-K 只在抽样 cache 上根据内部聚类分数选 K，结果会明显欠分裂：

| 数据集/配置 | NMI | ARI | Hungarian Acc. | Coverage | fit_K | effective_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle old sample auto-K | 0.759842 | 0.571284 | 0.507773 | 0.948833 | 3 | 3 |
| WiSig old sample auto-K | 0.949306 | 0.840598 | 0.832708 | 1.000000 | 10 | 10 |

原因是抽样阶段看不到完整过滤/合并后的簇结构。Oracle 的 `K=3` 在抽样上规模均衡，但实际把多个真实未知类合在一起；WiSig 的 `K=10` 也欠分裂。

## 3. 当前统一候选评分

当前做法是对候选 `fit_K` 执行完整细分后，用同一套无标签得分选择候选：

```text
score(fit_K)
  = normalized(GMM lower_bound)
  + 3.0 * cluster_balance
  - 0.03 * fit_K
```

各项含义：

- `GMM lower_bound`：GMM 对 unknown cache 的平均对数似然下界，用于避免欠分裂。
- `cluster_balance`：后处理后簇规模均衡度，当前使用 `mean_cluster_size / max_cluster_size`，用于惩罚极端大簇吞并多个类别。
- `0.03 * fit_K`：复杂度惩罚，用于避免无限增加候选分量。

这个规则不使用真实未知标签，也不使用真实未知类数。

## 4. 自动合并规则

候选 GMM 可能把一个真实未知类拆成一个大簇和一个小簇。为避免把这种子模态当作新类别，当前加入无标签簇均衡合并：

```text
while min_cluster_size / mean_cluster_size < 0.75:
    将最小簇合并到最近中心
```

WiSig 的 13 分量直接输出时：

| 配置 | NMI | ARI | Hungarian Acc. | Coverage | fit_K | effective_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct 13 no merge | 0.986711 | 0.975586 | 0.960625 | 1.000000 | 13 | 13 |
| unified auto-merge | 0.998125 | 0.998637 | 0.999375 | 1.000000 | 13 | 12 |

自动合并只触发 1 次，把冗余小簇并回最近簇。

## 5. 整体细分流程

当前完整流程如下：

```text
开放集拒识输出
  -> unknown cache
  -> 构造 embedding + I/Q statistics
  -> 标准化
  -> PCA96
  -> 扫描候选 fit_K
      -> GMM-full-direct
      -> posterior confidence 过滤
      -> 小簇过滤
      -> 簇均衡自动合并
      -> 计算无标签候选分数
  -> 选择分数最高的 fit_K
  -> 输出有效未知簇与 -1 不确定样本
  -> 仅离线评估 NMI / ARI / Hungarian Acc. / Coverage
```

其中真实未知标签只用于最后一行离线评估。

## 6. 当前主结果

| 数据集 | NMI | ARI | Hungarian Acc. | Coverage | fit_K | effective_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | 0.991948 | 0.993851 | 0.997400 | 0.945583 | 7 | 6 |
| WiSig | 0.998125 | 0.998637 | 0.999375 | 1.000000 | 13 | 12 |

Oracle 三个聚类一致性指标均超过 0.99，coverage 为 0.945583。WiSig 四项细分指标均高于当前 OpenRFI 细分结果。

## 7. 论文写法建议

建议在论文方法部分写成：

```text
PCBM does not assume the number of unknown emitters is known in advance. 
It first searches the number of GMM candidate components with an unsupervised criterion combining likelihood, cluster-size balance, and model complexity. 
The final number of effective unknown clusters is then determined by confidence filtering and an automatic balance-based merge rule.
```

中文口径：

```text
本文方法不预先给定未知类数，而是自动选择 GMM 候选分量数。最终有效未知簇数由低置信过滤、小簇过滤和簇规模均衡合并共同决定。
```

避免写法：

```text
给定真实未知类数 K 后运行 K+m 搜索。
```

这个说法只适合旧版消融或诊断实验，不适合当前主结果。
