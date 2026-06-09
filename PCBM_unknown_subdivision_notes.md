# PCBM 未知类细分流程说明：最新实现口径与 GMM/KMeans 对比

## 1. 模块定位

PCBM 的完整处理过程由两个相互衔接但目标不同的阶段组成：

1. **未知发射源拒识（unknown-emitter rejection）**：判断接收信号是否属于已登记的已知发射源集合。
2. **拒识后未知类细分（post-rejection unknown-emitter subdivision）**：仅对已经被判定为 `unknown` 的样本进一步划分，分析其中可能存在的不同未知发射源类别结构。

因此，未知类细分不是对全部测试样本重新分类，也不是在训练阶段引入真实未知标签，而是拒识完成后的结构解析步骤。

```text
测试样本
   │
   ├── 判为已知 ──> 输出已知发射源类别
   │
   └── 判为 unknown ──> unknown cache
                             │
                             └── 未知类细分 ──> U_hat_1, U_hat_2, ..., U_hat_K 或 -1
```

其中，`-1` 表示该样本在细分阶段被判定为**疑似误拒的已知样本**或不确定样本，不强行归入某个未知簇。

---

## 2. unknown cache：细分对象从哪里来

开放集拒识完成后，所有被最终判为 `unknown` 的测试样本被存入 **unknown cache**：

\[
\mathcal{X}_{u}^{\mathrm{cache}}
=
\left\{
x_i \mid \widehat{y}_i=\mathrm{unknown}
\right\}.
\]

理想情况下，unknown cache 中的大部分样本来自真实未知发射源；但在实际拒识过程中，少量真实已知样本也可能因拒识边界、噪声或阈值作用而被误拒，从而混入缓存。

| 样本类型 | 含义 | 细分中的处理目标 |
|---|---|---|
| 正确拒识的真实未知样本 | 应属于某个潜在未知发射源 | 被划分到未知簇 |
| 被误拒的真实已知样本 | 对未知细分构成污染 | 尽量过滤为不确定样本 `-1` |

这一设计意味着，细分阶段不仅要完成未知结构聚类，还要考虑 unknown cache 中的已知类污染问题。

---

## 3. 从拒识到细分的完整数据流

以当前 Oracle 与 WiSig 的统一细分管线为准，完整流程如下。两者共享 `embedding_stats + PCA96 + unknown-only GMM-full-direct` 口径；GMM 候选分量数统一写为 `K+m`，其中 `K` 为协议给定未知类数，`m∈{0,1,2,3}` 为小范围冗余候选因子。Oracle 因未知类混合更明显，主结果取 `m=2`；WiSig 当前未知类更干净，主结果取 `m=0`。

```text
原始 I/Q 信号
   ↓
CVCNN 特征提取器
   ↓
128 维 L2 归一化 embedding
   ↓
原型分类与 PCBM 开放集拒识
   ↓
被判为 unknown 的样本进入 unknown cache
   ↓
为每个缓存样本构造细分特征
Oracle: embedding_stats
WiSig: embedding_stats
   ↓
按数据集配置拟合 Standardization 与 PCA
Oracle: 仅用 unknown cache 特征拟合预处理器
WiSig: 仅用 unknown cache 特征拟合预处理器
   ↓
映射到数据集对应的 PCA 表示空间
Oracle: 96 维
WiSig: 96 维
   ↓
GMM-full-direct 拟合 K+m 个候选分量
Oracle: K=6, m=2，即 8 个候选分量
WiSig: K=12, m=0，即 12 个候选分量
   ↓
后验低置信过滤、小簇剔除，以及按配置启用的已知原型污染过滤
   ↓
有效未知细分标签或不确定标签 -1
   ↓
使用真实未知标签进行离线指标评估
```

真实未知标签仅在最后的离线评估阶段使用，不参与拒识判决、特征构造、聚类拟合或污染过滤。

---

## 4. 细分特征：embedding 与 `embedding_stats`

### 4.1 128 维深度嵌入特征

在闭集训练阶段，复值卷积神经网络（complex-valued convolutional neural network, CVCNN）已经学习了从 I/Q 信号到特征空间的映射。对进入 unknown cache 的每个样本 \(x_i\)，特征提取器输出一个 128 维归一化 embedding：

\[
\mathbf{z}_i \in \mathbb{R}^{128}.
\]

该向量描述样本自身在深度表征空间中的位置。

### 4.2 到所有已知原型的距离特征

设系统中包含 \(C\) 个已知发射源，其类别原型为：

\[
\mathbf{p}_1,\mathbf{p}_2,\ldots,\mathbf{p}_C.
\]

对样本 \(\mathbf{z}_i\)，计算其到全部已知类原型的距离，形成距离向量：

\[
\mathbf{d}_i =
\left[
d(\mathbf{z}_i,\mathbf{p}_1),
d(\mathbf{z}_i,\mathbf{p}_2),
\ldots,
d(\mathbf{z}_i,\mathbf{p}_C)
\right].
\]

该向量描述样本相对于已有合法设备体系的关系模式。例如，两个未知样本本身的深度特征较近，但它们与多个已知类原型的距离排序不同，距离向量仍可为细分提供额外区分依据。该特征仍作为备选实现保留，但当前主结果不再默认使用。

### 4.3 当前数据集特定特征

当前两个数据集统一使用同一个主特征输入：

| 数据集 | 当前特征模式 | 细分输入 |
|---|---|---|
| Oracle | `embedding_stats` | 128 维深度 embedding \(\mathbf{z}_i\) 加原始 I/Q 统计特征 |
| WiSig | `embedding_stats` | 128 维深度 embedding \(\mathbf{z}_i\) 加原始 I/Q 统计特征 |

### 4.4 为什么当前不再默认拼接原型距离

若仅使用 embedding，聚类器只能观察未知样本彼此之间的几何结构；加入原型距离向量后，聚类器还可获得每个未知样本相对于已知发射源体系的位置关系。

但在当前 Oracle 协议中，距离向量会把部分真实未知发射源映射到相似的已知原型关系模式，反而加重未知类之间的混合。因此当前统一使用 `embedding_stats`，即在深度 embedding 外补充每个样本自身的 I/Q、幅度和相位差统计特征。同步到 WiSig 后，细分指标没有明显回退，因此当前主流程也采用 `embedding_stats`。

| 特征组成 | 主要信息 |
|---|---|
| Deep embedding | 未知样本自身的表征结构 |
| Prototype-distance vector | 未知样本相对于已知类原型的关系结构 |

这一处理与 PCBM 的总体思想一致：原型关系主要参与拒识边界建模；拒识后的未知结构分析则优先依赖未知样本自身的深度表征和 I/Q 统计结构。

---

## 5. 标准化与 PCA：拼接特征并非直接送入聚类器

细分特征构造完成后，当前实现会先进行统一预处理，而不是直接交给聚类模型。

### 5.1 预处理器拟合方式

细分前，当前实现按数据集配置决定是否把已知原型锚点纳入预处理坐标系：

| 数据集 | 预处理器拟合样本 |
|---|---|
| Oracle | 仅使用 unknown cache 的 `embedding_stats` 特征 |
| WiSig | 仅使用 unknown cache 的 `embedding_stats` 特征 |

之后执行：

1. 标准化（standardization）；
2. 主成分分析（principal component analysis, PCA）降维。

当前 PCA 维度为：

| 数据集 | PCA 维度 |
|---|---:|
| Oracle | 96 |
| WiSig | 96 |

因此，最终传入聚类模型的是降维后的 PCA 表示，而不是原始 embedding 或原始拼接特征。

### 5.2 已知原型锚点参与预处理的意义

已知原型锚点后续可用于判断 unknown cache 中的样本是否过于接近某个已知类。将未知样本与已知锚点放入同一个标准化与降维坐标系，可使二者的距离比较保持一致尺度。

当前 Oracle 与 WiSig 都关闭这一机制，因为两个数据集当前主流程都采用 unknown-only 的 `embedding_stats` 细分口径，避免已知原型关系对未知类结构造成额外混合。

---

## 6. 聚类后端：`gmm_full_direct`

当前 Oracle 与 WiSig 的默认细分后端统一为：

```python
clustering_backend = gmm_full_direct
```

其中，GMM 指 **Gaussian mixture model（高斯混合模型）**，`full` 表示每个未知簇采用完整协方差矩阵，`direct` 表示直接采用 GMM 的预测标签，不再进行后续基于余弦距离的未知簇迭代重分配。

### 6.1 GMM-full 的建模方式

GMM 将未知样本分布描述为 \(K\) 个高斯成分的混合：

\[
p(\mathbf{v}) =
\sum_{k=1}^{K}
\pi_k
\mathcal{N}(\mathbf{v}\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k),
\]

其中：

- \(\pi_k\) 为第 \(k\) 个未知簇的混合权重；
- \(\boldsymbol{\mu}_k\) 为该未知簇的中心；
- \(\boldsymbol{\Sigma}_k\) 为该未知簇的协方差矩阵；
- \(\mathbf{v}\) 为标准化和 PCA 后的细分特征。

采用全协方差矩阵时，每个未知簇可以呈现不同大小、不同方向和不同伸展程度的椭球形分布。

### 6.2 `direct` 的具体含义

在 `gmm_full_direct` 设置下，主流程为：

```text
GMM-full 直接预测未知簇标签
        ↓
不执行基于余弦距离的聚类中心迭代重分配
        ↓
按配置执行已知原型污染过滤
```

因此，当前主实验中，GMM 负责直接给出候选未知簇。Oracle 与 WiSig 当前都不启用已知原型锚点过滤，主要依靠 GMM 后验置信度和小簇规则处理不确定样本。

---

## 7. GMM 与 KMeans 的差别

### 7.1 KMeans：以最近中心为依据的硬划分

KMeans 通过最小化样本到所属聚类中心的平方欧氏距离完成分组：

\[
\min_{\{\boldsymbol{\mu}_k\}}
\sum_i
\left\|
\mathbf{v}_i - \boldsymbol{\mu}_{c_i}
\right\|_2^2.
\]

它为每个样本直接给出一个标签，并主要依据“离哪个中心更近”决定归属。KMeans 较适合各簇近似球形、扩散程度接近、特征相关性不强的情况。

**形象理解：**KMeans 像是在特征平面中放置若干个圆形磁铁。每个样本被最近的磁铁吸引；磁铁关心中心位置，却不会主动描述某个簇是否狭长、倾斜或比其他簇更分散。

### 7.2 GMM-full：用可旋转的椭球分布解释样本

GMM-full 不仅估计每个簇的中心，还估计其完整协方差，因此可以表达：

- 某个簇较紧，另一个簇较松散；
- 某个簇沿某一方向明显拉长；
- 特征维度之间存在相关变化；
- 不同簇具有不同形状与朝向。

**形象理解：**GMM-full 更像是在特征空间中为不同未知发射源拟合若干个可以旋转和拉伸的椭圆云团。样本归属不只取决于离哪个中心更近，还取决于它是否符合该云团的形状、方向和密度。

### 7.3 对比总结

| 比较项目 | KMeans | GMM-full |
|---|---|---|
| 聚类依据 | 到中心的距离 | 概率密度与后验归属 |
| 每个簇的描述 | 一个中心 | 中心、协方差和混合权重 |
| 簇形状假设 | 近似球形、尺度接近 | 可为不同方向和大小的椭球 |
| 特征相关性 | 不显式刻画 | 可由全协方差刻画 |
| 标签形式 | 硬标签 | 由后验概率产生标签 |
| 更适合的结构 | 规则、近球形簇 | 方向差异明显或分布形状复杂的簇 |

### 7.4 当前为什么使用 GMM-full-direct

当前细分输入来自闭集 embedding 空间及其数据集特定增强特征，并经过标准化和 PCA。Oracle 使用 96 维 embedding-statistics-PCA 表示，WiSig 使用 64 维 embedding-distance-PCA 表示。这类表示中的未知类别不一定呈规则球形分布，可能存在方向性、尺度差异和特征相关性。因此，GMM-full 比单纯依赖中心距离的 KMeans 更灵活。

采用 `direct` 设置，则避免在 GMM 已获得椭球状分布解释之后，再使用球面或余弦距离假设对未知簇标签进行强制重分配。

---

## 8. 不确定样本过滤

### 8.1 过滤目的

unknown cache 可能混入误拒的已知样本。如果直接将所有缓存样本强制分入未知簇，会降低细分纯度，也可能将已知类污染错误解释为未知发射源结构。

同时，GMM 过聚类产生的低后验置信样本或过小候选簇也可能对应边界混合区域。当前实现把这些样本统一标为不确定标签 `-1`，不强行纳入某个未知簇。

### 8.2 当前判断规则

当前包含三类过滤：

1. 后验低置信过滤：取 GMM 后验概率最大值作为样本置信度；Oracle 将最低 10% 标为不确定，WiSig 将最低 15% 标为不确定。
2. 小簇剔除：Oracle 中候选簇少于 800 个样本、WiSig 中候选簇少于 50 个样本时，视为不稳定小簇。
3. 已知原型污染过滤：该规则按配置启用。Oracle 当前关闭，WiSig 当前启用。启用时，对每个缓存样本计算：

- 到最近未知聚类中心的余弦距离 \(d_u\)；
- 到最近已知原型锚点的余弦距离 \(d_k\)。

若满足：

\[
d_k \leq d_u + m,
\]

其中：

\[
m = \texttt{known\_reject\_margin}.
\]

当前 Oracle 的 `known_reject_margin = -1.0` 仅保留为兼容字段，实际通过 `use_known_prototype_anchors = False` 关闭额外的已知原型距离过滤；WiSig 设置为 0.10。

则将该样本标记为疑似已知污染，并赋予不确定标签：

\[
\widehat{y}_i=-1.
\]

未被过滤的样本保留 GMM 预测得到的未知簇标签。

### 8.3 准确性与覆盖率的权衡

| margin 设置倾向 | 细分结果变化 |
|---|---|
| 更大 | 过滤更多边界模糊样本，簇可能更纯，但 coverage 下降 |
| 更小 | 保留更多样本，coverage 提高，但污染风险上升 |

因此，细分结果不能只报告聚类质量指标，还应同时报告有效细分覆盖率或不确定样本比例。

---

### 8.4 类内子模态与冗余候选分量

类内子模态指的是：同一个真实未知发射源在特征空间里不一定只形成一个完全紧凑的团，它可能因为信号切片位置、瞬时幅相扰动、噪声、边界样本或 I/Q 统计差异，形成两个或多个局部小团。例如真实未知类 A 可能在特征空间里表现为 A1 和 A2 两个小团，但它们本质上仍属于同一个未知发射源。

如果真实未知类数为 6，而 GMM 也只拟合 6 个分量，那么一旦某个真实类 A 被拆成 A1 和 A2 两个分量，就会占掉两个候选分量。这样剩余 5 个分量要解释另外 5 个或更多真实未知类，某两个相邻未知类就可能被迫合并，导致 NMI、ARI 和匈牙利准确率下降。

因此当前方法把候选分量数统一写成 `K+m`：`K` 是协议给定的未知类数量，`m` 是冗余候选因子。`m` 不是新的未知类数量，而是全局冗余缓冲容量，用于吸收边界样本、低置信样本或短暂出现的类内子模态。随后通过后验置信度和最小簇规模过滤，把冗余或不稳定候选分量标为不确定样本 `-1`，最终有效未知簇数仍回到协议规定的 `K`。

本文只在小范围 `m∈{0,1,2,3}` 中做离线敏感性分析。候选最终有效簇数必须等于 K；覆盖率修正后的 NMI、ARI、Purity、Hungarian Accuracy 均值提升超过 1 个百分点，才接受更大的 `m`。Oracle 因此选择 `m=2`：它相对 `m=1` 提升约 3.31 个百分点，而 `m=3` 不再提升。WiSig 选择 `m=0`，因为额外冗余分量会产生超过 K 的最终有效簇。该规则依赖离线真实未知标签，不用于在线判别，论文中应明确披露。

---

## 9. 聚类数设置与论文声明边界

当前主实验根据数据协议固定未知簇数：

| 数据集 | 真实未知发射源数量 | 当前细分簇数 |
|---|---:|---:|
| Oracle | 6 | \(K=6\) |
| WiSig | 12 | \(K=12\) |

代码保留了自动搜索 \(K\) 的能力，可以综合轮廓系数、Davies--Bouldin 指数、Calinski--Harabasz 指数、跨种子稳定性、不确定率惩罚与未知中心靠近已知原型的惩罚选择聚类数；但当前主结果采用固定 \(K\)。

因此，论文中建议表述为：

> PCBM performs post-rejection unknown-emitter subdivision under the protocol-specified number of unseen emitters.

当前不宜表述为：

> PCBM automatically discovers an arbitrary number of unseen emitter identities.

---

## 10. Oracle 与 WiSig 当前配置

| 设置项 | Oracle | WiSig |
|---|---:|---:|
| 特征模式 | `embedding_stats` | `embedding_stats` |
| embedding 维度 | 128 | 128 |
| PCA 维度 | 96 | 96 |
| 聚类后端 | `gmm_full_direct` | `gmm_full_direct` |
| `known_reject_margin` | -1.0 | -1.0 |
| 目标未知簇数 | 6 | 12 |
| 实际拟合候选簇数 | 8 | 12 |
| 过聚类冗余候选数 `m` | 2 | 0 |
| 后验低置信过滤分位数 | 0.10 | 0.10 |
| 小簇剔除阈值 | 800 | 160 |
| 是否启用已知原型锚点 | 否 | 否 |

两组数据采用同一套 `embedding_stats + PCA96 + unknown-only GMM-full-direct` 细分口径。区别只体现在冗余候选因子 `m`：Oracle 使用 `m=2` 吸收不稳定样本，WiSig 当前数据更干净，额外过聚类会拆出类内子模态，因此使用 `m=0`。

---

## 11. 当前细分结果快照

### 11.1 Oracle

| 指标 | 当前结果 |
|---|---:|
| Normalized mutual information (NMI) | 0.999112 |
| Adjusted Rand index (ARI) | 0.999469 |
| Purity | 0.999801 |
| Hungarian accuracy | 0.999801 |
| Resolved number of clusters | 6 |
| Fit number of clusters | 8 |
| Uncertain sample size | 3147 |
| Coverage of total test unknown samples | 0.839458 |

Oracle 上的结果说明六类未知发射源具有较明显的可分结构，但污染过滤和边界重叠仍对有效覆盖率造成影响。

### 11.2 WiSig

| 指标 | 当前结果 |
|---|---:|
| Normalized mutual information (NMI) | 0.998125 |
| Adjusted Rand index (ARI) | 0.998637 |
| Purity | 0.999375 |
| Hungarian accuracy | 0.999375 |
| Resolved number of clusters | 12 |
| Fit number of clusters | 12 |
| Uncertain sample size | 0 |
| Coverage of total test unknown samples | 1.000000 |

WiSig 上的细分结构接近完全可分，说明当前 `embedding_stats + PCA96` 对该数据协议中的未知类别差异保留充分。额外 `K+2` 过聚类会把部分干净未知类拆成类内子模态，因此未作为正式 WiSig 设置。

---

## 12. 与此前版本相比的关键更新

| 项目 | 旧口径 | 当前口径 |
|---|---|---|
| Oracle 细分特征 | `embedding_distance` | `embedding_stats` |
| WiSig 细分特征 | `score_distance` / `embedding_distance` | `embedding_stats` |
| Oracle PCA 维度 | 32 / 64 | 96 |
| WiSig PCA 维度 | 13 / 64 | 96 |
| Oracle 目标簇数 | 曾为 5 | 6，与协议对齐 |
| WiSig 后端 | 曾为 `gmm` | `gmm_full_direct` |
| 跨数据集流程 | 设置不统一 | `embedding_stats + PCA96 + unknown-only GMM-full-direct` 统一 |
| 过滤机制表述 | 不够清晰 | 明确为后验低置信、小簇和已知原型污染过滤 |

---

## 13. 可用于论文的方法概述表述

> After unknown-emitter rejection, the rejected samples are retained in an unknown cache for further subdivision. PCBM constructs subdivision features from the closed-set embedding augmented with per-sample I/Q statistical descriptors. The resulting unknown-cache features are standardized and projected through PCA, and a full-covariance Gaussian mixture model directly predicts candidate unknown groups. For Oracle, two redundant candidate components are used to absorb low-confidence or unstable submodes before pruning; for WiSig, the GMM is fitted directly with the protocol-specified number of unknown emitters because additional over-clustering tends to split already clean classes into intra-class submodes. Low-posterior-confidence samples or unstable tiny components are marked as uncertain rather than forcibly assigned to an unknown cluster.

---

## 14. 一句话总结

当前细分模块的核心不是“把 unknown 样本直接交给聚类器”，而是：

> 将拒识样本表示为数据集特定的深度增强特征，在标准化与 PCA 后的空间中通过全协方差 GMM 划分潜在未知发射源，并用低置信、小簇和按配置启用的已知原型锚点过滤减少不稳定样本。
