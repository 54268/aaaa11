# 对比方法适配计划

## 目标

把 `raw/` 中适合当前任务的论文代码统一适配到当前项目的 WiSig / Oracle 协议上，形成可直接用于论文第四章的对比方法集合。

## 已纳入

- Softmax
- OpenMax
- HyperRSI
- HyDRA
- OpenRFI
- ARPL

## 不纳入主对比

- `meta-open-master`
- `NS-RFF-main`

`meta-open-master` 更偏少样本元学习式开放集设定，和当前项目使用的标准已知类 / 未知类划分协议不一致，而且仓库里的 `openmany` 也没有完整实现。

`NS-RFF-main` 是 ZigBee 开放集认证框架，主要输出 pairwise 特征距离的 ROC/AUC/EER，不直接对应当前多已知类分类 + 未知类拒识的总体准确率口径，因此不作为主表对比方法。

## 适配原则

1. 所有方法都统一接入 WiSig 和 Oracle 两个数据集。
2. 训练、验证、测试划分沿用主项目已有协议。
3. 对比表里的指标名与主方法保持一致。
4. 根目录 `final_comparison_tables.md` 只保留最核心的拒识和细分指标。
5. 没有原生未知类细分模块的方法，只进入拒识表，不强行补细分指标。

## 方法定位

- Softmax：最基础的闭集 CNN + 阈值拒识。
- OpenMax：经典 OpenMax / EVT 拒识。
- HyperRSI：保留 paper-style 512 维 hypersphere / CosFace / GPD 模块，同时在当前 256 点协议主表中使用更稳定的紧凑适配版。
- HyDRA：CNN + Transformer 风格拒识。
- OpenRFI：保留 RoInformer 风格表示和原型分组细分。
- ARPL：保留 reciprocal-point 训练，并用 EVT / OpenMax 风格校准 logits 做拒识。

## 当前结果输出

统一通过：

```text
Comparison method/adapted_baselines/run_comparison.py
```

生成：

- `Comparison method/adapted_results/...`
- 项目根目录下的 `final_comparison_tables.md`
