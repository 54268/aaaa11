# 对比方法适配计划

## 目标

把 `raw/` 中保留的三篇论文代码适配为可在当前项目 WiSig 与 Oracle 数据划分上运行的对比方法。所有新增代码、日志、缓存和结果只放在 `Comparison method/` 内，不修改主方法目录。

## 目录

- `raw/`：原论文代码，只读保留。
- `adapted_baselines/`：统一适配入口和辅助代码。
- `adapted_results/`：对比方法运行输出。

## 方法定位

- HyperRSI：作为开放集拒识对比方法，采用 hypersphere embedding、类中心余弦相似度和验证集已知类分位阈值。
- HyDRA：作为开放集拒识对比方法，采用原代码中的 CNN + Transformer 主体，并用 softmax 置信度阈值完成拒识。
- OpenRFI：作为未知类细分对比方法，采用 OpenRFI 的原型分组思想，对未知样本特征进行 prototype grouping 和 spectral clustering。

## 公平设置

- 读取当前项目已经生成的 `data/processed/.../*.npz`，不重新定义数据划分。
- WiSig 使用 `data/processed/wisig_singleday_osr_k16_u12`。
- Oracle 使用 `data/processed/oracle_kri16_demod`。
- 每个方法都使用相同的 `train_known`、`val_known`、`test_known`、`test_unknown`。
- 拒识指标统一计算 overall accuracy、known accuracy、unknown recall、macro F1、AUROC、FPR95 等。
- 细分指标统一计算 NMI、ARI、Purity、Hungarian Accuracy。

## 执行方式

运行入口为：

```text
Comparison method/adapted_baselines/run_comparison.py
```

入口文件顶部提供中文注释参数。默认先运行 `smoke` 小样本流程检查代码能否跑通，正式实验时把 `RUN_MODE` 改为 `formal`。
