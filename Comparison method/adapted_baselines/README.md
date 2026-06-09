# 对比方法适配说明

本目录用于把 `Comparison method/raw/` 里的对比方法统一适配到当前项目的 WiSig / Oracle 协议上。这里的代码只读取主项目已有数据，不修改主项目的 `outputs/`、`settings/`、`functions/` 或原始数据目录。

## 已纳入的方法

- `Softmax`
- `OpenMax`
- `HyperRSI`
- `HyDRA`
- `OpenRFI`
- `ARPL`
- `PCBM (ours)`

其中：

- `Softmax / OpenMax / HyperRSI / HyDRA / ARPL` 只进入开放集拒识对比；
- `OpenRFI` 同时进入拒识和未知类细分对比；
- `PCBM` 同时进入拒识和未知类细分对比。

## 未纳入的方法

- `meta-open-master`

它更偏少样本元学习式开放集设置，核心流程是 episodic few-shot，而且仓库里的 `openmany` 并未完整实现，不适合作为当前 WiSig / Oracle 标准协议下的主对比方法。

## ARPL 说明

当前正式结果中的 `ARPL` 保留了 reciprocal-point 训练思想，并使用 EVT / OpenMax 风格的校准做拒识。
代码里也保留了 confusing-sample 分支，但这一路在当前 1D I/Q 任务上会明显牺牲已知类准确率，因此不作为正式表默认结果。

## HyperRSI 说明

`src/models.py` 里同时保留了两套实现：

- `HyperRSIClassifier`：更接近论文的 512 维 hypersphere + CosFace + GPD 头。
- `CompactHyperRSIClassifier`：当前 Oracle / WiSig 协议下用于正式对比的紧凑适配版。

原因很简单：论文原始 Wi-Fi 设置是 6000 点输入，而我们当前协议是 256 点截片，完整 paper 头在这里并不稳定；因此正式表使用紧凑版，完整版保留给代码审查和后续复现实验。

## OpenRFI 补全说明

OpenRFI 原仓库依赖的不是一个单独线性分类头，而是一条更完整的表示学习链路。当前适配版已经补齐了这些关键模块：

- SimCLR 风格的两类射频增强：高斯噪声抖动和信号帧重排；
- RoInformer 风格的残差 1-D CNN + Transformer 编码器；
- ArcMargin 分类头，用于把已知类拉开间隔；
- 可学习原型、group mask 和原型/分组/熵正则项；
- 旧 checkpoint 与新结构不兼容时自动重训，不再直接报错退出。

Oracle 上的 OpenRFI 拒识阈值现在使用更合理的 `0.10`，避免把大量已知样本误拒；OpenMax 的 Oracle 阈值保留在 `0.20`，用于平衡 known accuracy 和 unknown recall。

## NS-RFF 说明

`NS-RFF` 更偏 ZigBee 认证/验证任务，核心是同步补偿 + 距离分布评估，不是当前 WiSig / Oracle 的多类开放集识别基线，因此未纳入主对比表。

## 目录说明

- `run_comparison.py`：唯一运行入口，文件顶部直接改常用参数。
- `src/data_io.py`：读取 WiSig / Oracle 的 `npz` 数据与协议信息。
- `src/models.py`：对比方法的统一模型实现。
- `src/trainers.py`：训练、特征提取、拒识、细分流程。
- `src/metrics.py`：开放集和聚类指标。
- `src/reporting.py`：CSV / JSON / Markdown 汇总。

## 输出

所有对比结果都写入：

```text
Comparison method/adapted_results/
```

其中最重要的汇总文件包括：

- `open_set_summary_results.csv`
- `unknown_subdivision_summary_results.csv`
- `comparison_report.md`
- 项目根目录下的 `final_comparison_tables.md`

根目录的 `final_comparison_tables.md` 只保留四个拒识指标、三个细分指标和细分覆盖率，方便论文直接引用。

## 运行方式

直接运行：

```text
D:\Anaconda3\envs\pytorch\python.exe "D:\learn_pytorch\笔记\方案\os_sei_code\Comparison method\adapted_baselines\run_comparison.py"
```

默认使用 `formal` 模式。若只想快速检查流程，把 `run_comparison.py` 顶部的 `RUN_MODE` 改成 `smoke` 即可。
