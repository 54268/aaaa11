
- WiSig 主实验：不再把官网四个 compact subset 无差别合并，也不把四个子集分别当主实验；而是从 `SingleDay` 中抽取一个受控条件数据池，再做可复现的 known / unknown 划分。
- Oracle 主实验：独立训练、独立评测，不和 WiSig 混成一个训练集。

## 推荐入口

### 运行 WiSig 主实验

```bash
python run_wisig_demo.py
```

默认会跑：

- `configs/wisig_singleday_osr_k16_u12.yaml`

对应的数据池条件是：

- 数据来源：`WiSig SingleDay`
- 固定接收机：`1-1`
- 固定采集日期：`2021_03_23`
- 固定 equalized：`0`
- 类别定义：`transmitter identity`

### 运行 Oracle 主实验

```bash
python run_oracle_demo.py
```

默认会跑：

- `configs/oracle_osr_main.yaml`

### 通用入口

```bash
python run_osr_experiment.py --config configs/wisig_singleday_osr_k16_u12.yaml
python run_osr_experiment.py --config configs/oracle_osr_main.yaml
```

## 数据划分与 split 文件

WiSig 和 Oracle 都已经改成 split 文件驱动：

- WiSig split：`data/splits/wisig/single_day_rx1_eq0/`
- Oracle split：`data/splits/oracle/`

这些 split 文件会明确记录：

- known classes 列表
- unknown classes 列表
- 随机种子
- 训练/验证比例
- 受控条件池的过滤条件
- openness

## 结果与图表

每次实验的输出在：

- `outputs/<实验名>/`

自动生成的图在：

- `figures/<实验名>/`

结果汇总快捷入口：

- `RESULT_SUMMARY.md`：总入口，不再保存单次实验结果
- `RESULT_SUMMARY_WISIG.md`：WiSig 最近一次运行结果汇总
- `RESULT_SUMMARY_ORACLE.md`：Oracle 最近一次运行结果汇总

## 结果表与 openness 曲线

如果你已经跑完多组 WiSig 主实验，比如 `K=16, U=4/8/12`，可以用下面的命令汇总结果并画 openness 曲线：

```bash
python tools/export_osr_results_table.py ^
  --configs configs/wisig_singleday_osr_k16_u4.yaml configs/wisig_singleday_osr_k16_u8.yaml configs/wisig_singleday_osr_k16_u12.yaml ^
  --output-dir outputs/tables/wisig_singleday_main
```

## 重点说明

- 本工程是纯 OSR，不包含 class-incremental 训练流程。
- known / unknown 是按类别严格隔离的。
- unknown 类不参与训练。
- WiSig 和 Oracle 分别独立训练与评测。
- 跨数据集泛化评测单独用 `tools/evaluate_cross_dataset.py`，作为补充实验，不替代主实验。

更详细的设计说明请看：

- `纯OSR实验设计与入口说明.md`
- `模型演进与仓库适配说明.md`
