# 纯 OSR 实验设计与入口说明

## 1. 任务定位

这套工程现在明确按“纯开放集识别（Open-Set Recognition, OSR）”来组织，不引入以下流程：

- 不做类增量训练
- 不做少样本增量学习
- 不把 WiSig 和 Oracle 混成一个联合训练集

主实验与补充实验的边界如下：

- 主实验 1：WiSig 受控条件池 OSR
- 主实验 2：Oracle 独立 OSR
- 补充实验：跨数据集泛化评测

## 2. 当前目录设计

建议你把这个工程理解成下面这几个层次：

### 配置层

- `configs/`
  存放每个实验的 YAML 配置文件。

当前新增的主实验配置有：

- `configs/wisig_singleday_osr_k16_u4.yaml`
- `configs/wisig_singleday_osr_k16_u8.yaml`
- `configs/wisig_singleday_osr_k16_u12.yaml`
- `configs/oracle_osr_main.yaml`

### 划分层

- `data/splits/`
  存放可复现的数据划分文件。

其中：

- `data/splits/wisig/single_day_rx1_eq0/`
  是 WiSig 主实验的受控条件池划分目录
- `data/splits/oracle/`
  是 Oracle 主实验划分目录

### 预处理层

- `tools/create_wisig_controlled_splits.py`
  从 WiSig 受控条件池生成 split 文件
- `tools/create_oracle_osr_split.py`
  为 Oracle 生成 split 文件
- `tools/prepare_wisig_compact.py`
  按 split 文件把 WiSig 切成 train / val / test-known / test-unknown
- `tools/prepare_oracle_sigmf.py`
  按 split 文件把 Oracle 切成 train / val / test-known / test-unknown

### 训练与评测层

- `run_osr_experiment.py`
  统一总入口，执行“预处理 -> 训练 -> 边界挖掘 -> 伪未知生成 -> OpenMax -> 融合校准 -> 最终评测”
- `trainers/train_closed.py`
  已知类表征学习
- `tools/mine_boundary.py`
  边界样本挖掘
- `tools/generate_pseudo_unknown.py`
  普通边缘样本与关键边界样本的伪未知生成
- `tools/fit_openmax.py`
  OpenMax 拟合
- `tools/calibrate_fusion.py`
  融合校准
- `eval/evaluate_open_set.py`
  最终开放集评测

### 结果整理层

- `tools/export_osr_results_table.py`
  汇总多组实验结果，导出表格并绘制 openness-metric 曲线
- `tools/evaluate_cross_dataset.py`
  做跨数据集补充实验

## 3. WiSig 主实验的数据构造方案

### 3.1 为什么不用四个子集直接合并

不直接把 `ManySig / ManyTx / ManyRx / SingleDay` 无差别合并，原因是：

- 采集条件差异太大
- receiver 数量不同
- day 数量不同
- transmitter 数量和每类样本量不同
- 容易把“条件变化”学成类别特征

这不符合你要的“受控条件下自定义挑类”的纯 OSR 设定。

### 3.2 当前主实验的数据池

当前默认 WiSig 主实验使用：

- 数据源：`SingleDay`
- receiver：固定为 `1-1`
- day：固定为 `2021_03_23`
- equalized：固定为 `0`

这样做的好处是：

- day 已固定
- receiver 已固定
- equalized 状态已固定
- 类别只由 transmitter identity 决定
- 每个 transmitter 在该池内样本数一致，当前是每类 800 个样本

这正好符合你提出的“条件尽量一致、类别受控、划分干净”。

### 3.3 当前默认类划分

当前已经生成了三组 WiSig 主实验 split：

- `K=16, U=4`
- `K=16, U=8`
- `K=16, U=12`

对应 split 文件在：

- `data/splits/wisig/single_day_rx1_eq0/`

每个 split 文件会记录：

- known 类编号列表
- unknown 类编号列表
- openness
- 随机种子
- 数据池过滤条件
- 每类样本数

### 3.4 WiSig 划分原则

当前实现遵守下面几点：

- known / unknown 按 transmitter 类别严格隔离
- known 类再划分为 train / val / test-known
- unknown 类只进入 test-unknown
- unknown 不参与训练
- 每次划分都固定随机种子
- 每次实验都保留 split 文件

## 4. Oracle 主实验的数据构造方案

Oracle 不与 WiSig 混合训练，而是独立做一条 OSR 主实验。

当前 split 文件为：

- `data/splits/oracle/oracle_k10_u6_seed42.json`

含义是：

- 10 个已知类
- 6 个未知类
- 类别列表明确保存
- 训练/验证比例明确保存

Oracle 预处理后同样输出：

- `train_known.npz`
- `val_known.npz`
- `test_known.npz`
- `test_unknown.npz`
- `class_split.csv`
- `split_manifest.json`

## 5. 统一训练入口

### 5.1 单个实验

使用统一入口：

```bash
python run_osr_experiment.py --config configs/wisig_singleday_osr_k16_u12.yaml
python run_osr_experiment.py --config configs/oracle_osr_main.yaml
```

### 5.2 快捷入口

WiSig 主实验：

```bash
python run_wisig_demo.py
```

Oracle 主实验：

```bash
python run_oracle_demo.py
```

## 6. 训练流程

当前总流程如下：

1. 按 split 文件预处理数据
2. 用 CVCNN 提取 I/Q 嵌入特征
3. 用原型分类头学习已知类
4. 使用三段损失训练：
   - 基础损失
   - 角度损失
   - 原型损失
5. 在已知类特征空间中挖掘普通边缘样本和关键边界样本
6. 生成伪未知样本
7. 用验证集拟合 OpenMax
8. 用伪未知和验证已知集做融合校准
9. 在 test-known + test-unknown 上输出开放集指标与图表

## 7. 输出结果

每次实验至少会输出：

- 指标文件：`open_set_metrics.json`
- 结果摘要：`final_report.md`
- 混淆矩阵：`confusion_matrix.csv`
- 逐样本预测：`open_set_predictions.csv`
- 图表目录：`figures/<实验名>/`

图表包括：

- ROC 曲线
- PR 曲线
- confusion matrix
- unknown score 分布图
- openness-metric 曲线

## 8. 多组 openness 实验

如果要对 WiSig 主实验输出多组 openness 结果，可以依次跑：

```bash
python run_osr_experiment.py --config configs/wisig_singleday_osr_k16_u4.yaml
python run_osr_experiment.py --config configs/wisig_singleday_osr_k16_u8.yaml
python run_osr_experiment.py --config configs/wisig_singleday_osr_k16_u12.yaml
```

跑完后汇总：

```bash
python tools/export_osr_results_table.py ^
  --configs configs/wisig_singleday_osr_k16_u4.yaml configs/wisig_singleday_osr_k16_u8.yaml configs/wisig_singleday_osr_k16_u12.yaml ^
  --output-dir outputs/tables/wisig_singleday_main
```

它会输出：

- `osr_results_table.csv`
- `osr_results_table.md`
- `openness_metric_curve.png`

## 9. 跨数据集补充实验

跨数据集实验单独做，不替代主实验。

推荐协议：

- 源数据集 test-known 仍作为已知类测试
- 目标数据集全部样本视作未知类

当前脚本：

```bash
python tools/evaluate_cross_dataset.py ^
  --source-config configs/wisig_singleday_osr_k16_u12.yaml ^
  --target-config configs/oracle_osr_main.yaml
```

或反过来：

```bash
python tools/evaluate_cross_dataset.py ^
  --source-config configs/oracle_osr_main.yaml ^
  --target-config configs/wisig_singleday_osr_k16_u12.yaml
```

## 10. 当前结论

就你现在的要求来说，最关键的不是继续把 WiSig 四个官网子集乱合，而是：

- 先固定一个受控数据池
- 再做可复现挑类
- 再围绕 K / U / openness 做纯 OSR 实验

现在这套目录和入口，已经是按这个思路落下来的。
