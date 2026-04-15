# 数据目录说明

## 1. 目录作用

### `raw/`

- 存放原始下载数据
- 这里的数据尽量不改动，作为可追溯原始输入

### `processed/`

- 存放统一预处理后的训练/验证/测试数据
- 当前统一格式为：
  - `train_known.npz`
  - `val_known.npz`
  - `test_known.npz`
  - `test_unknown.npz`

### `logs/`

- 存放下载探测、网页抓取和排障日志

## 2. WiSig 数据状态

WiSig 官网 compact subset 一共有四个：

- `ManyRx`
- `ManyTx`
- `ManySig`
- `SingleDay`

当前本地已经下载的是：

- `ManySig`
- `SingleDay`

当前自动下载未成功的是：

- `ManyRx`
- `ManyTx`

对应文件：

- `raw/wisig/ManySig.pkl`
- `raw/wisig/SingleDay.pkl`

解包后文件：

- `raw/wisig/ManySig_unpacked/ManySig.pkl`

处理后目录：

- `processed/wisig_manysig/`

当前已经完成预处理的是：

- `ManySig`

处理后文件含义：

- `train_known.npz`：已知类训练集
- `val_known.npz`：已知类验证集
- `test_known.npz`：已知类测试集
- `test_unknown.npz`：未知类测试集
- `dataset_summary.json`：类别划分和样本数量摘要

## 3. Oracle 数据状态

当前本地已经接入的是你手动下载的第二个数据集：

- `raw/oracle/KRI-16IQImbalances-DemodulatedData/`

这个目录下的文件成对出现：

- `*.sigmf-meta`：元数据
- `*.sigmf-data`：原始复数 I/Q 数据

处理后目录：

- `processed/oracle_kri16_demod/`

处理后文件含义：

- `train_known.npz`：已知类训练集
- `val_known.npz`：已知类验证集
- `test_known.npz`：已知类测试集
- `test_unknown.npz`：未知类测试集
- `dataset_summary.json`：类别划分和样本数量摘要
- `record_manifest.json`：原始文件到处理结果的映射表

## 4. 常用命令

### 预处理 WiSig ManySig

```bash
python tools/prepare_wisig_compact.py --config configs/wisig_manysig.yaml
```

### 预处理 Oracle

```bash
python tools/prepare_oracle_sigmf.py --config configs/oracle_sigmf.yaml
```
