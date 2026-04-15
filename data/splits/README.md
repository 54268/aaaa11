# split 文件说明

这个目录专门存放可复现的数据划分文件。

## 设计原则

- 每个实验划分都落成一个独立 JSON
- JSON 中明确记录 known classes 和 unknown classes
- 固定随机种子
- 记录 train / val 比例
- 对 WiSig 额外记录受控条件池过滤条件

## 当前目录

- `wisig/`
  WiSig 主实验 split
- `oracle/`
  Oracle 主实验 split

## WiSig

当前主实验使用：

- 数据源：`SingleDay`
- receiver：`1-1`
- date：`2021_03_23`
- equalized：`0`

目录：

- `wisig/single_day_rx1_eq0/`

其中：

- `candidate_class_counts.csv`
  记录当前受控条件池内每个 transmitter 的可用样本数
- `split_index.json`
  记录同一池里生成的多组 K/U 划分
- `wisig_single_day_rx1_eq0_k16_u*.json`
  具体某一组 OSR 实验划分

## Oracle

- `oracle/oracle_k10_u6_seed42.json`
  Oracle 主实验划分
