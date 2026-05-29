# 消融实验 1：仅普通边缘样本外推

## 实验目的

验证“关键边界样本筛选”这一创新点是否有效。

## 改动内容

- 去掉关键边界样本的筛选
- `top_m = 0`
- 只保留普通边缘样本
- 伪未知样本仅沿局部法向外推生成

## 与主实验相比

主实验：

- 普通边缘样本 + 关键边界样本
- 两路伪未知生成

本消融：

- 只有普通边缘样本
- 没有关键边界样本分支

## 运行

```bash
python experiments/ablations/run_ablation.py --config experiments/ablations/01_仅普通边缘样本外推/config.yaml
```

