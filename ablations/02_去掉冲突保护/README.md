# 消融实验 2：去掉冲突保护

## 实验目的

验证关键边界样本生成伪未知时，“冲突保护”这一设计是否有效。

## 改动内容

- 保留关键边界样本筛选
- 保留关键边界样本分支
- 在关键边界样本生成方向中，去掉冲突保护
- `enable_conflict_protection = false`

## 与主实验相比

主实验：

- 当本类法向与异类排斥方向发生冲突时，会先做冲突保护，再形成最终生成方向

本消融：

- 不做冲突保护
- 直接使用混合方向生成关键伪未知样本

## 运行

```bash
python experiments/ablations/run_ablation.py --config experiments/ablations/02_去掉冲突保护/config.yaml
```

