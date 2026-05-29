# 消融实验 3：去掉冲突保护并去掉原型校准

## 实验目的

进一步验证“冲突保护”的效果是否被后续原型距离校准掩盖。

## 改动内容

- 保留关键边界样本筛选
- 去掉冲突保护
- 最终校准阶段去掉原型距离分支
- 只保留 OpenMax 分支参与最终拒识

## 运行

```bash
python experiments/ablations/run_ablation.py --config experiments/ablations/03_去掉冲突保护并去掉原型校准/config.yaml
```

