# 对比方法适配运行报告

本报告由 `Comparison method/adapted_baselines/run_comparison.py` 生成。指标已经拆分为“开放集拒识”和“未知类细分”两张口径一致的表，指标名与本方法保持一致，不使用 mean/std 统计列。

## 开放集拒识指标

本表字段与本方法 `open_set_metrics.json` 的核心指标保持一致。

| dataset | method | task | seed | overall_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | known_accuracy | unknown_precision | unknown_recall | known_fpr_as_unknown | unknown_false_accept_rate | auroc | fpr95 | oscr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | HyDRA | open_set_rejection | 42 | 0.098485 | 0.024508 | 0.098485 | 0.028735 | 0.028735 | 0.095833 | 0.176471 | 0.125000 | 0.058333 | 0.875000 | 0.544097 | 0.983333 | 0.053038 |
| oracle_kri16_demod | HyperRSI | open_set_rejection | 42 | 0.106061 | 0.118558 | 0.106061 | 0.106695 | 0.106695 | 0.091667 | 0.333333 | 0.250000 | 0.050000 | 0.750000 | 0.643403 | 0.941667 | 0.059549 |
| wisig_singleday_osr_k16_u12 | HyDRA | open_set_rejection | 42 | 0.068627 | 0.012119 | 0.068627 | 0.016039 | 0.016039 | 0.062500 | 0.142857 | 0.166667 | 0.062500 | 0.833333 | 0.619954 | 0.770833 | 0.057292 |
| wisig_singleday_osr_k16_u12 | HyperRSI | open_set_rejection | 42 | 0.843137 | 0.853473 | 0.843137 | 0.846319 | 0.846319 | 0.869792 | 0.322581 | 0.416667 | 0.054688 | 0.583333 | 0.837565 | 0.705729 | 0.757161 |

## 未知类细分指标

本表字段与本方法 `unknown_subdivision_metrics.json` 的主要评价指标保持一致。

| dataset | method | task | seed | nmi | ari | purity | hungarian_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | OpenRFI-style Prototype Grouping | unknown_subdivision | 42 | 0.072861 | 0.010133 | 0.270833 | 0.256944 |
| wisig_singleday_osr_k16_u12 | OpenRFI-style Prototype Grouping | unknown_subdivision | 42 | 0.217762 | 0.047772 | 0.253472 | 0.239583 |

## 逐次结果文件

- 开放集拒识逐次指标：`open_set_per_seed_results.csv`
- 开放集拒识主指标表：`open_set_summary_results.csv`
- 未知类细分逐次指标：`unknown_subdivision_per_seed_results.csv`
- 未知类细分主指标表：`unknown_subdivision_summary_results.csv`

## 本次运行规模

- 开放集拒识逐次记录：4
- 未知类细分逐次记录：2
