# 对比方法适配运行报告

本报告由 `Comparison method/adapted_baselines/run_comparison.py` 生成。指标已经拆分为“开放集拒识”和“未知类细分”两张口径一致的表，指标名与本方法保持一致，不使用 mean/std 统计列。

## 开放集拒识指标

本表字段与本方法 `open_set_metrics.json` 的核心指标保持一致。

| dataset | method | task | seed | overall_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | known_accuracy | unknown_precision | unknown_recall | known_fpr_as_unknown | unknown_false_accept_rate | auroc | fpr95 | oscr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | HyDRA | open_set_rejection | 42 | 0.579937 | 0.745633 | 0.899322 | 0.737291 | 0.653020 | 0.943375 | 0.960904 | 0.458792 | 0.056000 | 0.541208 | 0.862237 | 0.572500 | 0.845436 |
| oracle_kri16_demod | HyperRSI | open_set_rejection | 42 | 0.823219 | 0.840882 | 0.925186 | 0.842857 | 0.862959 | 0.939250 | 0.977825 | 0.784542 | 0.053375 | 0.215458 | 0.960489 | 0.140250 | 0.925772 |
| wisig_singleday_osr_k16_u12 | HyDRA | open_set_rejection | 42 | 0.895312 | 0.909702 | 0.948456 | 0.889685 | 0.921012 | 0.952734 | 0.986109 | 0.880000 | 0.046484 | 0.120000 | 0.927425 | 0.812500 | 0.887231 |
| wisig_singleday_osr_k16_u12 | HyperRSI | open_set_rejection | 42 | 0.990789 | 0.999322 | 0.958824 | 0.977636 | 0.990495 | 0.956250 | 0.988468 | 1.000000 | 0.043750 | 0.000000 | 0.998480 | 0.007031 | 0.956217 |

## 未知类细分指标

本表字段与本方法 `unknown_subdivision_metrics.json` 的主要评价指标保持一致。

| dataset | method | task | seed | nmi | ari | purity | hungarian_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | OpenRFI-style Prototype Grouping | unknown_subdivision | 42 | 0.734032 | 0.585648 | 0.751167 | 0.695042 |
| wisig_singleday_osr_k16_u12 | OpenRFI-style Prototype Grouping | unknown_subdivision | 42 | 0.805228 | 0.597835 | 0.677917 | 0.636563 |

## 逐次结果文件

- 开放集拒识逐次指标：`open_set_per_seed_results.csv`
- 开放集拒识主指标表：`open_set_summary_results.csv`
- 未知类细分逐次指标：`unknown_subdivision_per_seed_results.csv`
- 未知类细分主指标表：`unknown_subdivision_summary_results.csv`

## 本次运行规模

- 开放集拒识逐次记录：4
- 未知类细分逐次记录：2
