# 对比方法适配运行报告

本报告由 `Comparison method/adapted_baselines/run_comparison.py` 生成。指标已经拆分为“开放集拒识”和“未知类细分”两张口径一致的表，指标名与本方法保持一致，不使用 mean/std 统计列。

## 开放集拒识指标

本表字段与本方法 `open_set_metrics.json` 的核心指标保持一致。

| dataset | method | task | seed | overall_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | known_accuracy | unknown_precision | unknown_recall | known_fpr_as_unknown | unknown_false_accept_rate | auroc | fpr95 | oscr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | ARPL | open_set_rejection | 42 | 0.885594 | 0.867344 | 0.892761 | 0.854296 | 0.904053 | 0.893750 | 0.966784 | 0.882875 | 0.091000 | 0.117125 | 0.932765 | 0.632500 | 0.841156 |
| oracle_kri16_demod | HyDRA | open_set_rejection | 42 | 0.824688 | 0.856642 | 0.875492 | 0.826454 | 0.860415 | 0.882500 | 0.953673 | 0.805417 | 0.117375 | 0.194583 | 0.904814 | 0.478375 | 0.838093 |
| oracle_kri16_demod | HyperRSI | open_set_rejection | 42 | 0.894469 | 0.888491 | 0.891860 | 0.866924 | 0.911085 | 0.891500 | 0.962643 | 0.895458 | 0.104250 | 0.104542 | 0.951477 | 0.186750 | 0.878105 |
| oracle_kri16_demod | OpenMax | open_set_rejection | 42 | 0.864000 | 0.860297 | 0.804133 | 0.804256 | 0.879224 | 0.795875 | 0.929626 | 0.886708 | 0.201375 | 0.113292 | 0.900686 | 0.318625 | 0.779745 |
| oracle_kri16_demod | OpenRFI | open_set_rejection | 42 | 0.934156 | 0.938944 | 0.903811 | 0.905725 | 0.941914 | 0.899625 | 0.965828 | 0.945667 | 0.100375 | 0.054333 | 0.969528 | 0.146750 | 0.878649 |
| oracle_kri16_demod | Softmax | open_set_rejection | 42 | 0.635844 | 0.718190 | 0.897970 | 0.721127 | 0.698599 | 0.934125 | 0.967025 | 0.536417 | 0.054875 | 0.463583 | 0.900906 | 0.464875 | 0.876319 |
| wisig_singleday_osr_k16_u12 | ARPL | open_set_rejection | 42 | 0.978865 | 0.998466 | 0.905515 | 0.948643 | 0.978258 | 0.899609 | 0.973927 | 1.000000 | 0.100391 | 0.000000 | 0.992654 | 0.021875 | 0.899579 |
| wisig_singleday_osr_k16_u12 | HyDRA | open_set_rejection | 42 | 0.979030 | 0.998478 | 0.906250 | 0.946136 | 0.977776 | 0.900391 | 0.974125 | 1.000000 | 0.099609 | 0.000000 | 0.998989 | 0.005078 | 0.900361 |
| wisig_singleday_osr_k16_u12 | HyperRSI | open_set_rejection | 42 | 0.977714 | 0.998385 | 0.900368 | 0.939272 | 0.975611 | 0.894141 | 0.972546 | 1.000000 | 0.105859 | 0.000000 | 0.998671 | 0.003125 | 0.894112 |
| wisig_singleday_osr_k16_u12 | OpenMax | open_set_rejection | 42 | 0.977138 | 0.982505 | 0.905025 | 0.928624 | 0.972932 | 0.899219 | 0.973775 | 0.997917 | 0.100781 | 0.002083 | 0.988474 | 0.025781 | 0.899039 |
| wisig_singleday_osr_k16_u12 | OpenRFI | open_set_rejection | 42 | 0.919408 | 0.950059 | 0.901808 | 0.891645 | 0.935144 | 0.900391 | 0.972070 | 0.924479 | 0.099609 | 0.075521 | 0.978104 | 0.201563 | 0.888071 |
| wisig_singleday_osr_k16_u12 | Softmax | open_set_rejection | 42 | 0.916036 | 0.927320 | 0.950000 | 0.916164 | 0.938259 | 0.952734 | 0.986395 | 0.906250 | 0.046875 | 0.093750 | 0.940071 | 0.688672 | 0.899614 |

## 未知类细分指标

本表字段与本方法 `unknown_subdivision_metrics.json` 的主要评价指标保持一致。

| dataset | method | task | seed | nmi | ari | purity | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | OpenRFI | unknown_subdivision | 42 | 0.931928 | 0.909938 | 0.982990 | 0.930196 | 0.850000 |
| wisig_singleday_osr_k16_u12 | OpenRFI | unknown_subdivision | 42 | 0.981091 | 0.967620 | 0.990986 | 0.962254 | 0.924479 |

## 逐次结果文件

- 开放集拒识逐次指标：`open_set_per_seed_results.csv`
- 开放集拒识主指标表：`open_set_summary_results.csv`
- 未知类细分逐次指标：`unknown_subdivision_per_seed_results.csv`
- 未知类细分主指标表：`unknown_subdivision_summary_results.csv`

## 本次运行规模

- 开放集拒识逐次记录：12
- 未知类细分逐次记录：2
