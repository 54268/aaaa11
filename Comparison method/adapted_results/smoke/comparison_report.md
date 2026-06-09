# 对比方法适配运行报告

本报告由 `Comparison method/adapted_baselines/run_comparison.py` 生成。指标已经拆分为“开放集拒识”和“未知类细分”两张口径一致的表，指标名与本方法保持一致，不使用 mean/std 统计列。

## 开放集拒识指标

本表字段与本方法 `open_set_metrics.json` 的核心指标保持一致。

| dataset | method | task | seed | overall_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | known_accuracy | unknown_precision | unknown_recall | known_fpr_as_unknown | unknown_false_accept_rate | auroc | fpr95 | oscr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | HyDRA | open_set_rejection | 42 | 0.098485 | 0.024508 | 0.098485 | 0.028735 | 0.028735 | 0.095833 | 0.176471 | 0.125000 | 0.058333 | 0.875000 | 0.544444 | 0.979167 | 0.053038 |
| oracle_kri16_demod | HyperRSI | open_set_rejection | 42 | 0.109848 | 0.093398 | 0.109848 | 0.096805 | 0.096805 | 0.120833 | 0.000000 | 0.000000 | 0.058333 | 1.000000 | 0.553993 | 0.916667 | 0.077778 |
| oracle_kri16_demod | OpenMax | open_set_rejection | 42 | 0.075758 | 0.035453 | 0.075758 | 0.038946 | 0.038946 | 0.075000 | 0.181818 | 0.083333 | 0.037500 | 0.916667 | 0.458854 | 0.929167 | 0.033941 |
| oracle_kri16_demod | OpenRFI | open_set_rejection | 42 | 0.102273 | 0.040314 | 0.102273 | 0.052700 | 0.052700 | 0.104167 | 0.066667 | 0.083333 | 0.116667 | 0.916667 | 0.496701 | 0.866667 | 0.051649 |
| oracle_kri16_demod | Softmax | open_set_rejection | 42 | 0.087121 | 0.008397 | 0.087121 | 0.015318 | 0.015318 | 0.095833 | 0.000000 | 0.000000 | 0.062500 | 1.000000 | 0.481684 | 0.958333 | 0.051128 |
| wisig_singleday_osr_k16_u12 | HyDRA | open_set_rejection | 42 | 0.068627 | 0.012119 | 0.068627 | 0.016039 | 0.016039 | 0.062500 | 0.142857 | 0.166667 | 0.062500 | 0.833333 | 0.620009 | 0.770833 | 0.057020 |
| wisig_singleday_osr_k16_u12 | HyperRSI | open_set_rejection | 42 | 0.879902 | 0.905821 | 0.879902 | 0.882088 | 0.882088 | 0.893229 | 0.470588 | 0.666667 | 0.046875 | 0.333333 | 0.908420 | 0.388021 | 0.824002 |
| wisig_singleday_osr_k16_u12 | OpenMax | open_set_rejection | 42 | 0.075980 | 0.049219 | 0.075980 | 0.051562 | 0.051562 | 0.075521 | 0.090909 | 0.083333 | 0.052083 | 0.916667 | 0.546007 | 0.864583 | 0.046224 |
| wisig_singleday_osr_k16_u12 | OpenRFI | open_set_rejection | 42 | 0.085784 | 0.053728 | 0.085784 | 0.048133 | 0.048133 | 0.080729 | 0.133333 | 0.166667 | 0.067708 | 0.833333 | 0.514106 | 0.869792 | 0.052192 |
| wisig_singleday_osr_k16_u12 | Softmax | open_set_rejection | 42 | 0.105392 | 0.073179 | 0.105392 | 0.040882 | 0.040882 | 0.111979 | 0.000000 | 0.000000 | 0.041667 | 1.000000 | 0.405382 | 0.940104 | 0.061144 |

## 未知类细分指标

本表字段与本方法 `unknown_subdivision_metrics.json` 的主要评价指标保持一致。

| dataset | method | task | seed | nmi | ari | purity | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_kri16_demod | HyDRA | unknown_subdivision | 42 | 0.733680 | 0.000000 | 1.000000 | 0.666667 | 0.125000 |
| oracle_kri16_demod | HyperRSI | unknown_subdivision | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_kri16_demod | OpenMax | unknown_subdivision | 42 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.083333 |
| oracle_kri16_demod | OpenRFI | unknown_subdivision | 42 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.083333 |
| oracle_kri16_demod | Softmax | unknown_subdivision | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wisig_singleday_osr_k16_u12 | HyDRA | unknown_subdivision | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.166667 |
| wisig_singleday_osr_k16_u12 | HyperRSI | unknown_subdivision | 42 | 0.839745 | 0.568656 | 0.875000 | 0.750000 | 0.666667 |
| wisig_singleday_osr_k16_u12 | OpenMax | unknown_subdivision | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.083333 |
| wisig_singleday_osr_k16_u12 | OpenRFI | unknown_subdivision | 42 | 0.666667 | -0.200000 | 0.750000 | 0.750000 | 0.166667 |
| wisig_singleday_osr_k16_u12 | Softmax | unknown_subdivision | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 逐次结果文件

- 开放集拒识逐次指标：`open_set_per_seed_results.csv`
- 开放集拒识主指标表：`open_set_summary_results.csv`
- 未知类细分逐次指标：`unknown_subdivision_per_seed_results.csv`
- 未知类细分主指标表：`unknown_subdivision_summary_results.csv`

## 本次运行规模

- 开放集拒识逐次记录：10
- 未知类细分逐次记录：10
