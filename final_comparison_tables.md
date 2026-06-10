# 最终对比表

本文件为项目根目录下的精简版最终汇总，只保留对比方法最核心的拒识与细分指标。
开放集拒识只保留 `overall_accuracy`、`known_accuracy`、`macro_f1`、`auroc`；未知类细分保留 `nmi`、`ari`、`purity` 与覆盖率 `coverage_of_total_test_unknown`。

## oracle_kri16_demod 开放集拒识

| method | overall_accuracy | known_accuracy | macro_f1 | auroc |
| --- | --- | --- | --- | --- |
| Softmax | 0.635844 | 0.934125 | 0.721127 | 0.900906 |
| OpenMax | 0.864000 | 0.795875 | 0.804256 | 0.900686 |
| HyperRSI | 0.894469 | 0.891500 | 0.866924 | 0.951477 |
| HyDRA | 0.824688 | 0.882500 | 0.826454 | 0.904814 |
| OpenRFI | 0.934156 | 0.899625 | 0.905725 | 0.969528 |
| ARPL | 0.885594 | 0.893750 | 0.854296 | 0.932765 |
| PCBM (ours) | 0.966187 | 0.963625 | 0.936905 | 0.986836 |

## oracle_kri16_demod 未知类细分

| method | nmi | ari | purity | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- |
| OpenRFI | 0.931928 | 0.909938 | 0.982990 | 0.850000 |
| PCBM (ours) | 0.999112 | 0.999469 | 0.999801 | 0.839458 |

## wisig_singleday_osr_k16_u12 开放集拒识

| method | overall_accuracy | known_accuracy | macro_f1 | auroc |
| --- | --- | --- | --- | --- |
| Softmax | 0.916036 | 0.952734 | 0.916164 | 0.940071 |
| OpenMax | 0.977138 | 0.899219 | 0.928624 | 0.988474 |
| HyperRSI | 0.977714 | 0.894141 | 0.939272 | 0.998671 |
| HyDRA | 0.979030 | 0.900391 | 0.946136 | 0.998989 |
| OpenRFI | 0.919408 | 0.900391 | 0.891645 | 0.978104 |
| ARPL | 0.978865 | 0.899609 | 0.948643 | 0.992654 |
| PCBM (ours) | 0.997204 | 0.986719 | 0.993554 | 0.995952 |

## wisig_singleday_osr_k16_u12 未知类细分

| method | nmi | ari | purity | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- |
| OpenRFI | 0.981091 | 0.967620 | 0.990986 | 0.924479 |
| PCBM (ours) | 0.998125 | 0.998637 | 0.999375 | 1.000000 |
