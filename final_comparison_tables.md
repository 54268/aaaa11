# 最终对比表

本文件为项目根目录下的精简版最终汇总，只保留对比方法最核心的拒识与细分指标。本文方法的未知类细分流程说明见 `未知类细分流程说明.md`。
开放集拒识重点汇总 `unknown_recall`、`known_accuracy`、`macro_f1`、`auroc`；未知类细分重点汇总 `nmi`、`ari`、`hungarian_accuracy` 与覆盖率 `coverage_of_total_test_unknown`。

## oracle_kri16_demod 开放集拒识

| method | unknown_recall | known_accuracy | macro_f1 | auroc |
| --- | --- | --- | --- | --- |
| Softmax | 0.536417 | 0.934125 | 0.721127 | 0.900906 |
| OpenMax | 0.886708 | 0.795875 | 0.804256 | 0.900686 |
| HyperRSI | 0.895458 | 0.891500 | 0.866924 | 0.951477 |
| HyDRA | 0.805417 | 0.882500 | 0.826454 | 0.904814 |
| OpenRFI | 0.945667 | 0.899625 | 0.905725 | 0.969528 |
| ARPL | 0.882875 | 0.893750 | 0.854296 | 0.932765 |
| PCBM (ours) | 0.965167 | 0.945500 | 0.928058 | 0.981370 |

## oracle_kri16_demod 未知类细分

| method | nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- |
| K-means | 0.852434 | 0.832207 | 0.921844 | 0.967083 |
| HDBSCAN | 0.684820 | 0.425885 | 0.487485 | 0.699167 |
| OpenRFI | 0.931928 | 0.909938 | 0.930196 | 0.850000 |
| PCBM (ours) | 0.998816 | 0.999263 | 0.999718 | 0.886292 |

## wisig_singleday_osr_k16_u12 开放集拒识

| method | unknown_recall | known_accuracy | macro_f1 | auroc |
| --- | --- | --- | --- | --- |
| Softmax | 0.906250 | 0.952734 | 0.916164 | 0.940071 |
| OpenMax | 0.997917 | 0.899219 | 0.928624 | 0.988474 |
| HyperRSI | 1.000000 | 0.894141 | 0.939272 | 0.998671 |
| HyDRA | 1.000000 | 0.900391 | 0.946136 | 0.998989 |
| OpenRFI | 0.924479 | 0.900391 | 0.891645 | 0.978104 |
| ARPL | 1.000000 | 0.899609 | 0.948643 | 0.992654 |
| PCBM (ours) | 1.000000 | 0.984375 | 0.992430 | 0.998926 |

## wisig_singleday_osr_k16_u12 未知类细分

| method | nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | --- | --- | --- | --- |
| K-means | 0.998125 | 0.998637 | 0.999375 | 1.000000 |
| HDBSCAN | 0.999088 | 0.999316 | 0.999686 | 0.994687 |
| OpenRFI | 0.981091 | 0.967620 | 0.962254 | 0.924479 |
| PCBM (ours) | 0.998125 | 0.998637 | 0.999375 | 1.000000 |
