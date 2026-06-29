# 最终对比表

本文件为 formal 结果目录下的精简版最终汇总，只保留对比方法最核心的拒识与细分指标。开放集拒识重点汇总 `unknown_recall`、`known_accuracy`、`macro_f1`、`oscr`；未知类细分重点汇总 `nmi`、`ari`、`hungarian_accuracy` 与覆盖率 `coverage_of_total_test_unknown`。

## oracle_kri16_demod 开放集拒识

| method | unknown_recall | known_accuracy | macro_f1 | oscr |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 0.536417 | 0.934125 | 0.721127 | 0.876319 |
| OpenMax | 0.886708 | 0.795875 | 0.804256 | 0.779745 |
| HyperRSI | 0.895458 | 0.891500 | 0.866924 | 0.878105 |
| HyDRA | 0.805417 | 0.882500 | 0.826454 | 0.838093 |
| OpenRFI | 0.945667 | 0.899625 | 0.905725 | 0.878649 |
| ARPL | 0.882875 | 0.893750 | 0.854296 | 0.841156 |
| PCBM (ours) | 0.965167 | 0.945500 | 0.928058 | 0.941550 |

## oracle_kri16_demod 未知类细分

| method | nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | ---: | ---: | ---: | ---: |
| K-means | 0.852434 | 0.832207 | 0.921844 | 0.967083 |
| HDBSCAN | 0.684820 | 0.425885 | 0.487485 | 0.699167 |
| OpenRFI | 0.931928 | 0.909938 | 0.930196 | 0.850000 |
| PCBM (ours) | 0.991948 | 0.993851 | 0.997400 | 0.945583 |

注：`coverage_of_total_test_unknown` 表示最终被分配到有效未知簇的真实未知样本比例，而不是单纯的拒识召回率。HDBSCAN 在细分聚类阶段会将低密度样本输出为 `-1`；OpenRFI 在 full-test-world 原型分组得到置信度后用 `confidence_threshold` 保留高置信结果，不是本文 PCBM 的 unknown cache 预筛选。

## wisig_singleday_osr_k16_u12 开放集拒识

| method | unknown_recall | known_accuracy | macro_f1 | oscr |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 0.906250 | 0.952734 | 0.916164 | 0.899614 |
| OpenMax | 0.997917 | 0.899219 | 0.928624 | 0.899039 |
| HyperRSI | 1.000000 | 0.894141 | 0.939272 | 0.894112 |
| HyDRA | 1.000000 | 0.900391 | 0.946136 | 0.900361 |
| OpenRFI | 0.924479 | 0.900391 | 0.891645 | 0.888071 |
| ARPL | 1.000000 | 0.899609 | 0.948643 | 0.899579 |
| PCBM (ours) | 1.000000 | 0.970703 | 0.985671 | 0.970622 |

## wisig_singleday_osr_k16_u12 未知类细分

| method | nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | ---: | ---: | ---: | ---: |
| K-means | 0.917103 | 0.806523 | 0.815104 | 1.000000 |
| HDBSCAN | 0.963178 | 0.933374 | 0.922617 | 0.975938 |
| OpenRFI | 0.981091 | 0.967620 | 0.962254 | 0.924479 |
| PCBM (ours) | 0.998125 | 0.998637 | 0.999375 | 1.000000 |

注：K-means/HDBSCAN 采用 WiSig 未知测试集上的 FFT-magnitude + PCA32 直接聚类基线；K-means 不产生噪声标签，因此 coverage 为 1.000000。OpenRFI 采用 formal 适配输出；PCBM 采用监督校准器 unknown cache、`embedding_iq_stats`、PCA96、`gmm_full_direct`、自动过聚类搜索和 merge-to-target。
