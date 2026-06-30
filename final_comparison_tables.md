# 最终对比表

本文档是项目根目录下的精简版最终汇总，只保留对比方法最核心的拒识与未知类细分指标。更完整的细分流程见 `未知类细分流程说明.md`，自动搜索候选 K 的口径见 `自动K搜索与未知类细分流程说明.md`。

开放集拒识汇总 `unknown_recall`、`known_accuracy`、`macro_f1`、`oscr`。未知类细分汇总 `nmi`、`ari`、`hungarian_accuracy` 和最终有效细分覆盖率 `coverage_of_total_test_unknown`。

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
| PCBM (ours, auto K) | 0.991948 | 0.993851 | 0.997400 | 0.945583 |

注：PCBM 不预先使用真实未知类数。Oracle 自动候选搜索选择 `fit_K=7`，经过低置信/小簇过滤后得到 `effective_K=6` 个有效未知簇。

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
| PCBM (ours, auto K) | 0.998125 | 0.998637 | 0.999375 | 1.000000 |

注：WiSig 自动候选搜索选择 `fit_K=13`。直接保留 13 个簇时 Hungarian accuracy 为 0.960625；采用统一的无标签簇均衡合并规则后，1 个冗余分量被并入最近簇，最终 `effective_K=12`，指标恢复为上表结果。

## 覆盖率口径

`coverage_of_total_test_unknown` 表示真实未知测试样本中最终被分配到有效未知簇的比例，而不是单纯的拒识召回率：

```text
coverage_of_total_test_unknown
= unknown_cache_recall * coverage_of_selected_true_unknown
```

K-means 不产生 `-1`，因此覆盖率等于进入其评估集合的真实未知样本比例。HDBSCAN 会在细分聚类阶段将低密度样本标为噪声。OpenRFI 采用 full-test-world prototype grouping，并在分组后按置信度保留结果。PCBM 的不确定样本来自 GMM-full-direct 之后的低置信过滤、小簇过滤，以及自动簇均衡合并后的有效簇判定。
