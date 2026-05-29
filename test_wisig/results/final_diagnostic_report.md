# WiSig 数据集未知类天然可分性诊断实验报告

## 1. 实验目的

本实验用于诊断 WiSig 当前受控协议是否天然容易进行未知类细分，不用于证明主方法性能。实验刻意不使用 CVCNN 深度特征、OpenMax、原型距离、伪未知样本、known prototype guided clustering 和当前 unknown cache，只对真实 unknown 类样本直接使用经典弱基线聚类。

## 2. 目录隔离说明

- 本实验所有新增内容均位于 `test_wisig/`。
- 原项目代码、配置与数据仅以只读方式访问。
- 原有 `outputs/` 未被覆盖或写入新的诊断结果。

## 3. 数据扫描结果

| subset | path | file_format | loaded | error | object_type | num_tx | num_rx | num_days | equalized_values | signal_shape | tx_sample_min | tx_sample_max | tx_sample_mean | tx_sample_median | can_run_p1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SingleDay | D:\learn_pytorch\笔记\方案\os_sei_code\data\raw\wisig\SingleDay_unpacked\SingleDay.pkl | pkl | True | nan | dict | 28 | 10 | 1 | 0,1 | (2, 256) | 16000.000000 | 16000.000000 | 16000.000000 | 16000.000000 | True |
| ManySig | D:\learn_pytorch\笔记\方案\os_sei_code\data\raw\wisig\ManySig_unpacked\ManySig.pkl | pkl | True | nan | dict | 6 | 12 | 4 | 0,1 | (2, 256) | 96000.000000 | 96000.000000 | 96000.000000 | 96000.000000 | True |
| ManyRx | D:\learn_pytorch\笔记\方案\os_sei_code\data\raw\wisig\ManyRx_unpacked\ManyRx.pkl | pkl | True | nan | dict | 10 | 32 | 4 | 0,1 | (2, 256) | 46875.000000 | 51200.000000 | 49735.000000 | 49696.000000 | True |
| ManyTx | D:\learn_pytorch\笔记\方案\os_sei_code\data\raw\wisig\ManyTx_unpacked\ManyTx.pkl | pkl | True | nan | dict | 150 | 18 | 4 | 0,1 | (2, 256) | 3164.000000 | 7200.000000 | 6804.286667 | 7143.000000 | True |

## 4. 当前 SingleDay 主协议直接聚类结果

- 当前 unknown 类数量：12
- receiver 筛选：['1-1']
- day 筛选：['2021_03_23']
- equalized 筛选：[0]
- 说明：此处未经过任何拒识模块和深度特征提取。

| method | NMI mean±std | ARI mean±std | Purity mean±std | Hungarian Accuracy mean±std |
| --- | --- | --- | --- | --- |
| FFT Magnitude + PCA + K-Means | 0.921285 ± 0.005880 | 0.821523 ± 0.025041 | 0.845550 ± 0.025677 | 0.829600 ± 0.028157 |
| Raw IQ + PCA + GMM | 0.368094 ± 0.009209 | 0.160568 ± 0.011044 | 0.366850 ± 0.008737 | 0.327333 ± 0.016690 |
| Raw IQ + PCA + K-Means | 0.223301 ± 0.007922 | 0.097590 ± 0.004025 | 0.258083 ± 0.005595 | 0.240533 ± 0.006633 |

## 5. 四个 subset 的 P1 统一 6 类结果

| subset | method | NMI mean±std | ARI mean±std | Purity mean±std | Hungarian Accuracy mean±std |
| --- | --- | --- | --- | --- | --- |
| ManyRx | FFT Magnitude + PCA + K-Means | 0.061221 ± 0.024566 | 0.038904 ± 0.020687 | 0.277833 ± 0.030060 | 0.270111 ± 0.030341 |
| ManyRx | Raw IQ + PCA + GMM | 0.174449 ± 0.041465 | 0.131745 ± 0.045752 | 0.369722 ± 0.044260 | 0.361222 ± 0.041629 |
| ManyRx | Raw IQ + PCA + K-Means | 0.048473 ± 0.032423 | 0.045569 ± 0.033838 | 0.261833 ± 0.038973 | 0.256444 ± 0.040154 |
| ManySig | FFT Magnitude + PCA + K-Means | 0.161239 ± 0.026884 | 0.111406 ± 0.019197 | 0.347167 ± 0.013495 | 0.339222 ± 0.014872 |
| ManySig | Raw IQ + PCA + GMM | 0.299290 ± 0.010702 | 0.275755 ± 0.012448 | 0.478722 ± 0.009432 | 0.474500 ± 0.010566 |
| ManySig | Raw IQ + PCA + K-Means | 0.078323 ± 0.015116 | 0.073306 ± 0.017421 | 0.295500 ± 0.014960 | 0.289667 ± 0.016645 |
| ManyTx | FFT Magnitude + PCA + K-Means | 0.116151 ± 0.084301 | 0.070343 ± 0.060048 | 0.299389 ± 0.041677 | 0.287556 ± 0.038259 |
| ManyTx | Raw IQ + PCA + GMM | 0.188527 ± 0.085457 | 0.136256 ± 0.070464 | 0.352111 ± 0.067310 | 0.343111 ± 0.062964 |
| ManyTx | Raw IQ + PCA + K-Means | 0.019654 ± 0.015950 | 0.011764 ± 0.012773 | 0.225833 ± 0.021661 | 0.219500 ± 0.020819 |
| SingleDay | FFT Magnitude + PCA + K-Means | 0.300984 ± 0.064146 | 0.206498 ± 0.050818 | 0.451944 ± 0.055195 | 0.426833 ± 0.055349 |
| SingleDay | Raw IQ + PCA + GMM | 0.271613 ± 0.078668 | 0.203745 ± 0.062068 | 0.434389 ± 0.061219 | 0.406333 ± 0.049480 |
| SingleDay | Raw IQ + PCA + K-Means | 0.032801 ± 0.036310 | 0.027410 ± 0.037446 | 0.241389 ± 0.040775 | 0.237111 ± 0.040884 |

## 6. Shared-Tx 的 P2 对齐结果

P2 使用共同 Tx 集合对齐 subset，能够减少不同 subset 随机抽到不同 Tx 难度所带来的偏差。

| protocol | subset | method | NMI mean±std | ARI mean±std | Purity mean±std | Hungarian Accuracy mean±std |
| --- | --- | --- | --- | --- | --- | --- |
| P2_all_successful_subsets | ManyRx | FFT Magnitude + PCA + K-Means | 0.068041 ± 0.012042 | 0.046371 ± 0.010691 | 0.317133 ± 0.015365 | 0.313200 ± 0.013753 |
| P2_all_successful_subsets | ManyRx | Raw IQ + PCA + GMM | 0.254909 ± 0.027109 | 0.214451 ± 0.040170 | 0.473267 ± 0.035367 | 0.459400 ± 0.043831 |
| P2_all_successful_subsets | ManyRx | Raw IQ + PCA + K-Means | 0.071184 ± 0.011039 | 0.069499 ± 0.011525 | 0.329067 ± 0.009457 | 0.325533 ± 0.008564 |
| P2_all_successful_subsets | ManySig | FFT Magnitude + PCA + K-Means | 0.194472 ± 0.018882 | 0.150272 ± 0.024616 | 0.406133 ± 0.015315 | 0.397533 ± 0.015569 |
| P2_all_successful_subsets | ManySig | Raw IQ + PCA + GMM | 0.360096 ± 0.009145 | 0.335644 ± 0.018030 | 0.549200 ± 0.014648 | 0.546400 ± 0.014174 |
| P2_all_successful_subsets | ManySig | Raw IQ + PCA + K-Means | 0.062646 ± 0.010160 | 0.053549 ± 0.014552 | 0.318600 ± 0.012273 | 0.307733 ± 0.016260 |
| P2_all_successful_subsets | ManyTx | FFT Magnitude + PCA + K-Means | 0.088310 ± 0.022173 | 0.059168 ± 0.016798 | 0.324400 ± 0.013264 | 0.320400 ± 0.013244 |
| P2_all_successful_subsets | ManyTx | Raw IQ + PCA + GMM | 0.261409 ± 0.012844 | 0.234366 ± 0.019003 | 0.486800 ± 0.012010 | 0.482800 ± 0.014397 |
| P2_all_successful_subsets | ManyTx | Raw IQ + PCA + K-Means | 0.055800 ± 0.009588 | 0.048870 ± 0.013079 | 0.312867 ± 0.011419 | 0.304667 ± 0.015129 |
| P2_all_successful_subsets | SingleDay | FFT Magnitude + PCA + K-Means | 0.351453 ± 0.014704 | 0.249824 ± 0.014854 | 0.528467 ± 0.014851 | 0.512733 ± 0.015030 |
| P2_all_successful_subsets | SingleDay | Raw IQ + PCA + GMM | 0.339680 ± 0.012080 | 0.312213 ± 0.011578 | 0.536733 ± 0.007975 | 0.530133 ± 0.011313 |
| P2_all_successful_subsets | SingleDay | Raw IQ + PCA + K-Means | 0.069185 ± 0.017680 | 0.065593 ± 0.020027 | 0.330133 ± 0.018279 | 0.322867 ± 0.024553 |

## 7. Receiver / Day 敏感性的 P3 结果

| protocol | subset | method | NMI mean±std | ARI mean±std | Purity mean±std | Hungarian Accuracy mean±std |
| --- | --- | --- | --- | --- | --- | --- |
| P3-1_SingleDay_FixedRx | SingleDay | FFT Magnitude + PCA + K-Means | 0.933533 ± 0.065200 | 0.898674 ± 0.103910 | 0.924944 ± 0.083334 | 0.923500 ± 0.085244 |
| P3-1_SingleDay_FixedRx | SingleDay | Raw IQ + PCA + GMM | 0.309143 ± 0.069913 | 0.188462 ± 0.072217 | 0.443556 ± 0.057900 | 0.428889 ± 0.064742 |
| P3-1_SingleDay_FixedRx | SingleDay | Raw IQ + PCA + K-Means | 0.100018 ± 0.030679 | 0.058603 ± 0.018508 | 0.300556 ± 0.024509 | 0.284444 ± 0.023372 |
| P3-2_SingleDay_MixedRx | SingleDay | FFT Magnitude + PCA + K-Means | 0.300984 ± 0.064146 | 0.206498 ± 0.050818 | 0.451944 ± 0.055195 | 0.426833 ± 0.055349 |
| P3-2_SingleDay_MixedRx | SingleDay | Raw IQ + PCA + GMM | 0.271613 ± 0.078668 | 0.203745 ± 0.062068 | 0.434389 ± 0.061219 | 0.406333 ± 0.049480 |
| P3-2_SingleDay_MixedRx | SingleDay | Raw IQ + PCA + K-Means | 0.032801 ± 0.036310 | 0.027410 ± 0.037446 | 0.241389 ± 0.040775 | 0.237111 ± 0.040884 |
| P3-3_ManySig_FixedRxDay | ManySig | FFT Magnitude + PCA + K-Means | 0.994475 ± 0.001991 | 0.996400 ± 0.001338 | 0.998500 ± 0.000558 | 0.998500 ± 0.000558 |
| P3-3_ManySig_FixedRxDay | ManySig | Raw IQ + PCA + GMM | 0.363712 ± 0.030377 | 0.217371 ± 0.028262 | 0.471611 ± 0.031868 | 0.433944 ± 0.035860 |
| P3-3_ManySig_FixedRxDay | ManySig | Raw IQ + PCA + K-Means | 0.190283 ± 0.026049 | 0.117746 ± 0.016057 | 0.346000 ± 0.014240 | 0.327222 ± 0.023624 |
| P3-4_ManySig_MixedDay | ManySig | FFT Magnitude + PCA + K-Means | 0.928197 ± 0.003567 | 0.904128 ± 0.005848 | 0.954778 ± 0.003203 | 0.954778 ± 0.003203 |
| P3-4_ManySig_MixedDay | ManySig | Raw IQ + PCA + GMM | 0.329131 ± 0.019879 | 0.238025 ± 0.012359 | 0.477944 ± 0.017450 | 0.466500 ± 0.022741 |
| P3-4_ManySig_MixedDay | ManySig | Raw IQ + PCA + K-Means | 0.153275 ± 0.023951 | 0.102159 ± 0.014634 | 0.327944 ± 0.014701 | 0.316444 ± 0.017098 |
| P3-5_ManyRx_MixedRx | ManyRx | FFT Magnitude + PCA + K-Means | 0.057869 ± 0.041048 | 0.036707 ± 0.029340 | 0.267778 ± 0.035663 | 0.263944 ± 0.037823 |
| P3-5_ManyRx_MixedRx | ManyRx | Raw IQ + PCA + GMM | 0.191018 ± 0.048543 | 0.150327 ± 0.051714 | 0.386556 ± 0.046221 | 0.377833 ± 0.044039 |
| P3-5_ManyRx_MixedRx | ManyRx | Raw IQ + PCA + K-Means | 0.039998 ± 0.032800 | 0.036691 ± 0.033908 | 0.249833 ± 0.043583 | 0.245278 ± 0.042513 |

## 8. Oracle 已有结果旁注

以下 Oracle 结果只是当前项目已有输出的只读摘录，不是本次同协议弱基线结果。

- `source_path`: D:\learn_pytorch\笔记\方案\os_sei_code\outputs\oracle_kri16_demod_known_first\unknown_subdivision\unknown_subdivision_metrics.json
- `nmi`: 0.8204682772495971
- `ari`: 0.6904274659160125
- `purity`: 0.7116005316119233
- `hungarian_accuracy`: 0.7116005316119233
- `resolved_num_clusters`: 5
- `uncertain_size`: 2966
- `unknown_cache_precision`: 0.9794087244159603
- `unknown_cache_recall`: 0.9869583333333334

## 9. 自动判断结论

- 自动规则未形成强结论：当前结果需要结合 P1/P2/P3 表格人工判断。

## 10. 论文实验建议

- WiSig 是否过易需要结合更多协议结果继续判断。
- 建议保留 Oracle 作为主要困难场景，并继续补充跨 receiver/day 设置。
