# P3 Receiver / Day 条件敏感性诊断

| protocol | subset | method | nmi_mean | nmi_std | ari_mean | ari_std | purity_mean | purity_std | hungarian_accuracy_mean | hungarian_accuracy_std | valid_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3-1_SingleDay_FixedRx | SingleDay | FFT Magnitude + PCA + K-Means | 0.933533 | 0.065200 | 0.898674 | 0.103910 | 0.924944 | 0.083334 | 0.923500 | 0.085244 | 10 |
| P3-1_SingleDay_FixedRx | SingleDay | Raw IQ + PCA + GMM | 0.309143 | 0.069913 | 0.188462 | 0.072217 | 0.443556 | 0.057900 | 0.428889 | 0.064742 | 10 |
| P3-1_SingleDay_FixedRx | SingleDay | Raw IQ + PCA + K-Means | 0.100018 | 0.030679 | 0.058603 | 0.018508 | 0.300556 | 0.024509 | 0.284444 | 0.023372 | 10 |
| P3-2_SingleDay_MixedRx | SingleDay | FFT Magnitude + PCA + K-Means | 0.300984 | 0.064146 | 0.206498 | 0.050818 | 0.451944 | 0.055195 | 0.426833 | 0.055349 | 10 |
| P3-2_SingleDay_MixedRx | SingleDay | Raw IQ + PCA + GMM | 0.271613 | 0.078668 | 0.203745 | 0.062068 | 0.434389 | 0.061219 | 0.406333 | 0.049480 | 10 |
| P3-2_SingleDay_MixedRx | SingleDay | Raw IQ + PCA + K-Means | 0.032801 | 0.036310 | 0.027410 | 0.037446 | 0.241389 | 0.040775 | 0.237111 | 0.040884 | 10 |
| P3-3_ManySig_FixedRxDay | ManySig | FFT Magnitude + PCA + K-Means | 0.994475 | 0.001991 | 0.996400 | 0.001338 | 0.998500 | 0.000558 | 0.998500 | 0.000558 | 10 |
| P3-3_ManySig_FixedRxDay | ManySig | Raw IQ + PCA + GMM | 0.363712 | 0.030377 | 0.217371 | 0.028262 | 0.471611 | 0.031868 | 0.433944 | 0.035860 | 10 |
| P3-3_ManySig_FixedRxDay | ManySig | Raw IQ + PCA + K-Means | 0.190283 | 0.026049 | 0.117746 | 0.016057 | 0.346000 | 0.014240 | 0.327222 | 0.023624 | 10 |
| P3-4_ManySig_MixedDay | ManySig | FFT Magnitude + PCA + K-Means | 0.928197 | 0.003567 | 0.904128 | 0.005848 | 0.954778 | 0.003203 | 0.954778 | 0.003203 | 10 |
| P3-4_ManySig_MixedDay | ManySig | Raw IQ + PCA + GMM | 0.329131 | 0.019879 | 0.238025 | 0.012359 | 0.477944 | 0.017450 | 0.466500 | 0.022741 | 10 |
| P3-4_ManySig_MixedDay | ManySig | Raw IQ + PCA + K-Means | 0.153275 | 0.023951 | 0.102159 | 0.014634 | 0.327944 | 0.014701 | 0.316444 | 0.017098 | 10 |
| P3-5_ManyRx_MixedRx | ManyRx | FFT Magnitude + PCA + K-Means | 0.057869 | 0.041048 | 0.036707 | 0.029340 | 0.267778 | 0.035663 | 0.263944 | 0.037823 | 10 |
| P3-5_ManyRx_MixedRx | ManyRx | Raw IQ + PCA + GMM | 0.191018 | 0.048543 | 0.150327 | 0.051714 | 0.386556 | 0.046221 | 0.377833 | 0.044039 | 10 |
| P3-5_ManyRx_MixedRx | ManyRx | Raw IQ + PCA + K-Means | 0.039998 | 0.032800 | 0.036691 | 0.033908 | 0.249833 | 0.043583 | 0.245278 | 0.042513 | 10 |
