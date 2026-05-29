# P2 共有 Tx 对齐弱基线诊断

| protocol | subset | method | nmi_mean | nmi_std | ari_mean | ari_std | purity_mean | purity_std | hungarian_accuracy_mean | hungarian_accuracy_std | valid_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_all_successful_subsets | ManyRx | FFT Magnitude + PCA + K-Means | 0.068041 | 0.012042 | 0.046371 | 0.010691 | 0.317133 | 0.015365 | 0.313200 | 0.013753 | 10 |
| P2_all_successful_subsets | ManyRx | Raw IQ + PCA + GMM | 0.254909 | 0.027109 | 0.214451 | 0.040170 | 0.473267 | 0.035367 | 0.459400 | 0.043831 | 10 |
| P2_all_successful_subsets | ManyRx | Raw IQ + PCA + K-Means | 0.071184 | 0.011039 | 0.069499 | 0.011525 | 0.329067 | 0.009457 | 0.325533 | 0.008564 | 10 |
| P2_all_successful_subsets | ManySig | FFT Magnitude + PCA + K-Means | 0.194472 | 0.018882 | 0.150272 | 0.024616 | 0.406133 | 0.015315 | 0.397533 | 0.015569 | 10 |
| P2_all_successful_subsets | ManySig | Raw IQ + PCA + GMM | 0.360096 | 0.009145 | 0.335644 | 0.018030 | 0.549200 | 0.014648 | 0.546400 | 0.014174 | 10 |
| P2_all_successful_subsets | ManySig | Raw IQ + PCA + K-Means | 0.062646 | 0.010160 | 0.053549 | 0.014552 | 0.318600 | 0.012273 | 0.307733 | 0.016260 | 10 |
| P2_all_successful_subsets | ManyTx | FFT Magnitude + PCA + K-Means | 0.088310 | 0.022173 | 0.059168 | 0.016798 | 0.324400 | 0.013264 | 0.320400 | 0.013244 | 10 |
| P2_all_successful_subsets | ManyTx | Raw IQ + PCA + GMM | 0.261409 | 0.012844 | 0.234366 | 0.019003 | 0.486800 | 0.012010 | 0.482800 | 0.014397 | 10 |
| P2_all_successful_subsets | ManyTx | Raw IQ + PCA + K-Means | 0.055800 | 0.009588 | 0.048870 | 0.013079 | 0.312867 | 0.011419 | 0.304667 | 0.015129 | 10 |
| P2_all_successful_subsets | SingleDay | FFT Magnitude + PCA + K-Means | 0.351453 | 0.014704 | 0.249824 | 0.014854 | 0.528467 | 0.014851 | 0.512733 | 0.015030 | 10 |
| P2_all_successful_subsets | SingleDay | Raw IQ + PCA + GMM | 0.339680 | 0.012080 | 0.312213 | 0.011578 | 0.536733 | 0.007975 | 0.530133 | 0.011313 | 10 |
| P2_all_successful_subsets | SingleDay | Raw IQ + PCA + K-Means | 0.069185 | 0.017680 | 0.065593 | 0.020027 | 0.330133 | 0.018279 | 0.322867 | 0.024553 | 10 |
