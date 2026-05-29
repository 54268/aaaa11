# 当前 SingleDay unknown 协议直接聚类诊断

| method | nmi_mean | nmi_std | ari_mean | ari_std | purity_mean | purity_std | hungarian_accuracy_mean | hungarian_accuracy_std | valid_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFT Magnitude + PCA + K-Means | 0.921285 | 0.005880 | 0.821523 | 0.025041 | 0.845550 | 0.025677 | 0.829600 | 0.028157 | 10 |
| Raw IQ + PCA + GMM | 0.368094 | 0.009209 | 0.160568 | 0.011044 | 0.366850 | 0.008737 | 0.327333 | 0.016690 | 10 |
| Raw IQ + PCA + K-Means | 0.223301 | 0.007922 | 0.097590 | 0.004025 | 0.258083 | 0.005595 | 0.240533 | 0.006633 | 10 |
