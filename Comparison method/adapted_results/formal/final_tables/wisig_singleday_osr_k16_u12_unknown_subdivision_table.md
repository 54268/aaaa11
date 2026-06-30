# wisig_singleday_osr_k16_u12 unknown subdivision

| method | nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown |
| --- | ---: | ---: | ---: | ---: |
| K-means | 0.917103 | 0.806523 | 0.815104 | 1.000000 |
| HDBSCAN | 0.963178 | 0.933374 | 0.922617 | 0.975938 |
| OpenRFI | 0.981091 | 0.967620 | 0.962254 | 0.924479 |
| PCBM (ours, auto K) | 0.998125 | 0.998637 | 0.999375 | 1.000000 |

Note: K-means/HDBSCAN use direct FFT-magnitude + PCA32 clustering on the WiSig unknown test split. OpenRFI uses the formal adapted OpenRFI subdivision output. PCBM uses the supervised-calibrator unknown cache with `embedding_iq_stats`, PCA96, `gmm_full_direct`, automatic candidate-K search, and balance-based automatic merging.
