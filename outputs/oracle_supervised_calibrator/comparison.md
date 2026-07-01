# Oracle Supervised Calibrator Current Result

This file records only the current paper-facing Oracle result. Historical manual-threshold,
auto-threshold, and leave-class-out comparison rows were removed from the main output to avoid
mixing old calibration settings with the final supervised-calibrator protocol.

## Open-Set Rejection

| known_accuracy | unknown_recall | unknown_precision | macro_f1 | oscr |
| ---: | ---: | ---: | ---: | ---: |
| 94.55 | 96.52 | 98.52 | 92.81 | 94.15 |

## Unknown Subdivision

| nmi | ari | hungarian_accuracy | coverage_of_total_test_unknown | fit_K | effective_K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96.34 | 95.48 | 97.93 | 94.31 | 8 | 6 |

The subdivision stage uses true automatic candidate search over `K=2..20` with
`target_num_clusters=null`; true unknown labels are used only for offline evaluation.
