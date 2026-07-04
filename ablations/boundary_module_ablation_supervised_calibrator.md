# Boundary Mining Module Ablation

All rows use the supervised-calibrator protocol. MBS-Calib uses ordinary marginal boundary samples only; PCBM further uses PCBS and conflict-aware extrapolation. Real test unknown samples are not used for calibration or threshold selection.

| Dataset | Variant | MBS | PCBS + CAE | Known Acc | Unknown Recall | Macro-F1 | OSCR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ORACLE | Prototype classifier | No | No | 97.82 | 0.00 | 64.89 | 0.00 |
| ORACLE | MBS-Calib | Yes | No | 92.19 | 74.95 | 79.45 | 89.36 |
| ORACLE | PCBM | Yes | Yes | 94.55 | 96.52 | 92.81 | 94.15 |
| WiSig | Prototype classifier | No | No | 99.80 | 0.00 | 61.21 | 0.00 |
| WiSig | MBS-Calib | Yes | No | 98.24 | 80.42 | 91.51 | 97.99 |
| WiSig | PCBM | Yes | Yes | 98.36 | 100.00 | 99.20 | 98.36 |

## Validation Evidence

| Dataset | Variant | Val Known Acc | Pseudo Unknown Recall | Val AUROC | Val OSCR |
| --- | --- | ---: | ---: | ---: | ---: |
| ORACLE | MBS-Calib | 93.00 | 59.75 | 93.07 | 89.82 |
| ORACLE | PCBM | 93.00 | 76.59 | 95.66 | 91.23 |
| WiSig | MBS-Calib | 99.53 | 23.94 | 94.10 | 93.93 |
| WiSig | PCBM | 99.53 | 58.62 | 97.51 | 97.31 |

Prototype classifier is a closed-set baseline and has no unknown-rejection operating point; its Unknown Recall is therefore 0. The WiSig MBS-Calib row uses the recall-oriented ordinary-MBS setting `scale=1.0, lambda=1.0, epochs=100, seed=44, q=0.90`; PCBM uses the formal supervised-calibrator result.
