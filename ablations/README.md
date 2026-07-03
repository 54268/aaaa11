# Ablation Experiments

This directory keeps the current paper-ready ablation summaries.

Primary entry files:

- summary markdown/json/csv in this directory
- `boundary_module_ablation_supervised_calibrator.md`

The module ablation no longer uses the old OpenMax / prototype-distance switch table. The current design is:

1. `Prototype classifier`: closed-set prototype classifier, without open-set calibration.
2. `MBS-Calib`: ordinary marginal boundary samples only, with the same supervised calibrator structure.
3. `PCBM`: MBS + PCBS + conflict-aware extrapolation, with the supervised calibrator.

The subdivision ablation compares only `Embedding only`, `I/Q descriptors only`, and `Feature fusion`. The post-GMM uncertainty handling is treated as part of the auto-K GMM fitting, screening, and balanced merging process, not as a separate module.

Historical K+M buffer-component sensitivity experiments are not part of the current paper-ready result set.

Common command:

```powershell
python ablations\run_ablation.py --summary-only
```
