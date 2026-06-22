# Pseudo-Unknown Supervised Calibrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a compact unknown-probability calibrator from known validation samples and feature-level pseudo-unknowns, select it using leakage-free five-fold held-out classes, and evaluate once on Oracle real unknowns.

**Architecture:** A focused calibrator module owns feature construction, balanced training, persistence, scoring, and threshold selection. A separate Oracle experiment runner reuses the five trained leave-class-out encoders, generates several pseudo-unknown extrapolation scales from saved source embeddings, selects hyperparameters from held-out known classes, trains the formal ten-class calibrator, freezes it, and invokes the existing final evaluator with a new calibrated-score mode.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing CVCNN/OpenMax/prototype-distance pipeline.

---

### Task 1: Calibrator model and training primitives

**Files:**
- Create: `functions/methods/supervised_calibrator.py`
- Create: `tests/test_supervised_calibrator.py`

- [ ] **Step 1: Write failing tests for the three-score feature matrix**

```python
def test_build_calibrator_features_preserves_linear_fusion():
    q_om = np.array([0.2, 0.8])
    q_pd = np.array([0.6, 0.4])
    features = build_calibrator_features(q_om, q_pd, fusion_lambda=0.25)
    assert features.shape == (2, 3)
    assert np.allclose(features[:, 2], 0.25 * q_om + 0.75 * q_pd)
```

- [ ] **Step 2: Run the test and verify missing-module failure**

Run: `D:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_supervised_calibrator.py -q`

Expected: import failure for `functions.methods.supervised_calibrator`.

- [ ] **Step 3: Implement `build_calibrator_features` and `UnknownScoreCalibrator`**

The model is `Linear(3, 8) → ReLU → Linear(8, 1)`. Its inference method applies sigmoid and returns a one-dimensional probability array.

- [ ] **Step 4: Add failing tests for balanced known/pseudo training and persistence**

```python
def test_training_learns_higher_probability_for_separated_pseudo_unknowns(tmp_path):
    known = np.tile([0.1, 0.1, 0.1], (100, 1))
    pseudo = np.tile([0.9, 0.9, 0.9], (100, 1))
    result = train_calibrator(known, pseudo, seed=42, epochs=100, lr=0.01)
    assert result.model.predict_proba(pseudo).mean() > 0.9
    assert result.model.predict_proba(known).mean() < 0.1
    save_calibrator(tmp_path / "calibrator.pt", result)
    restored = load_calibrator(tmp_path / "calibrator.pt")
    assert np.allclose(restored.predict_proba(pseudo), result.model.predict_proba(pseudo))
```

- [ ] **Step 5: Implement deterministic balanced BCE training, save/load, then run tests**

Use equal known and pseudo counts, `BCEWithLogitsLoss`, Adam, seed-controlled shuffling, and CPU persistence. Return loss history and sample counts.

### Task 2: Threshold search under a 95% known-accuracy constraint

**Files:**
- Modify: `functions/methods/supervised_calibrator.py`
- Modify: `tests/test_supervised_calibrator.py`

- [ ] **Step 1: Write failing tests for classwise quantile thresholds**

```python
def test_classwise_thresholds_keep_requested_known_fraction():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    pred = np.array([0, 0, 1, 1])
    thresholds = thresholds_from_quantile(scores, pred, 2, quantile=0.5)
    assert thresholds == [0.1, 0.3]
```

- [ ] **Step 2: Implement threshold construction and held-out evaluation**

Add `thresholds_from_quantile` and `evaluate_calibrator_candidate`. Candidate metrics include Known Acc, held-out unknown recall, Macro F1 and AUROC. Candidate score is:

```python
0.45 * known_accuracy + 0.35 * heldout_unknown_recall + 0.15 * macro_f1 + 0.05 * auroc
```

- [ ] **Step 3: Add and pass a test that rejects candidates below Known Acc 0.95**

Search quantiles `[0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]`; mark feasibility from the measured known accuracy, not from the nominal quantile.

### Task 3: Five-fold pseudo-scale and calibrator selection

**Files:**
- Create: `run_oracle_supervised_calibrator.py`
- Modify: `settings/oracle_settings.py`
- Modify: `tests/test_main_calibration_config.py`
- Create: `tests/test_oracle_supervised_calibrator_runner.py`

- [ ] **Step 1: Add failing configuration tests**

Require:

```python
{
    "enabled": True,
    "hidden_dim": 8,
    "pseudo_scale_grid": [0.75, 1.0, 1.25, 1.5, 2.0],
    "lambda_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "epoch_grid": [100, 250],
    "seed_grid": [42, 43, 44],
    "threshold_quantile_grid": [0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99],
    "min_known_accuracy": 0.95,
    "uses_real_unknown_for_selection": False,
}
```

- [ ] **Step 2: Implement pseudo-unknown rescaling**

For a saved pseudo embedding \(u\), source embedding \(z\), and multiplier \(s\), construct:

```python
u_scaled = z + s * (u - z)
```

Balance samples across source class and pseudo kind before scoring.

- [ ] **Step 3: Implement the five-fold runner**

For each fold and pseudo scale:

1. Load the existing eight-class checkpoint, OpenMax, distance statistics, boundary embeddings, and pseudo source metadata.
2. Compute known, held-out and scaled-pseudo score features.
3. For every lambda, epoch count and seed, train only on retained validation-known versus pseudo-unknown.
4. Evaluate held-out classes over the threshold quantile grid.
5. Save fold candidates without reading formal real-unknown test data.

Select a single hyperparameter tuple by five-fold mean selection score among candidates with mean Known Acc at least 0.95. Use worst-fold Known Acc and lower seed as deterministic tie-breakers.

- [ ] **Step 4: Add runner tests**

Test pseudo scaling, deterministic candidate selection, and an audit flag proving the held-out fold samples are evaluation-only.

### Task 4: Formal calibrator and final evaluation

**Files:**
- Modify: `functions/pipeline.py`
- Modify: `run_oracle_supervised_calibrator.py`
- Modify: `tests/test_supervised_calibrator.py`

- [ ] **Step 1: Write a failing evaluator test**

Create a temporary calibrator artifact and assert the evaluation path uses its probability output when `fusion.json` contains:

```python
{
    "score_calibration": {
        "mode": "supervised_calibrator",
        "path": "outputs/oracle_supervised_calibrator/final/supervised_calibrator.pt",
    },
    "fusion_lambda": 0.5,
}
```

- [ ] **Step 2: Add supervised-calibrator score application**

The evaluator first computes `q_om`, `q_pd`, and the retained linear fusion score, then replaces the final score with the frozen calibrator probability. Existing non-supervised fusion behavior remains unchanged.

- [ ] **Step 3: Train the formal calibrator**

Use the selected pseudo scale, lambda, epoch count and seed on formal ten-class validation-known plus formal pseudo-unknowns. Restore ten classwise thresholds from the selected quantile and save:

- `supervised_calibrator.pt`;
- `supervised_calibrator_training.json`;
- `fusion.json`;
- `selection_manifest.json`.

- [ ] **Step 4: Freeze configuration and evaluate real unknowns once**

Copy the formal closed-set/OpenMax/distance artifacts into `outputs/oracle_supervised_calibrator/final`, write a SHA-256 configuration fingerprint, and only then invoke `evaluate_open_set_artifacts`.

### Task 5: Comparison, verification and leakage audit

**Files:**
- Create: `outputs/oracle_supervised_calibrator/comparison.json`
- Create: `outputs/oracle_supervised_calibrator/comparison.md`

- [ ] **Step 1: Generate the four-way comparison**

Compare:

- old manual/leaking version;
- pseudo-only automatic threshold version;
- five-fold threshold-transfer version;
- supervised-calibrator version.

Report Known Acc, Unknown Recall, Macro F1 and AUROC with absolute percentage-point changes.

- [ ] **Step 2: Run the focused verification suite**

Run:

`D:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_supervised_calibrator.py tests/test_oracle_supervised_calibrator_runner.py tests/test_leave_class_out.py tests/test_oracle_leave_class_out_runner.py tests/test_fusion_calibration.py tests/test_main_calibration_config.py -q`

Expected: all tests pass.

- [ ] **Step 3: Run leakage audit**

Assert five fold manifests and the final selection manifest all contain `uses_real_unknown_for_selection=false`; assert no `test_unknown.npz` from the formal Oracle data root appears in any training-input manifest.

- [ ] **Step 4: Report outcomes honestly**

State whether Known Acc ≥95%, Unknown Recall ≥95%, and Macro F1 ≥92% were achieved. If not, report the exact shortfall without further tuning against real unknown labels.
