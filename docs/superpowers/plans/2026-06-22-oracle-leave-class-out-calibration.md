# Oracle Leave-Class-Out Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train five Oracle eight-class temporary models, use held-out known classes plus feature pseudo-unknowns to select leakage-free fusion parameters, transfer classwise threshold quantiles to the existing ten-class model, and report metric changes.

**Architecture:** A focused leave-class-out module owns fold construction, label remapping, per-fold calibration scoring, and quantile aggregation. A root experiment entry point creates fold NPZ datasets, invokes the existing training/boundary/OpenMax components, writes the transferred `fusion.json` into a separate result directory, and evaluates the untouched real unknown test set with the existing evaluator.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing CVCNN/OpenMax/prototype-distance pipeline.

---

### Task 1: Fold construction and threshold transfer primitives

**Files:**
- Create: `functions/methods/leave_class_out.py`
- Create: `tests/test_leave_class_out.py`

- [ ] **Step 1: Write failing tests for deterministic folds and label remapping**

```python
def test_build_leave_class_out_folds_covers_each_class_once():
    folds = build_leave_class_out_folds(10, 5)
    assert [fold.held_out_classes for fold in folds] == [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
    ]
    assert sorted(cls for fold in folds for cls in fold.held_out_classes) == list(range(10))

def test_subset_and_remap_uses_contiguous_local_labels():
    x = np.arange(24).reshape(6, 1, 4)
    y = np.array([0, 1, 2, 3, 4, 5])
    subset_x, local_y, global_y = subset_and_remap(x, y, (1, 3, 5))
    assert local_y.tolist() == [0, 1, 2]
    assert global_y.tolist() == [1, 3, 5]
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_leave_class_out.py -q`

Expected: import failure because `functions.methods.leave_class_out` does not exist.

- [ ] **Step 3: Implement fold dataclass and remapping**

Implement `LeaveClassOutFold`, `build_leave_class_out_folds`, `subset_and_remap`, and `map_local_class_to_global`. Reject class counts not divisible by the fold count and reject empty subsets.

- [ ] **Step 4: Add tests for threshold-to-quantile conversion and aggregation**

```python
def test_threshold_quantile_round_trip():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    q = threshold_to_known_quantile(scores, 0.3)
    restored = threshold_from_known_quantile(scores, q)
    assert 0.2 <= restored <= 0.4

def test_aggregate_class_quantiles_uses_four_known_folds():
    rows = [
        {"global_class": 0, "quantile": 0.94},
        {"global_class": 0, "quantile": 0.96},
        {"global_class": 0, "quantile": 0.95},
        {"global_class": 0, "quantile": 0.97},
    ]
    assert aggregate_class_quantiles(rows, 1) == [0.955]
```

- [ ] **Step 5: Implement quantile conversion and aggregation, then run tests**

Run: `pytest tests/test_leave_class_out.py -q`

Expected: all Task 1 tests pass.

### Task 2: Per-fold calibration using simulated and pseudo unknowns

**Files:**
- Modify: `functions/methods/leave_class_out.py`
- Modify: `tests/test_leave_class_out.py`

- [ ] **Step 1: Write a failing test for separate unknown-group scoring**

Create synthetic known, held-out, and pseudo arrays where one candidate has better held-out recall but equal known accuracy. Assert `search_leave_class_out_candidate` selects it using weights:

```python
{
    "known_accuracy": 0.40,
    "heldout_unknown_recall": 0.35,
    "pseudo_unknown_recall": 0.10,
    "macro_f1": 0.10,
    "auroc": 0.05,
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_leave_class_out.py::test_search_prefers_heldout_unknown_recall -q`

Expected: missing search function.

- [ ] **Step 3: Implement balanced pseudo sampling and candidate evaluation**

Add:

```python
def balanced_pseudo_indices(
    source_labels: np.ndarray,
    pseudo_kind: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic indices balanced across source-class and pseudo-kind groups."""

def evaluate_leave_class_out_candidate(
    known_labels: np.ndarray,
    known_pred: np.ndarray,
    heldout_pred: np.ndarray,
    pseudo_pred: np.ndarray,
    known_scores: np.ndarray,
    heldout_scores: np.ndarray,
    pseudo_scores: np.ndarray,
    thresholds: np.ndarray,
    unknown_label: int,
) -> dict[str, float]:
    """Return the separate group metrics and weighted selection score."""
```

The metric function reports known accuracy, held-out unknown recall, pseudo unknown recall, combined macro F1, AUROC, selection score, and feasibility.

- [ ] **Step 4: Implement lambda/classwise-threshold search**

For every configured lambda, fit classwise score normalization on known validation only and search classwise thresholds with the existing penalty grid. Return the best feasible candidate for every lambda instead of collapsing the fold to one lambda. Save every local threshold as its class-known score quantile so the runner can choose one global lambda from five-fold mean scores.

- [ ] **Step 5: Run focused and existing fusion tests**

Run: `pytest tests/test_leave_class_out.py tests/test_fusion_calibration.py -q`

Expected: all tests pass.

### Task 3: Oracle five-fold training runner

**Files:**
- Create: `run_oracle_leave_class_out.py`
- Modify: `settings/oracle_settings.py`
- Modify: `tests/test_main_calibration_config.py`

- [ ] **Step 1: Add failing configuration tests**

Assert Oracle leave-class-out settings define five folds, enabled calibration, deterministic fold seeds, the metric weights above, and no manual threshold or real-unknown calibration source.

- [ ] **Step 2: Run the configuration test and verify it fails**

Run: `pytest tests/test_main_calibration_config.py -q`

Expected: missing `leave_class_out` configuration.

- [ ] **Step 3: Add configuration**

Add `fusion.leave_class_out` containing `enabled`, `num_folds`, `pseudo_max_samples`, `selection_weights`, `min_known_accuracy`, and output directory. Keep the current automatic calibration as a separately reproducible baseline.

- [ ] **Step 4: Implement the experiment runner**

The runner:

1. Loads `train_known.npz` and `val_known.npz`.
2. Materializes fold-local NPZ files under `outputs/oracle_leave_class_out_calibration/folds/fold_N/data`.
3. Trains each eight-class model with seed `42 + fold_index`.
4. Runs existing boundary mining, pseudo generation, and OpenMax fitting.
5. Extracts held-out validation embeddings through the fold model.
6. Runs the leave-class-out search and saves `fold_calibration.json`.
7. Chooses the lambda with the highest five-fold mean feasible selection score, with mean known accuracy as the deterministic tie-breaker, then aggregates the selected lambda's per-global-class threshold quantiles.
8. Copies the existing formal ten-class checkpoint, OpenMax state, and distance statistics into a separate final directory, leaving the current automatic experiment untouched.
9. Fits formal validation score calibration, restores ten thresholds from quantiles, writes `fusion.json`, and evaluates the real unknown test set once.

- [ ] **Step 5: Add a dry-run test**

Use a temporary miniature NPZ fixture and monkeypatched training/artifact functions to verify that each class is held out once, no test-unknown path is read during calibration, and final thresholds have length ten.

- [ ] **Step 6: Run unit tests**

Run: `pytest tests/test_leave_class_out.py tests/test_main_calibration_config.py tests/test_fusion_calibration.py -q`

Expected: all tests pass.

### Task 4: Full training, comparison, and leakage audit

**Files:**
- Create: `outputs/oracle_leave_class_out_calibration/comparison.json`
- Create: `outputs/oracle_leave_class_out_calibration/comparison.md`
- Modify: `README.md`

- [ ] **Step 1: Capture immutable baselines**

Read the old manual metrics and current automatic metrics into a baseline JSON before starting the new final evaluation. Record source paths and timestamps.

- [ ] **Step 2: Train and calibrate all five folds**

Run: `python run_oracle_leave_class_out.py`

Expected: five `best_closed_set.pt` files, five fold calibration summaries, one aggregate calibration summary, and final open-set metrics.

- [ ] **Step 3: Verify fold coverage and isolation**

Programmatically assert:

```python
held_out_counts == {str(cls): 1 for cls in range(10)}
known_counts == {str(cls): 4 for cls in range(10)}
real_unknown_used_for_calibration is False
```

- [ ] **Step 4: Generate comparison tables**

For Known Acc, Unknown Recall, Macro F1, and AUROC, write:

- old manual value;
- current pseudo-only automatic value;
- new leave-class-out value;
- new minus old manual in percentage points;
- new minus current automatic in percentage points;
- direction marker: increase, decrease, or unchanged.

- [ ] **Step 5: Run the complete verification suite**

Run:

`pytest tests/test_leave_class_out.py tests/test_fusion_calibration.py tests/test_main_calibration_config.py tests/test_ablation_support.py tests/test_comparison_figures.py -q`

Expected: all tests pass.

- [ ] **Step 6: Inspect final artifacts**

Confirm `fusion.json` uses `leave_class_out_quantile_transfer`, has ten thresholds, includes the five-fold provenance, and contains no real unknown labels, predictions, or metrics in its calibration history.
