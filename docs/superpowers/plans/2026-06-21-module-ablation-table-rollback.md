# Module Ablation Table Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the module ablation to a pure closed-set baseline followed by OpenMax, prototype-distance calibration, and full PCBM, while showing only Known Acc., Unknown Recall, Macro F1, and AUROC.

**Architecture:** Reuse the existing four ablation result directories and change only their orchestration and presentation. The pure closed-set row is recomputed from the shared checkpoint without unknown rejection; the remaining rows continue to load or generate their existing real experiment results. Markdown, CSV/JSON summaries, and the table PNG are generated from the same `ResultRow` objects.

**Tech Stack:** Python, NumPy, PyTorch, matplotlib, pytest.

---

### Task 1: Lock the desired module-ablation contract

**Files:**
- Modify: `tests/test_ablation_support.py`

- [ ] **Step 1: Update the module metric test**

Assert that `_module_metric_fields()` contains exactly `known_accuracy`, `unknown_recall`, `macro_f1`, and `auroc`.

- [ ] **Step 2: Add a pure closed-set variant test**

Assert that `MODULE_VARIANTS[0]` has slug `closed_set_only`, mode `closed_set`, and that the obsolete confidence-quantile baseline is no longer selected.

- [ ] **Step 3: Add a generated-figure contract test**

Create four small `ResultRow` fixtures for Oracle and WiSig, call `write_summary`, and assert that `模块消融.png` exists and the Markdown table contains the four metric headers but not Overall Acc., Unknown Precision, Known FPR, or OSCR.

- [ ] **Step 4: Run tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_ablation_support.py -q
```

Expected: failures showing the current confidence-rejection baseline, eight metric columns, and deleted module figure.

### Task 2: Restore the pure closed-set baseline and four-metric summary

**Files:**
- Modify: `ablations/ablation_suite.py`

- [ ] **Step 1: Restore the module variant**

Set the first module variant to `("closed_set_only", "闭集原型分类", {"mode": "closed_set"})`.

- [ ] **Step 2: Restore pure closed-set evaluation**

Replace `_run_basic_confidence_rejection_module_baseline` with `_run_closed_set_module_baseline`. It must predict only the nearest known prototype, compute `1 - max_softmax` solely as the AUROC score, and save `threshold_mode="none"` without applying a rejection threshold.

- [ ] **Step 3: Route the module runner**

Make `run_module_ablations` call the closed-set evaluator for mode `closed_set`.

- [ ] **Step 4: Reduce module metric fields**

Return only:

```python
[
    ("known_accuracy", "Known Acc."),
    ("unknown_recall", "Unknown Recall"),
    ("macro_f1", "Macro F1"),
    ("auroc", "AUROC"),
]
```

- [ ] **Step 5: Restore the module table figure**

Generate `模块消融.png` from the same switch matrix and metric fields used by Markdown. Arrange Oracle and WiSig vertically, format metrics as percentages, alternate row shading, and bold the best value in each metric column.

- [ ] **Step 6: Update explanatory text**

State that the first row is pure closed-set classification with no unknown output, the second row is formal OpenMax, the third adds prototype-distance calibration, and the final row is PCBM. Describe the result as an overall improvement rather than strict per-metric monotonicity.

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```powershell
D:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_ablation_support.py -q
```

Expected: all tests pass.

### Task 3: Update documentation and rebuild artifacts

**Files:**
- Modify: `ablations/README.md`
- Modify: `ablations/消融结果汇总.md`
- Modify: `ablations/消融结果汇总.json`
- Modify: `ablations/消融结果汇总.csv`
- Create: `ablations/模块消融.png`

- [ ] **Step 1: Update README**

Document the four-stage sequence and four displayed metrics.

- [ ] **Step 2: Recompute the two pure closed-set rows**

Run only the module ablation for Oracle and WiSig using the existing checkpoints so `closed_set_only/open_set_metrics.json` records `threshold_mode="none"` and `unknown_recall=0`.

- [ ] **Step 3: Rebuild summaries**

Collect existing result directories and regenerate Markdown, JSON, CSV, and PNG from the current code.

- [ ] **Step 4: Inspect generated data**

Verify that the Markdown module tables have seven columns total, the first-row Unknown Recall is zero, and values match the source JSON files.

- [ ] **Step 5: Inspect the PNG**

Open `ablations/模块消融.png` and confirm both datasets, all headers, all four rows, and all percentages are readable and unclipped.

### Task 4: Final verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused tests**

```powershell
D:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_ablation_support.py -q
```

- [ ] **Step 2: Run project checks**

```powershell
D:\Anaconda3\envs\pytorch\python.exe check_project.py
```

- [ ] **Step 3: Audit changed files**

Use `git diff --check` and inspect `git diff` only for the planned ablation files.
