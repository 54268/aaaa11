# 伪未知驱动自动融合校准实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 取消最终实验中的手动开放集参数，使 Oracle 和 WiSig 仅利用已知验证集与特征层伪未知集自动搜索融合权重和拒识阈值。

**Architecture:** 扩展现有 `search_fusion_params`，通过正式配置与运行入口移除手动分支。Oracle 在全局已知准确率约束下联合选择各类别阈值，WiSig 使用全局阈值；两者共享固定融合网格、选择权重和可行性检查。复用已有闭集检查点，仅刷新边界、伪未知、OpenMax、融合、开放集评估和未知类细分。

**Tech Stack:** Python、NumPy、PyTorch、pytest。

---

### Task 1: 配置行为回归测试

**Files:**
- Create: `tests/test_main_calibration_config.py`
- Test: `tests/test_main_calibration_config.py`

- [ ] **Step 1: Write the failing test**

```python
from run_oracle import build_config as build_oracle_config
from run_wisig import build_config as build_wisig_config


def _assert_shared_auto_search(config: dict) -> None:
    fusion = config["fusion"]
    assert fusion["lambda_grid"] == [i / 10 for i in range(11)]
    assert fusion.get("manual_threshold") is None
    assert fusion.get("manual_thresholds_per_class") is None
    assert fusion["known_rescue"] == {"enabled": False}
    assert fusion["selection_weights"] == {
        "known_accuracy": 0.45,
        "unknown_recall": 0.35,
        "macro_f1": 0.15,
        "auroc": 0.05,
    }
    assert fusion["require_feasible"] is True


def test_oracle_uses_pseudo_unknown_auto_calibration() -> None:
    config = build_oracle_config()
    _assert_shared_auto_search(config)
    assert config["fusion"]["threshold_mode"] == "classwise_balanced"
    assert config["fusion"]["score_calibration"] == "classwise_z"
    assert config["fusion"]["classwise_min_known_accept"] == 0.99
    assert config["fusion"]["min_known_accuracy"] == 0.96


def test_wisig_uses_pseudo_unknown_auto_calibration() -> None:
    config = build_wisig_config()
    _assert_shared_auto_search(config)
    assert config["fusion"]["threshold_mode"] == "global"
    assert config["fusion"].get("score_calibration", "none") == "none"
    assert config["fusion"]["min_known_accuracy"] == 0.985
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Anaconda3\envs\pytorch\python.exe -m pytest tests\test_main_calibration_config.py -q`

Expected: FAIL because current main configs still contain manual thresholds and one-point lambda grids.

### Task 2: 切换正式实验配置

**Files:**
- Modify: `settings/oracle_settings.py`
- Modify: `settings/wisig_settings.py`
- Modify: `run_oracle.py`
- Modify: `run_wisig.py`

- [ ] **Step 1: Set shared automatic search protocol**

Set both datasets to:

```python
lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
manual_threshold = None
manual_thresholds_per_class = None
known_rescue = {"enabled": False}
selection_weights = {
    "known_accuracy": 0.45,
    "unknown_recall": 0.35,
    "macro_f1": 0.15,
    "auroc": 0.05,
}
min_known_accuracy = 0.95
```

Oracle additionally uses `threshold_mode="classwise_joint"` and `score_calibration="classwise_z"`，在全局已知准确率约束下联合选择各类别阈值。WiSig uses `threshold_mode="global"` and `score_calibration="none"`.

- [ ] **Step 2: Run focused tests**

Run: `D:\Anaconda3\envs\pytorch\python.exe -m pytest tests\test_main_calibration_config.py tests\test_fusion_calibration.py -q`

Expected: PASS.

### Task 3: 刷新正式实验

**Files:**
- Regenerate: `outputs/oracle_kri16_demod_known_first/*`
- Regenerate: `outputs/wisig_singleday_osr_k16_u12/*`

- [ ] **Step 1: Save pre-change metrics outside output directories**

Save the nine comparison metrics and old threshold modes before rerunning.

- [ ] **Step 2: Run Oracle with existing checkpoint**

Run: `D:\Anaconda3\envs\pytorch\python.exe run_oracle.py`

Expected: no data preparation or closed-set retraining; `fusion.json` reports an automatic threshold mode.

- [ ] **Step 3: Run WiSig with existing checkpoint**

Run: `D:\Anaconda3\envs\pytorch\python.exe run_wisig.py`

Expected: no data preparation or closed-set retraining; `fusion.json` reports `threshold_mode="global"` selected automatically.

### Task 4: 结果审计

**Files:**
- Inspect: `outputs/*/fusion.json`
- Inspect: `outputs/*/open_set_metrics.json`

- [ ] **Step 1: Verify parameter provenance**

Confirm that both result files have no manual threshold strategy and that selected `lambda` belongs to the predefined grid.

- [ ] **Step 2: Compare metrics**

For each dataset report old value, new value, absolute change and direction for Overall Acc.、Known Acc.、Unknown Precision、Unknown Recall、Known FPR、Macro F1、AUROC、FPR95、OSCR.

- [ ] **Step 3: Run final tests**

Run: `D:\Anaconda3\envs\pytorch\python.exe -m pytest tests\test_main_calibration_config.py tests\test_fusion_calibration.py tests\test_ablation_support.py -q`

Expected: all tests pass.
