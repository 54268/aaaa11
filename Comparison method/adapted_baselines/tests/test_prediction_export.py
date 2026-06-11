from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainers import _save_open_set_predictions


def test_save_open_set_predictions_writes_roc_ready_columns(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _save_open_set_predictions(
        path,
        y_true=np.asarray([0, 2], dtype=np.int64),
        y_pred=np.asarray([0, 2], dtype=np.int64),
        unknown_score=np.asarray([0.1, 0.9], dtype=np.float32),
        unknown_label=2,
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0].keys() == {
        "y_true",
        "y_pred",
        "unknown_score",
        "is_unknown",
        "unknown_label",
    }
    assert rows[0]["is_unknown"] == "0"
    assert rows[1]["is_unknown"] == "1"
