from __future__ import annotations

from pathlib import Path

from figures.generate_comparison_figures import load_prediction_csv


def test_load_prediction_csv_supports_project_and_baseline_formats(tmp_path: Path) -> None:
    project_csv = tmp_path / "project.csv"
    project_csv.write_text(
        "y_true,y_pred,unknown_score,q_om,q_pd,d_min\n0,0,0.1,0.1,0.1,0.1\n2,2,0.9,0.9,0.9,0.9\n",
        encoding="utf-8",
    )
    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text(
        "y_true,y_pred,unknown_score,is_unknown,unknown_label\n0,0,0.2,0,2\n2,2,0.8,1,2\n",
        encoding="utf-8",
    )

    project = load_prediction_csv(project_csv, fallback_unknown_label=2)
    baseline = load_prediction_csv(baseline_csv)

    assert project.unknown_label == 2
    assert baseline.unknown_label == 2
    assert project.y_true.tolist() == baseline.y_true.tolist() == [0, 2]
