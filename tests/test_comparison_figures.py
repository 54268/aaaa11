from __future__ import annotations

from pathlib import Path
import inspect

from figures.generate_comparison_figures import confusion_output_paths, load_prediction_csv
from figures.generate_tsne import plot_global_tsne


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


def test_confusion_matrices_use_separate_dataset_files(tmp_path: Path) -> None:
    paths = confusion_output_paths(tmp_path)

    assert paths == {
        "oracle": tmp_path / "confusion_matrix_oracle.png",
        "wisig": tmp_path / "confusion_matrix_wisig.png",
    }
    assert all("oracle_wisig" not in path.name for path in paths.values())


def test_tsne_plot_only_requires_true_labels() -> None:
    params = list(inspect.signature(plot_global_tsne).parameters)

    assert params == ["points_2d", "prototypes_2d", "y_true", "unknown_label", "save_path"]
