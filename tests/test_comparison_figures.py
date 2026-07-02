from __future__ import annotations

from pathlib import Path
import inspect

import matplotlib.pyplot as plt
import numpy as np

from figures.generate_comparison_figures import confusion_output_paths, load_prediction_csv
from figures.generate_tsne import FORMAL_OUTPUT_DIRS, build_experiment_config, plot_global_tsne


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


def test_tsne_generation_prefers_current_formal_output_dir(tmp_path: Path, monkeypatch) -> None:
    formal_output = tmp_path / "outputs" / "oracle_supervised_calibrator" / "final"
    formal_output.mkdir(parents=True)
    monkeypatch.setitem(FORMAL_OUTPUT_DIRS, "oracle", formal_output)

    config = build_experiment_config("oracle")

    assert Path(config["project"]["output_dir"]) == formal_output.resolve()


def test_tsne_plot_uses_paper_readable_axis_text_without_legend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
        ],
        dtype=np.float32,
    )
    prototypes = np.asarray([[0.5, 0.2], [2.5, 1.2]], dtype=np.float32)
    y_true = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)

    plot_global_tsne(points, prototypes, y_true, 2, tmp_path / "tsne_oracle.png")

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    tick_sizes = [tick.label1.get_fontsize() for tick in ax.xaxis.get_major_ticks()]
    assert tick_sizes
    assert min(tick_sizes) >= 12.0
    assert ax.get_legend() is None
    plt.close(fig)
