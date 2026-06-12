from __future__ import annotations

import numpy as np

import ablations.ablation_suite as ablation_suite
from ablations.ablation_suite import (
    LOSS_VARIANTS,
    MODULE_VARIANTS,
    ResultRow,
    _ablation_matrix_metric_table,
    _module_metric_matrix_rows,
    _selected_loss_variants,
    write_summary,
)
from functions.methods.pseudo_unknown import generate_hybrid_pseudo_unknown
from functions.subdivision_pipeline import build_cluster_features


def test_iq_stats_feature_mode_does_not_include_embeddings() -> None:
    embeddings = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    distances = np.zeros((2, 2), dtype=np.float32)
    scores = np.zeros(2, dtype=np.float32)
    known_pred = np.zeros(2, dtype=np.int64)
    prototypes = np.zeros((2, 2), dtype=np.float32)
    signal_samples = np.asarray(
        [
            [[1.0, -1.0, 1.0, -1.0], [0.5, 0.5, -0.5, -0.5]],
            [[2.0, 2.0, -2.0, -2.0], [1.0, -1.0, 1.0, -1.0]],
        ],
        dtype=np.float32,
    )

    features = build_cluster_features(
        "iq_stats",
        embeddings,
        distances,
        scores,
        scores,
        scores,
        known_pred,
        prototypes,
        signal_samples=signal_samples,
    )

    assert features.shape[0] == 2
    assert features.shape[1] > embeddings.shape[1]
    assert not np.allclose(features[:, :2], embeddings)


def test_unified_mbs_generation_treats_pcbs_as_ordinary_edges() -> None:
    embeddings = np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    labels = np.asarray([0, 0], dtype=np.int64)
    prototypes = np.asarray([[1.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    boundary = {
        "scores": np.asarray([0.5, 0.5], dtype=np.float32),
        "local_scale": np.asarray([1.0, 1.0], dtype=np.float32),
        "nearest_foreign": np.asarray([1, 1], dtype=np.int64),
        "critical_mask": np.asarray([True, False]),
        "ordinary_edge_mask": np.asarray([False, True]),
    }

    result = generate_hybrid_pseudo_unknown(
        embeddings=embeddings,
        labels=labels,
        prototypes=prototypes,
        boundary_result=boundary,
        ordinary_eta=1.0,
        critical_eta=1.0,
        critical_beta=0.0,
        ordinary_variations=1,
        critical_variations=1,
        jitter=0.0,
        use_critical_boundary=False,
        seed=42,
    )

    np.testing.assert_allclose(
        result["pseudo_embeddings"],
        [[3.075, 0.0], [5.075, 0.0]],
        atol=1e-6,
    )
    assert result["pseudo_kind"].tolist() == ["ordinary_edge", "ordinary_edge"]
    assert result["source_indices"].tolist() == [0, 1]


def test_loss_ablation_can_select_one_resumable_variant() -> None:
    selected = _selected_loss_variants("ce_angular")

    assert [variant[0] for variant in selected] == ["ce_angular"]
    assert len(_selected_loss_variants("all")) == 4


def test_loss_ablation_uses_consistent_component_weights() -> None:
    variants = {slug: (angle, prototype) for slug, _, angle, prototype in LOSS_VARIANTS}

    assert variants["full_embedding_learning"][0] == variants["ce_angular"][0]
    assert variants["full_embedding_learning"][1] == variants["ce_prototype"][1]


def test_loss_summary_uses_open_set_task_metrics_only() -> None:
    fields = getattr(ablation_suite, "_loss_metric_fields", lambda: [])()

    assert [key for key, _ in fields] == [
        "known_accuracy",
        "unknown_recall",
        "macro_f1",
        "auroc",
    ]


def test_summary_removes_deprecated_config_only_figures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ablation_suite, "ABLATION_ROOT", tmp_path)
    deprecated = [tmp_path / "模块消融.png", tmp_path / "损失函数消融.png"]
    for path in deprecated:
        path.write_bytes(b"stale")

    write_summary([])

    assert all(not path.exists() for path in deprecated)


def test_ablation_table_starts_with_switch_columns() -> None:
    row = ResultRow(
        category="losses",
        dataset="oracle",
        variant="CE only",
        variant_slug="ce_only",
        output_dir="unused",
        metrics={"overall_accuracy": 0.9},
    )

    table = _ablation_matrix_metric_table(
        [row],
        [("ce_only", [True, False, False])],
        ["Classification Loss", "Angular Loss", "Prototype Loss"],
        [("overall_accuracy", "Overall Acc.")],
    )

    assert table[0].startswith(
        "| Classification Loss | Angular Loss | Prototype Loss |"
    )
    assert table[2] == "| √ | X | X | 0.900000 |"


def test_module_ablation_rows_are_cumulative_additions() -> None:
    expected_rows = [
        ("closed_set_only", [False, False, False]),
        ("openmax_only", [False, False, True]),
        ("ordinary_mbs_only", [False, True, True]),
        ("full_method", [True, True, True]),
    ]

    assert _module_metric_matrix_rows() == expected_rows
    assert [variant[0] for variant in MODULE_VARIANTS] == [slug for slug, _ in expected_rows]
