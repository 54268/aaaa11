from __future__ import annotations

from pathlib import Path

import numpy as np

import ablations.ablation_suite as ablation_suite
from ablations.ablation_suite import (
    LOSS_VARIANTS,
    MODULE_PIPELINE_OVERRIDES,
    MODULE_VARIANTS,
    ResultRow,
    SUBDIVISION_VARIANTS,
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
    embeddings = np.asarray([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0], dtype=np.int64)
    prototypes = np.asarray([[1.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    boundary = {
        "scores": np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        "local_scale": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "nearest_foreign": np.asarray([1, 1, 1], dtype=np.int64),
        "marginal_mask": np.asarray([True, True, True]),
        "critical_mask": np.asarray([True, False, False]),
        "ordinary_edge_mask": np.asarray([False, True, False]),
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
        [[3.075, 0.0], [5.075, 0.0], [7.075, 0.0]],
        atol=1e-6,
    )
    assert result["pseudo_kind"].tolist() == ["ordinary_edge", "ordinary_edge", "ordinary_edge"]
    assert result["source_indices"].tolist() == [0, 1, 2]


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
        "oscr",
    ]


def test_module_summary_exposes_module_sensitive_open_set_metrics() -> None:
    fields = getattr(ablation_suite, "_module_metric_fields", lambda: [])()

    assert fields == [
        ("known_accuracy", "Known Acc."),
        ("unknown_recall", "Unknown Recall"),
        ("macro_f1", "Macro F1"),
        ("oscr", "OSCR"),
    ]
    assert getattr(ablation_suite, "MODULE_OPEN_SET_KEYS", []) == [
        key for key, _ in fields
    ]


def test_module_table_uses_conventional_half_up_percentage_rounding() -> None:
    assert ablation_suite._format_percentage(0.97825) == "97.83%"


def test_summary_generates_four_metric_module_table_and_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ablation_suite, "ABLATION_ROOT", tmp_path)
    rows = []
    values = {
        "closed_set_only": (0.98, 0.0, 0.65, 0.96),
        "openmax_only": (0.80, 0.89, 0.80, 0.90),
        "ordinary_mbs_only": (0.87, 0.99, 0.93, 0.99),
        "full_method": (0.96, 0.97, 0.94, 0.99),
    }
    variants = {slug: name for slug, name, _ in MODULE_VARIANTS}
    for dataset in ["oracle", "wisig"]:
        for slug, metrics in values.items():
            rows.append(
                ResultRow(
                    category="modules",
                    dataset=dataset,
                    variant=variants[slug],
                    variant_slug=slug,
                    output_dir="unused",
                    metrics=dict(zip(ablation_suite.MODULE_OPEN_SET_KEYS, metrics)),
                )
            )

    write_summary(rows)

    markdown = (tmp_path / "消融结果汇总.md").read_text(encoding="utf-8")
    assert (tmp_path / "模块消融.png").exists()
    assert "| Known Acc. | Unknown Recall | Macro F1 | OSCR |" in markdown
    assert "Overall Acc." not in markdown
    assert "Unknown Precision" not in markdown
    assert "Known FPR" not in markdown
    assert "AUROC" not in markdown


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
    variant_map = {slug: overrides for slug, _, overrides in MODULE_VARIANTS}

    assert _module_metric_matrix_rows() == expected_rows
    assert [variant[0] for variant in MODULE_VARIANTS] == [slug for slug, _ in expected_rows]
    assert variant_map["closed_set_only"]["mode"] == "closed_set"
    assert variant_map["openmax_only"]["mode"] == "formal_openmax"
    assert variant_map["ordinary_mbs_only"]["use_critical_boundary"] is False
    assert variant_map["full_method"]["mode"] == "formal_pcbm"
    assert MODULE_PIPELINE_OVERRIDES == {}


def test_distance_module_uses_shared_fusion_settings_without_dataset_tuning() -> None:
    config = {
        "train": {},
        "pseudo_unknown": {},
        "fusion": {
            "classwise_known_weight": 0.45,
            "classwise_unknown_weight": 0.55,
            "classwise_min_known_accept": 0.88,
            "selection_weights": {"macro_f1": 1.0},
        },
    }
    overrides = {"use_critical_boundary": False, "fusion_lambda": None}

    ablation_suite._configure_module_pipeline_fusion(
        config,
        dataset="oracle",
        slug="ordinary_mbs_only",
        overrides=overrides,
        base_lambda=0.35,
    )

    assert config["fusion"]["lambda_grid"] == [0.35]
    assert config["fusion"]["manual_fusion_lambda"] == 0.35
    assert config["fusion"]["classwise_known_weight"] == 0.45
    assert config["fusion"]["classwise_unknown_weight"] == 0.55
    assert config["fusion"]["classwise_min_known_accept"] == 0.88
    assert config["fusion"]["selection_weights"] == {"macro_f1": 1.0}


def test_single_feature_subdivision_variants_do_not_include_filtering() -> None:
    variants = {slug: use_filtering for slug, _, _, use_filtering in SUBDIVISION_VARIANTS}

    assert variants["iq_descriptors_only"] is False
    assert variants["embedding_only"] is False
    assert variants["feature_fusion_wo_filtering"] is False
    assert variants["full_subdivision"] is True


def test_km_sensitivity_reuses_formal_rejection_outputs(tmp_path, monkeypatch) -> None:
    ablation_root = tmp_path / "ablations"
    formal_output = tmp_path / "formal_pcbm"
    checkpoint = tmp_path / "best_closed_set.pt"
    formal_output.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (formal_output / "open_set_metrics.json").write_text(
        '{"unknown_recall": 0.9670416666666667}\n',
        encoding="utf-8",
    )
    (formal_output / "open_set_predictions.csv").write_text(
        "y_true,y_pred,unknown_score,q_om,q_pd,d_min\n0,0,0.1,0.1,0.1,0.1\n",
        encoding="utf-8",
    )

    captured: list[dict] = []

    def fake_run_unknown_subdivision(config: dict) -> dict:
        captured.append(config)
        subdivision_dir = Path(config["project"]["output_dir"]) / "unknown_subdivision"
        subdivision_dir.mkdir(parents=True)
        (subdivision_dir / "m_selection_history.json").write_text("[]\n", encoding="utf-8")
        (subdivision_dir / "unknown_subdivision_metrics.json").write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(ablation_suite, "ABLATION_ROOT", ablation_root)
    monkeypatch.setitem(
        ablation_suite.DATASETS,
        "oracle",
        {
            **ablation_suite.DATASETS["oracle"],
            "checkpoint": checkpoint,
            "formal_output": formal_output,
        },
    )
    monkeypatch.setattr(
        ablation_suite,
        "_base_config",
        lambda dataset: {
            "project": {},
            "reporting": {},
            "eval": {},
            "unknown_subdivision": {},
        },
    )
    monkeypatch.setattr(ablation_suite, "_run_unknown_subdivision_only", fake_run_unknown_subdivision)

    output_dir = ablation_suite._run_reused_rejection_subdivision_variant(
        dataset="oracle",
        category="km",
        variant_slug="m_0_1_2_3_auto",
        variant_name="m=0,1,2,3,Auto",
        config_mutator=lambda config: config["unknown_subdivision"].update({"overcluster_extra_candidates": [0, 1]}),
    )

    assert (output_dir / "open_set_metrics.json").read_text(encoding="utf-8") == (
        formal_output / "open_set_metrics.json"
    ).read_text(encoding="utf-8")
    assert (output_dir / "open_set_predictions.csv").exists()
    assert captured[0]["unknown_subdivision"]["reuse_open_set_predictions"] is True
    assert captured[0]["unknown_subdivision"]["open_set_predictions_path"] == str(
        (output_dir / "open_set_predictions.csv").resolve()
    )


def test_subdivision_ablations_reuse_formal_rejection_outputs(tmp_path, monkeypatch) -> None:
    ablation_root = tmp_path / "ablations"
    captured: list[tuple[str, dict]] = []

    def forbidden_pipeline_variant(*args, **kwargs):
        raise AssertionError("subdivision ablations must not rerun the OSR pipeline")

    def fake_reused_variant(
        *,
        dataset: str,
        category: str,
        variant_slug: str,
        variant_name: str,
        config_mutator,
    ) -> Path:
        config = {
            "project": {},
            "reporting": {},
            "eval": {},
            "unknown_subdivision": {
                "direct_confidence_quantile": 0.02,
                "direct_min_cluster_size": 200,
            },
        }
        config_mutator(config)
        captured.append((variant_slug, config))

        output_dir = ablation_suite._variant_dir(category, dataset, variant_slug)
        subdivision_dir = output_dir / "unknown_subdivision"
        subdivision_dir.mkdir(parents=True)
        (subdivision_dir / "unknown_subdivision_metrics.json").write_text(
            (
                '{"nmi": 0.9, "ari": 0.8, "purity": 0.85, '
                '"hungarian_accuracy": 0.86, "coverage_of_total_test_unknown": 0.967, '
                '"resolved_num_clusters": 6, "uncertain_ratio": 0.0}\n'
            ),
            encoding="utf-8",
        )
        return output_dir

    monkeypatch.setattr(ablation_suite, "ABLATION_ROOT", ablation_root)
    monkeypatch.setattr(
        ablation_suite,
        "_base_config",
        lambda dataset: {
            "unknown_subdivision": {
                "direct_confidence_quantile": 0.02,
                "direct_min_cluster_size": 200,
            },
        },
    )
    monkeypatch.setattr(ablation_suite, "_run_pipeline_variant", forbidden_pipeline_variant)
    monkeypatch.setattr(ablation_suite, "_run_reused_rejection_subdivision_variant", fake_reused_variant)

    rows = ablation_suite.run_subdivision_ablations("oracle")

    assert [row.variant_slug for row in rows] == [slug for slug, *_ in SUBDIVISION_VARIANTS]
    assert [slug for slug, _ in captured] == [slug for slug, *_ in SUBDIVISION_VARIANTS]
    by_slug = {slug: config["unknown_subdivision"] for slug, config in captured}
    assert by_slug["embedding_only"]["direct_confidence_quantile"] == 0.0
    assert by_slug["embedding_only"]["direct_min_cluster_size"] == 0
    assert by_slug["feature_fusion_wo_filtering"]["direct_confidence_quantile"] == 0.0
    assert by_slug["feature_fusion_wo_filtering"]["merge_extra_clusters_to_target"] is True
    assert by_slug["full_subdivision"]["direct_confidence_quantile"] == 0.02
    assert by_slug["full_subdivision"]["direct_min_cluster_size"] == 200
    assert by_slug["full_subdivision"]["merge_extra_clusters_to_target"] is True
