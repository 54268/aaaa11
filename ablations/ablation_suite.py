from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


ABLATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.common.io import ensure_dir, load_json, save_json
from functions.common.metrics import evaluate_open_set, save_confusion_matrix, save_prediction_csv
from functions.common.visualization import generate_open_set_figures
from functions.data.data_build import build_data_module
from functions.methods.prototype_utils import predict_with_prototypes
from functions.model.closed_set import ClosedSetTrainer
from functions.pipeline import generate_pseudo_unknown_artifacts, mine_boundary_artifacts, run_osr_pipeline
from run_oracle import build_config as build_oracle_config
from run_wisig import build_config as build_wisig_config


GROUP_DIRS = {
    "modules": "01_模块消融",
    "km": "02_KM簇数消融",
    "losses": "03_损失函数消融",
    "subdivision": "04_细分流程消融",
}

DATASETS = {
    "oracle": {
        "display": "Oracle",
        "formal_name": "oracle_kri16_demod",
        "build_config": build_oracle_config,
        "checkpoint": PROJECT_ROOT / "outputs" / "oracle_kri16_demod_known_first" / "best_closed_set.pt",
        "formal_output": PROJECT_ROOT / "outputs" / "oracle_kri16_demod_known_first",
    },
    "wisig": {
        "display": "WiSig",
        "formal_name": "wisig_singleday_osr_k16_u12",
        "build_config": build_wisig_config,
        "checkpoint": PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12" / "best_closed_set.pt",
        "formal_output": PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12",
    },
}

MODULE_VARIANTS = [
    ("closed_set_only", "普通 MBS（基础拒识）", {"mode": "confidence_rejection"}),
    (
        "openmax_only",
        "OpenMax 校准",
        {
            "mode": "pipeline",
            "use_critical_boundary": False,
            "fusion_lambda": 1.0,
            "score_calibration": "none",
        },
    ),
    (
        "ordinary_mbs_only",
        "OpenMax + 原型距离校准",
        {"mode": "pipeline", "use_critical_boundary": False, "fusion_lambda": None},
    ),
    (
        "full_method",
        "完整方法",
        {"mode": "formal_pcbm"},
    ),
]

BASE_CONFIDENCE_REJECTION_QUANTILE = 0.85

MODULE_PIPELINE_OVERRIDES = {
    ("oracle", "ordinary_mbs_only"): {
        "fusion_lambda_grid": [0.1, 0.3, 0.5, 0.7, 0.9],
        "classwise_known_weight": 0.75,
        "classwise_unknown_weight": 0.25,
        "classwise_min_known_accept": 0.90,
        "selection_weights": {
            "known_accuracy": 0.30,
            "unknown_recall": 0.20,
            "macro_f1": 0.20,
            "oscr": 0.20,
            "auroc": 0.10,
        },
    }
}

LOSS_VARIANTS = [
    ("ce_only", "CE only", 0.0, 0.0),
    ("ce_angular", "CE + Angular", 0.15, 0.0),
    ("ce_prototype", "CE + Prototype", 0.0, 0.10),
    ("full_embedding_learning", "Full embedding learning", 0.15, 0.10),
]

SUBDIVISION_VARIANTS = [
    ("embedding_only", "Embedding only", "embedding", False),
    ("iq_descriptors_only", "I/Q descriptors only", "iq_stats", False),
    ("feature_fusion_wo_filtering", "Feature fusion w/o filtering", "embedding_iq_stats", False),
    ("full_subdivision", "Full subdivision", "embedding_iq_stats", True),
]

CORE_OPEN_SET_KEYS = [
    "overall_accuracy",
    "known_accuracy",
    "unknown_recall",
    "macro_f1",
    "auroc",
]

MODULE_OPEN_SET_KEYS = [
    "overall_accuracy",
    "known_accuracy",
    "unknown_recall",
    "unknown_precision",
    "known_fpr_as_unknown",
    "macro_f1",
    "oscr",
    "auroc",
]

SUBDIVISION_KEYS = [
    "nmi",
    "ari",
    "purity",
    "hungarian_accuracy",
    "coverage_of_total_test_unknown",
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ResultRow:
    category: str
    dataset: str
    variant: str
    variant_slug: str
    output_dir: str
    metrics: dict[str, Any]

    def flat(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "dataset": self.dataset,
            "variant": self.variant,
            "variant_slug": self.variant_slug,
            "output_dir": self.output_dir,
            **self.metrics,
        }


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _base_config(dataset: str) -> dict[str, Any]:
    return deepcopy(DATASETS[dataset]["build_config"]())


def _variant_dir(category: str, dataset: str, variant_slug: str) -> Path:
    return ABLATION_ROOT / GROUP_DIRS[category] / dataset / variant_slug


def _configure_output(config: dict[str, Any], output_dir: Path, experiment_name: str) -> None:
    config["project"]["name"] = experiment_name
    config["project"]["output_dir"] = str(output_dir.resolve())
    config["reporting"]["write_root_summaries"] = False
    config["reporting"]["write_figures_inside_output_dir"] = True
    config["eval"]["save_predictions"] = True


def _write_manifest(
    output_dir: Path,
    dataset: str,
    category: str,
    variant: str,
    config: dict[str, Any],
    checkpoint: Path | None,
) -> None:
    save_json(
        output_dir / "ablation_manifest.json",
        {
            "dataset": dataset,
            "category": category,
            "variant": variant,
            "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
            "config": config,
        },
    )


def _prepare_checkpoint(output_dir: Path, source_checkpoint: Path) -> None:
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {source_checkpoint}")
    ensure_dir(output_dir)
    target = output_dir / "best_closed_set.pt"
    if source_checkpoint.resolve() != target.resolve():
        shutil.copy2(source_checkpoint, target)
    train_summary = source_checkpoint.parent / "train_summary.json"
    if train_summary.exists():
        shutil.copy2(train_summary, output_dir / "train_summary_from_base.json")


def _run_pipeline_variant(
    *,
    dataset: str,
    category: str,
    variant_slug: str,
    variant_name: str,
    config_mutator: Callable[[dict[str, Any]], None],
    train: bool,
    with_unknown_subdivision: bool,
) -> Path:
    output_dir = _variant_dir(category, dataset, variant_slug)
    config = _base_config(dataset)
    _configure_output(config, output_dir, f"{dataset}_{category}_{variant_slug}")
    config_mutator(config)
    checkpoint = None if train else Path(DATASETS[dataset]["checkpoint"])
    ensure_dir(output_dir)
    if checkpoint is not None:
        _prepare_checkpoint(output_dir, checkpoint)
    _write_manifest(output_dir, dataset, category, variant_name, config, checkpoint)
    run_osr_pipeline(
        config,
        skip_prepare=True,
        skip_training=not train,
        ckpt_path=checkpoint,
        with_unknown_subdivision=with_unknown_subdivision,
    )
    return output_dir


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def _confidence_unknown_score(logits: np.ndarray) -> np.ndarray:
    return 1.0 - _stable_softmax(logits).max(axis=1)


def _global_known_quantile_threshold(scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(scores, dtype=np.float64), float(quantile)))


def _run_basic_confidence_rejection_module_baseline(dataset: str, variant_slug: str, variant_name: str) -> Path:
    output_dir = _variant_dir("modules", dataset, variant_slug)
    config = _base_config(dataset)
    _configure_output(config, output_dir, f"{dataset}_modules_{variant_slug}")
    config["train"]["device"] = "cuda"
    config["pseudo_unknown"]["use_critical_boundary"] = False
    checkpoint = Path(DATASETS[dataset]["checkpoint"])
    ensure_dir(output_dir)
    _prepare_checkpoint(output_dir, checkpoint)
    _write_manifest(output_dir, dataset, "modules", variant_name, config, checkpoint)

    mine_boundary_artifacts(config, ckpt_path=checkpoint)
    generate_pseudo_unknown_artifacts(config)

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config,
        datamodule.bundle.num_known_classes,
        datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(checkpoint)
    val_known = trainer.extract_embeddings(datamodule.val_known_dataloader())
    test_known = trainer.extract_embeddings(datamodule.test_known_dataloader())
    test_unknown = trainer.extract_embeddings(datamodule.test_unknown_dataloader())
    _, val_logits, _ = predict_with_prototypes(
        val_known["embeddings"],
        val_known["prototypes"],
        float(config["model"]["temperature"]),
    )
    all_embeddings = np.concatenate(
        [test_known["embeddings"], test_unknown["embeddings"]],
        axis=0,
    )
    unknown_label = datamodule.bundle.num_known_classes
    all_labels = np.concatenate(
        [
            test_known["labels"],
            np.full(len(test_unknown["labels"]), unknown_label, dtype=np.int64),
        ]
    )
    known_pred, logits, distances = predict_with_prototypes(
        all_embeddings,
        test_known["prototypes"],
        float(config["model"]["temperature"]),
    )
    val_unknown_score = _confidence_unknown_score(val_logits)
    threshold = _global_known_quantile_threshold(
        val_unknown_score,
        BASE_CONFIDENCE_REJECTION_QUANTILE,
    )
    unknown_score = _confidence_unknown_score(logits)
    y_pred = known_pred.copy()
    y_pred[unknown_score > threshold] = unknown_label
    metrics = evaluate_open_set(all_labels, y_pred, unknown_score, unknown_label)
    metrics.update(
        {
            "threshold_strategy_used": "global_confidence_known_quantile",
            "threshold_mode": "global_confidence_known_quantile",
            "threshold": threshold,
            "threshold_quantile": BASE_CONFIDENCE_REJECTION_QUANTILE,
            "known_classes": int(unknown_label),
            "known_val_sample_count": int(len(val_known["labels"])),
            "test_known_sample_count": int(len(test_known["labels"])),
            "test_unknown_sample_count": int(len(test_unknown["labels"])),
            "output_dir": str(output_dir),
        }
    )
    save_json(output_dir / "open_set_metrics.json", metrics)
    save_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        all_labels,
        y_pred,
        labels=list(range(unknown_label)) + [unknown_label],
    )
    unavailable = np.full_like(unknown_score, np.nan, dtype=np.float64)
    d_min = distances[np.arange(len(distances)), known_pred]
    save_prediction_csv(
        output_dir / "open_set_predictions.csv",
        all_labels,
        y_pred,
        unknown_score,
        unavailable,
        unavailable,
        d_min,
    )
    generate_open_set_figures(
        config=config,
        output_dir=output_dir,
        y_true=all_labels,
        y_pred=y_pred,
        unknown_score=unknown_score,
        unknown_label=unknown_label,
        threshold=threshold,
    )
    return output_dir


def _run_formal_openmax_module_baseline(dataset: str, variant_slug: str, variant_name: str) -> Path:
    output_dir = _variant_dir("modules", dataset, variant_slug)
    config = _base_config(dataset)
    _configure_output(config, output_dir, f"{dataset}_modules_{variant_slug}")
    checkpoint = Path(DATASETS[dataset]["checkpoint"])
    ensure_dir(output_dir)
    _prepare_checkpoint(output_dir, checkpoint)

    formal_name = str(DATASETS[dataset]["formal_name"])
    source_dir = (
        PROJECT_ROOT
        / "Comparison method"
        / "adapted_results"
        / "formal"
        / formal_name
        / "openmax"
    )
    metrics_path = source_dir / f"OpenMax_{formal_name}_seed42_metrics.json"
    predictions_path = source_dir / f"OpenMax_{formal_name}_seed42_predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing formal OpenMax metrics: {metrics_path}")

    metrics = load_json(metrics_path)
    save_json(output_dir / "open_set_metrics.json", metrics)
    if predictions_path.exists():
        shutil.copy2(predictions_path, output_dir / "open_set_predictions.csv")
    save_json(
        output_dir / "ablation_manifest.json",
        {
            "dataset": dataset,
            "category": "modules",
            "variant": variant_name,
            "checkpoint": str(checkpoint.resolve()),
            "source_metrics": str(metrics_path.resolve()),
            "source_predictions": str(predictions_path.resolve()) if predictions_path.exists() else None,
            "config": config,
        },
    )
    return output_dir


def _run_formal_pcbm_module_result(dataset: str, variant_slug: str, variant_name: str) -> Path:
    output_dir = _variant_dir("modules", dataset, variant_slug)
    config = _base_config(dataset)
    _configure_output(config, output_dir, f"{dataset}_modules_{variant_slug}")
    checkpoint = Path(DATASETS[dataset]["checkpoint"])
    ensure_dir(output_dir)
    _prepare_checkpoint(output_dir, checkpoint)

    source_dir = Path(DATASETS[dataset]["formal_output"])
    metrics_path = source_dir / "open_set_metrics.json"
    predictions_path = source_dir / "open_set_predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing formal PCBM metrics: {metrics_path}")

    metrics = load_json(metrics_path)
    save_json(output_dir / "open_set_metrics.json", metrics)
    if predictions_path.exists():
        shutil.copy2(predictions_path, output_dir / "open_set_predictions.csv")
    save_json(
        output_dir / "ablation_manifest.json",
        {
            "dataset": dataset,
            "category": "modules",
            "variant": variant_name,
            "checkpoint": str(checkpoint.resolve()),
            "source_metrics": str(metrics_path.resolve()),
            "source_predictions": str(predictions_path.resolve()) if predictions_path.exists() else None,
            "config": config,
        },
    )
    return output_dir


def _configure_module_pipeline_fusion(
    config: dict[str, Any],
    dataset: str,
    slug: str,
    overrides: dict[str, Any],
    base_lambda: float,
) -> None:
    config["train"]["device"] = "cuda"
    config["pseudo_unknown"]["use_critical_boundary"] = bool(overrides["use_critical_boundary"])
    profile = MODULE_PIPELINE_OVERRIDES.get((dataset, slug), {})

    lambda_grid = profile.get("fusion_lambda_grid")
    if lambda_grid:
        chosen_grid = [float(value) for value in lambda_grid]
        config["fusion"]["lambda_grid"] = chosen_grid
        config["fusion"]["manual_fusion_lambda"] = chosen_grid[0]
    else:
        chosen_lambda = profile.get("fusion_lambda", overrides["fusion_lambda"])
        if chosen_lambda is None:
            chosen_lambda = base_lambda
        config["fusion"]["manual_fusion_lambda"] = float(chosen_lambda)
        config["fusion"]["lambda_grid"] = [float(chosen_lambda)]

    config["fusion"]["manual_threshold"] = None
    config["fusion"]["manual_thresholds_per_class"] = None
    config["fusion"]["threshold_mode"] = "classwise_balanced"
    config["fusion"]["known_rescue"] = {"enabled": False}
    if "score_calibration" in overrides:
        config["fusion"]["score_calibration"] = str(overrides["score_calibration"])
    config["fusion"]["classwise_known_weight"] = float(
        profile.get("classwise_known_weight", config["fusion"].get("classwise_known_weight", 0.50))
    )
    config["fusion"]["classwise_unknown_weight"] = float(
        profile.get("classwise_unknown_weight", config["fusion"].get("classwise_unknown_weight", 0.50))
    )
    config["fusion"]["classwise_min_known_accept"] = float(
        profile.get(
            "classwise_min_known_accept",
            config["fusion"].get("classwise_min_known_accept", 0.88),
        )
    )
    if "selection_weights" in profile:
        config["fusion"]["selection_weights"] = dict(profile["selection_weights"])


def run_module_ablations(dataset: str) -> list[ResultRow]:
    rows: list[ResultRow] = []
    base_lambda = float(_base_config(dataset)["fusion"].get("manual_fusion_lambda", 0.5))
    for slug, name, overrides in MODULE_VARIANTS:
        if overrides["mode"] == "confidence_rejection":
            output_dir = _run_basic_confidence_rejection_module_baseline(dataset, slug, name)
            metrics = load_json(output_dir / "open_set_metrics.json")
            rows.append(
                ResultRow(
                    category="modules",
                    dataset=dataset,
                    variant=name,
                    variant_slug=slug,
                    output_dir=str(output_dir),
                    metrics={key: metrics.get(key) for key in MODULE_OPEN_SET_KEYS},
                )
            )
            continue
        if overrides["mode"] == "formal_openmax":
            output_dir = _run_formal_openmax_module_baseline(dataset, slug, name)
            metrics = load_json(output_dir / "open_set_metrics.json")
            rows.append(
                ResultRow(
                    category="modules",
                    dataset=dataset,
                    variant=name,
                    variant_slug=slug,
                    output_dir=str(output_dir),
                    metrics={key: metrics.get(key) for key in MODULE_OPEN_SET_KEYS},
                )
            )
            continue
        if overrides["mode"] == "formal_pcbm":
            output_dir = _run_formal_pcbm_module_result(dataset, slug, name)
            metrics = load_json(output_dir / "open_set_metrics.json")
            rows.append(
                ResultRow(
                    category="modules",
                    dataset=dataset,
                    variant=name,
                    variant_slug=slug,
                    output_dir=str(output_dir),
                    metrics={key: metrics.get(key) for key in MODULE_OPEN_SET_KEYS},
                )
            )
            continue

        def mutate(config: dict[str, Any], overrides=overrides, slug=slug) -> None:
            _configure_module_pipeline_fusion(
                config,
                dataset=dataset,
                slug=slug,
                overrides=overrides,
                base_lambda=base_lambda,
            )

        output_dir = _run_pipeline_variant(
            dataset=dataset,
            category="modules",
            variant_slug=slug,
            variant_name=name,
            config_mutator=mutate,
            train=False,
            with_unknown_subdivision=False,
        )
        metrics = load_json(output_dir / "open_set_metrics.json")
        rows.append(
            ResultRow(
                category="modules",
                dataset=dataset,
                variant=name,
                variant_slug=slug,
                output_dir=str(output_dir),
                metrics={key: metrics.get(key) for key in MODULE_OPEN_SET_KEYS},
            )
        )
    return rows


def run_km_sensitivity(dataset: str) -> list[ResultRow]:
    slug = "m_0_1_2_3_auto"
    name = "m=0,1,2,3,Auto"

    def mutate(config: dict[str, Any]) -> None:
        config["unknown_subdivision"]["overcluster_extra_candidates"] = [0, 1, 2, 3]
        config["unknown_subdivision"]["m_selection_mode"] = "offline_min_gain"
        config["unknown_subdivision"]["m_selection_min_quality_gain"] = 0.01
        config["unknown_subdivision"]["output_subdir"] = "unknown_subdivision"

    output_dir = _run_pipeline_variant(
        dataset=dataset,
        category="km",
        variant_slug=slug,
        variant_name=name,
        config_mutator=mutate,
        train=False,
        with_unknown_subdivision=True,
    )
    subdivision_dir = output_dir / "unknown_subdivision"
    history = _read_json(subdivision_dir / "m_selection_history.json")
    selected = _read_json(subdivision_dir / "unknown_subdivision_metrics.json")
    rows: list[ResultRow] = []
    for item in history:
        m = int(item["overcluster_extra_clusters"])
        rows.append(
            ResultRow(
                category="km",
                dataset=dataset,
                variant=f"m={m}",
                variant_slug=f"m{m}",
                output_dir=str(output_dir),
                metrics={
                    "selected_m": m,
                    "adjusted_quality": item.get("m_selection_offline_adjusted_quality"),
                    "nmi": item.get("m_selection_offline_nmi"),
                    "ari": item.get("m_selection_offline_ari"),
                    "purity": item.get("m_selection_offline_purity"),
                    "hungarian_accuracy": item.get("m_selection_offline_hungarian_accuracy"),
                    "coverage_of_total_test_unknown": item.get("m_selection_offline_coverage"),
                    "resolved_num_clusters": item.get("resolved_num_clusters"),
                    "uncertain_ratio": item.get("uncertain_ratio"),
                },
            )
        )
    rows.append(
        ResultRow(
            category="km",
            dataset=dataset,
            variant="Auto",
            variant_slug="auto",
            output_dir=str(output_dir),
            metrics={
                "selected_m": selected.get("auto_selected_overcluster_extra_clusters"),
                "adjusted_quality": selected.get("m_selection_offline_adjusted_quality"),
                **{key: selected.get(key) for key in SUBDIVISION_KEYS},
                "resolved_num_clusters": selected.get("resolved_num_clusters"),
                "uncertain_ratio": selected.get("uncertain_ratio"),
            },
        )
    )
    return rows


def run_loss_ablations(dataset: str) -> list[ResultRow]:
    return run_selected_loss_ablations(dataset, "all")


def _selected_loss_variants(selected_variant: str) -> list[tuple[str, str, float, float]]:
    if selected_variant == "all":
        return list(LOSS_VARIANTS)
    selected = [variant for variant in LOSS_VARIANTS if variant[0] == selected_variant]
    if not selected:
        raise ValueError(f"Unknown loss ablation variant: {selected_variant}")
    return selected


def _restore_train_summary(output_dir: Path) -> None:
    summary_path = output_dir / "train_summary.json"
    checkpoint_path = output_dir / "best_closed_set.pt"
    if summary_path.exists() or not checkpoint_path.exists():
        return

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    save_json(
        summary_path,
        {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "best_val_acc": float(checkpoint.get("best_val_acc", float("nan"))),
            "resumed_from_checkpoint": True,
        },
    )


def _loss_checkpoint_matches(
    checkpoint_path: Path,
    angle_weight: float,
    prototype_weight: float,
) -> bool:
    if not checkpoint_path.exists():
        return False

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    loss_config = config.get("loss", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    expected_prototype_mode = "euclidean_mse"
    prototype_mode_matches = (
        prototype_weight == 0.0
        or str(loss_config.get("prototype_loss_mode", "")) == expected_prototype_mode
    )
    return (
        int(data_config.get("batch_size", -1)) == 64
        and str(train_config.get("device", "")).lower() == "cuda"
        and float(loss_config.get("lambda_basic", -1.0)) == 1.0
        and float(loss_config.get("lambda_angle", -1.0)) == float(angle_weight)
        and float(loss_config.get("lambda_prototype", -1.0)) == float(prototype_weight)
        and prototype_mode_matches
    )


def _run_loss_variant(
    dataset: str,
    slug: str,
    name: str,
    angle_weight: float,
    prototype_weight: float,
) -> Path:
    def mutate(config: dict[str, Any]) -> None:
        config["train"]["device"] = "cuda"
        config["data"]["batch_size"] = min(int(config["data"]["batch_size"]), 64)
        config["loss"]["lambda_basic"] = 1.0
        config["loss"]["lambda_angle"] = float(angle_weight)
        config["loss"]["lambda_prototype"] = float(prototype_weight)
        config["loss"]["prototype_loss_mode"] = "euclidean_mse"
        config["fusion"]["manual_threshold"] = None
        config["fusion"]["manual_thresholds_per_class"] = None
        config["fusion"]["score_calibration"] = "none"
        config["fusion"]["threshold_mode"] = "classwise_balanced"
        config["fusion"]["known_rescue"] = {"enabled": False}
        config["fusion"]["loss_ablation_calibration_version"] = 1
        config["train"]["loss_ablation_training_version"] = 1

    output_dir = _variant_dir("losses", dataset, slug)
    metrics_path = output_dir / "open_set_metrics.json"
    summary_path = output_dir / "train_summary.json"
    local_checkpoint = output_dir / "best_closed_set.pt"
    checkpoint_matches = _loss_checkpoint_matches(
        local_checkpoint,
        angle_weight,
        prototype_weight,
    )
    manifest_path = output_dir / "ablation_manifest.json"
    calibration_is_current = False
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        calibration_is_current = (
            manifest.get("config", {})
            .get("fusion", {})
            .get("loss_ablation_calibration_version")
            == 1
        )
    if (
        metrics_path.exists()
        and summary_path.exists()
        and calibration_is_current
        and checkpoint_matches
    ):
        return output_dir

    if checkpoint_matches:
        config = _base_config(dataset)
        _configure_output(config, output_dir, f"{dataset}_losses_{slug}")
        mutate(config)
        _restore_train_summary(output_dir)
        _write_manifest(output_dir, dataset, "losses", name, config, local_checkpoint)
        run_osr_pipeline(
            config,
            skip_prepare=True,
            skip_training=True,
            ckpt_path=local_checkpoint,
            with_unknown_subdivision=False,
        )
        return output_dir

    return _run_pipeline_variant(
        dataset=dataset,
        category="losses",
        variant_slug=slug,
        variant_name=name,
        config_mutator=mutate,
        train=True,
        with_unknown_subdivision=False,
    )


def _collect_completed_loss_rows(dataset: str) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for slug, name, angle_weight, prototype_weight in LOSS_VARIANTS:
        output_dir = _variant_dir("losses", dataset, slug)
        metrics_path = output_dir / "open_set_metrics.json"
        train_summary_path = output_dir / "train_summary.json"
        if not metrics_path.exists() or not train_summary_path.exists():
            continue
        metrics = load_json(metrics_path)
        train_summary = load_json(train_summary_path)
        rows.append(
            ResultRow(
                category="losses",
                dataset=dataset,
                variant=name,
                variant_slug=slug,
                output_dir=str(output_dir),
                metrics={
                    "classification_loss": True,
                    "angular_loss": angle_weight > 0.0,
                    "prototype_loss": prototype_weight > 0.0,
                    "best_val_acc": train_summary.get("best_val_acc"),
                    **{key: metrics.get(key) for key in CORE_OPEN_SET_KEYS},
                },
            )
        )
    return rows


def run_selected_loss_ablations(
    dataset: str,
    selected_variant: str,
) -> list[ResultRow]:
    for slug, name, angle_weight, prototype_weight in _selected_loss_variants(selected_variant):
        _run_loss_variant(
            dataset,
            slug,
            name,
            angle_weight,
            prototype_weight,
        )
        gc.collect()
    return _collect_completed_loss_rows(dataset)


def run_subdivision_ablations(dataset: str) -> list[ResultRow]:
    rows: list[ResultRow] = []
    base = _base_config(dataset)["unknown_subdivision"]
    base_quantile = float(base.get("direct_confidence_quantile", 0.0))
    base_min_cluster = int(base.get("direct_min_cluster_size", 0))
    for slug, name, feature_mode, use_filtering in SUBDIVISION_VARIANTS:
        def mutate(
            config: dict[str, Any],
            feature_mode=feature_mode,
            use_filtering=use_filtering,
        ) -> None:
            config["unknown_subdivision"]["feature_mode"] = feature_mode
            config["unknown_subdivision"]["use_known_prototype_anchors"] = False
            config["unknown_subdivision"]["direct_confidence_quantile"] = (
                base_quantile if use_filtering else 0.0
            )
            config["unknown_subdivision"]["direct_min_cluster_size"] = (
                base_min_cluster if use_filtering else 0
            )
            config["unknown_subdivision"]["output_subdir"] = "unknown_subdivision"

        output_dir = _run_pipeline_variant(
            dataset=dataset,
            category="subdivision",
            variant_slug=slug,
            variant_name=name,
            config_mutator=mutate,
            train=False,
            with_unknown_subdivision=True,
        )
        metrics = load_json(output_dir / "unknown_subdivision" / "unknown_subdivision_metrics.json")
        rows.append(
            ResultRow(
                category="subdivision",
                dataset=dataset,
                variant=name,
                variant_slug=slug,
                output_dir=str(output_dir),
                metrics={
                    "feature_mode": feature_mode,
                    "filtering": use_filtering,
                    **{key: metrics.get(key) for key in SUBDIVISION_KEYS},
                    "resolved_num_clusters": metrics.get("resolved_num_clusters"),
                    "uncertain_ratio": metrics.get("uncertain_ratio"),
                },
            )
        )
        gc.collect()
    return rows


def _write_csv(rows: list[ResultRow]) -> None:
    flat = [row.flat() for row in rows]
    fieldnames = sorted({key for row in flat for key in row})
    path = ABLATION_ROOT / "消融结果汇总.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat:
            writer.writerow(row)
    save_json(ABLATION_ROOT / "消融结果汇总.json", flat)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "✓" if value else ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _markdown_table(rows: list[ResultRow], fields: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in fields) + " |",
        "| " + " | ".join("---:" if key != "variant" else "---" for key, _ in fields) + " |",
    ]
    for row in rows:
        payload = row.flat()
        lines.append("| " + " | ".join(_format(payload.get(key)) for key, _ in fields) + " |")
    return lines


def _ablation_matrix_metric_table(
    rows: list[ResultRow],
    matrix_rows: list[tuple[str, list[bool]]],
    switch_columns: list[str],
    metric_fields: list[tuple[str, str]],
) -> list[str]:
    row_map = {row.variant_slug: row for row in rows}
    ordered_rows = []
    for slug, flags in matrix_rows:
        row = row_map.get(slug)
        if row is not None:
            ordered_rows.append((flags, row))

    labels = [*switch_columns, *(label for _, label in metric_fields)]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| "
        + " | ".join(
            ["---"] * len(switch_columns) + ["---:"] * len(metric_fields)
        )
        + " |",
    ]
    for flags, row in ordered_rows:
        payload = row.flat()
        values = [
            *(_binary_symbol(flag) for flag in flags),
            *(_format(payload.get(key)) for key, _ in metric_fields),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _binary_symbol(flag: bool) -> str:
    return "√" if flag else "X"


def _matrix_markdown(columns: list[str], rows: list[tuple[str, list[bool]]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, flags in rows:
        lines.append("| " + " | ".join(_binary_symbol(flag) for flag in flags) + " |")
    return lines


def _matrix_figure(
    rows: list[tuple[str, list[bool]]],
    columns: list[str],
    output_name: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    ax.axis("off")
    cell_text = [[_binary_symbol(flag) for flag in flags] for _, flags in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.0)
    table.scale(1.0, 1.9)

    for col_idx in range(len(columns)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#2F3B52")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    for row_idx in range(1, len(rows) + 1):
        shade = "#F5F7FA" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(columns)):
            table[(row_idx, col_idx)].set_facecolor(shade)
            table[(row_idx, col_idx)].get_text().set_fontsize(12.0)
            if table[(row_idx, col_idx)].get_text().get_text() == "√":
                table[(row_idx, col_idx)].get_text().set_fontweight("bold")

    fig.suptitle(title, fontsize=13, y=0.94)
    fig.text(
        0.5,
        0.06,
        "行顺序: " + " / ".join(name for name, _ in rows),
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.90])
    fig.savefig(ABLATION_ROOT / output_name, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _module_matrix_rows(module_rows: list[ResultRow]) -> list[tuple[str, list[bool]]]:
    order = [
        "closed_set_only",
        "openmax_only",
        "ordinary_mbs_only",
        "full_method",
    ]
    row_map = {row.variant_slug: row for row in module_rows}
    matrix = [
        ("普通 MBS（基础拒识）", [False, False, False]),
        ("OpenMax 校准", [False, False, True]),
        ("OpenMax + 原型距离校准", [False, True, True]),
        ("完整方法", [True, True, True]),
    ]
    result = []
    for slug, (label, flags) in zip(order, matrix):
        if slug in row_map:
            result.append((label, flags))
    return result


def _module_metric_matrix_rows() -> list[tuple[str, list[bool]]]:
    return [
        ("closed_set_only", [False, False, False]),
        ("openmax_only", [False, False, True]),
        ("ordinary_mbs_only", [False, True, True]),
        ("full_method", [True, True, True]),
    ]


def _loss_matrix_rows(loss_rows: list[ResultRow]) -> list[tuple[str, list[bool]]]:
    order = [
        "ce_only",
        "ce_angular",
        "ce_prototype",
        "full_embedding_learning",
    ]
    row_map = {row.variant_slug: row for row in loss_rows}
    matrix = [
        ("CE only", [True, False, False]),
        ("CE + Angular", [True, True, False]),
        ("CE + Prototype", [True, False, True]),
        ("Full embedding learning", [True, True, True]),
    ]
    result = []
    for slug, (label, flags) in zip(order, matrix):
        if slug in row_map:
            result.append((label, flags))
    return result


def _loss_metric_matrix_rows() -> list[tuple[str, list[bool]]]:
    return [
        ("ce_only", [True, False, False]),
        ("ce_angular", [True, True, False]),
        ("ce_prototype", [True, False, True]),
        ("full_embedding_learning", [True, True, True]),
    ]


def _loss_metric_fields() -> list[tuple[str, str]]:
    return [
        ("known_accuracy", "Known Acc."),
        ("unknown_recall", "Unknown Recall"),
        ("macro_f1", "Macro F1"),
        ("auroc", "AUROC"),
    ]


def _module_metric_fields() -> list[tuple[str, str]]:
    return [
        ("overall_accuracy", "Overall Acc."),
        ("known_accuracy", "Known Acc."),
        ("unknown_recall", "Unknown Recall"),
        ("unknown_precision", "Unknown Precision"),
        ("known_fpr_as_unknown", "Known FPR↓"),
        ("macro_f1", "Macro F1"),
        ("oscr", "OSCR"),
        ("auroc", "AUROC"),
    ]


def _write_markdown(rows: list[ResultRow]) -> None:
    lines = [
        "# 消融实验结果汇总",
        "",
        "四类消融均使用 Oracle 与 WiSig 两个数据集。所有汇总图位于本目录根部，",
        "各数据集的代码、配置快照与原始输出位于四个分类子目录。",
        "",
    ]
    for dataset in ["oracle", "wisig"]:
        display = DATASETS[dataset]["display"]
        lines.extend([f"## {display}", ""])

        module_rows = [row for row in rows if row.dataset == dataset and row.category == "modules"]
        lines.extend(["### 模块消融", ""])
        lines.extend(
            _ablation_matrix_metric_table(
                module_rows,
                _module_metric_matrix_rows(),
                ["原型竞争边界建模", "原型距离校准", "OpenMax 校准"],
                _module_metric_fields(),
            )
        )
        lines.extend(
            [
                "",
                "注：前三行使用同一闭集检查点和普通 MBS 伪未知样本；",
                "第一行仅使用验证已知集的全局置信度分位数做基础拒识，第二、三行依次加入 OpenMax 和原型距离校准；完整方法行使用正式 PCBM 结果。",
                "原型距离校准主要改善分数排序，因此重点观察 AUROC/OSCR；PCBM 主要减少已知类被误拒为未知类，因此重点观察 Known FPR↓、Unknown Precision 和 OSCR。",
            ]
        )
        if dataset == "oracle":
            lines.append(
                "Oracle 的原型距离校准行在验证集上从 λ={0.1,0.3,0.5,0.7,0.9} 自动选择融合权重，"
                "并设置每类已知接收率下限为 90%；测试标签不参与参数选择。"
            )
        lines.append("")

        km_rows = [row for row in rows if row.dataset == dataset and row.category == "km"]
        lines.extend(["### K+M 缓冲分量敏感性", ""])
        lines.extend(
            _markdown_table(
                km_rows,
                [
                    ("variant", "m"),
                    ("selected_m", "Selected m"),
                    ("adjusted_quality", "Adjusted Quality"),
                    ("nmi", "NMI"),
                    ("ari", "ARI"),
                    ("purity", "Purity"),
                    ("hungarian_accuracy", "Hungarian Acc."),
                    ("coverage_of_total_test_unknown", "Coverage"),
                    ("resolved_num_clusters", "Resolved K"),
                ],
            )
        )
        lines.append("")

        loss_rows = [row for row in rows if row.dataset == dataset and row.category == "losses"]
        lines.extend(["### 闭集表征学习损失消融", ""])
        lines.extend(
            _ablation_matrix_metric_table(
                loss_rows,
                _loss_metric_matrix_rows(),
                ["Classification Loss", "Angular Loss", "Prototype Loss"],
                _loss_metric_fields(),
            )
        )
        lines.append("")

        subdivision_rows = [
            row for row in rows if row.dataset == dataset and row.category == "subdivision"
        ]
        lines.extend(["### 未知类细分流程消融", ""])
        lines.extend(
            _markdown_table(
                subdivision_rows,
                [
                    ("variant", "Variant"),
                    ("nmi", "NMI"),
                    ("ari", "ARI"),
                    ("purity", "Purity"),
                    ("hungarian_accuracy", "Hungarian Acc."),
                    ("coverage_of_total_test_unknown", "Coverage"),
                    ("resolved_num_clusters", "Resolved K"),
                    ("uncertain_ratio", "Uncertain Ratio"),
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 汇总图",
            "",
            "- `KM簇数敏感性.png`",
            "- `细分流程消融.png`",
            "",
            "Auto 的规则是：在 `m=0,1,2,3` 中选择满足目标簇数、且覆盖率修正质量提升超过 1 个百分点的最小 m。",
            "若更大的 m 只带来不超过 1% 的提升，则视为冗余，不增加结构。",
            "",
        ]
    )
    (ABLATION_ROOT / "消融结果汇总.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_bar_category(
    rows: list[ResultRow],
    category: str,
    metric_keys: list[tuple[str, str]],
    output_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    for ax, dataset in zip(axes, ["oracle", "wisig"]):
        selected = [row for row in rows if row.category == category and row.dataset == dataset]
        if not selected:
            ax.set_visible(False)
            continue
        x = np.arange(len(selected))
        width = 0.18
        for idx, (key, label) in enumerate(metric_keys):
            values = [float(row.metrics.get(key) or 0.0) for row in selected]
            ax.bar(x + (idx - (len(metric_keys) - 1) / 2) * width, values, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([row.variant for row in selected], rotation=18, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(DATASETS[dataset]["display"])
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ABLATION_ROOT / output_name, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_table_category(
    rows: list[ResultRow],
    category: str,
    metric_keys: list[tuple[str, str]],
    output_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.8))
    for ax, dataset in zip(axes, ["oracle", "wisig"]):
        selected = [row for row in rows if row.category == category and row.dataset == dataset]
        ax.axis("off")
        ax.set_title(DATASETS[dataset]["display"], pad=10)
        if not selected:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        col_labels = ["Variant", *[label for _, label in metric_keys]]
        cell_text = []
        metric_values: dict[str, list[float]] = {key: [] for key, _ in metric_keys}
        for row in selected:
            payload = row.flat()
            values = [_format(payload.get(key)) for key, _ in metric_keys]
            cell_text.append([row.variant, *values])
            for key, _ in metric_keys:
                metric_values[key].append(float(payload.get(key) or 0.0))

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.0)
        table.scale(1.0, 1.55)

        for col_idx in range(len(col_labels)):
            cell = table[(0, col_idx)]
            cell.set_facecolor("#2F3B52")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

        for row_idx, row in enumerate(selected, start=1):
            if row.variant_slug == "full_method":
                for col_idx in range(len(col_labels)):
                    table[(row_idx, col_idx)].set_facecolor("#F5F7FA")
            for col_idx, (key, _) in enumerate(metric_keys, start=1):
                values = metric_values[key]
                best = max(values)
                current = float(row.flat().get(key) or 0.0)
                if abs(current - best) < 1e-12:
                    table[(row_idx, col_idx)].get_text().set_fontweight("bold")
                    table[(row_idx, col_idx)].set_facecolor("#EAF2FF")

    fig.tight_layout()
    fig.savefig(ABLATION_ROOT / output_name, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_km(rows: list[ResultRow]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, dataset in zip(axes, ["oracle", "wisig"]):
        selected = [
            row for row in rows if row.category == "km" and row.dataset == dataset and row.variant != "Auto"
        ]
        auto_rows = [
            row for row in rows if row.category == "km" and row.dataset == dataset and row.variant == "Auto"
        ]
        if not selected or not auto_rows:
            ax.set_visible(False)
            continue
        selected.sort(key=lambda row: int(row.metrics["selected_m"]))
        x = [int(row.metrics["selected_m"]) for row in selected]
        quality = [float(row.metrics.get("adjusted_quality") or 0.0) for row in selected]
        auto = auto_rows[0]
        auto_m = int(auto.metrics["selected_m"])
        auto_quality = float(auto.metrics.get("adjusted_quality") or 0.0)
        ax.plot(x, quality, marker="o", linewidth=2.2, color="#4E79A7")
        ax.scatter([auto_m], [auto_quality], s=90, color="#E15759", zorder=5, label=f"Auto = {auto_m}")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Buffer component m")
        ax.set_ylabel("Coverage-adjusted quality")
        ax.set_title(DATASETS[dataset]["display"])
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ABLATION_ROOT / "KM簇数敏感性.png", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(rows: list[ResultRow]) -> None:
    for deprecated_name in ["模块消融.png", "损失函数消融.png"]:
        deprecated_path = ABLATION_ROOT / deprecated_name
        if deprecated_path.exists():
            deprecated_path.unlink()
    _write_csv(rows)
    _write_markdown(rows)
    if any(row.category == "km" for row in rows):
        _plot_km(rows)
    if any(row.category == "subdivision" for row in rows):
        _plot_bar_category(
            rows,
            "subdivision",
            [("nmi", "NMI"), ("ari", "ARI"), ("purity", "Purity"), ("coverage_of_total_test_unknown", "Coverage")],
            "细分流程消融.png",
        )


def _write_dataset_runner(category: str, dataset: str) -> None:
    dataset_dir = ensure_dir(ABLATION_ROOT / GROUP_DIRS[category] / dataset)
    runner = dataset_dir / "run.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n\n"
        "root = Path(__file__).resolve().parents[3]\n"
        f"subprocess.run([sys.executable, str(root / 'ablations' / 'run_ablation.py'), '--category', '{category}', '--dataset', '{dataset}'], check=True, cwd=root)\n",
        encoding="utf-8",
    )


def _selected_categories(value: str) -> list[str]:
    return list(GROUP_DIRS) if value == "all" else [value]


def _selected_datasets(value: str) -> list[str]:
    return list(DATASETS) if value == "all" else [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unified Oracle/WiSig ablation suite.")
    parser.add_argument("--category", choices=["all", *GROUP_DIRS], default="all")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument(
        "--loss-variant",
        choices=["all", *(variant[0] for variant in LOSS_VARIANTS)],
        default="all",
        help="Run one loss ablation variant so long training jobs can be resumed independently.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild tables and figures from existing result files without running experiments.",
    )
    return parser.parse_args()


def collect_existing_results() -> list[ResultRow]:
    rows: list[ResultRow] = []
    summary_path = ABLATION_ROOT / "消融结果汇总.json"
    if summary_path.exists():
        for item in _read_json(summary_path):
            metrics = {
                key: value
                for key, value in item.items()
                if key not in {"category", "dataset", "variant", "variant_slug", "output_dir"}
            }
            rows.append(
                ResultRow(
                    category=item["category"],
                    dataset=item["dataset"],
                    variant=item["variant"],
                    variant_slug=item["variant_slug"],
                    output_dir=item["output_dir"],
                    metrics=metrics,
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.summary_only:
        rows = collect_existing_results()
        if not rows:
            raise FileNotFoundError("No existing ablation summary JSON was found.")
        write_summary(rows)
        print(f"Summary: {ABLATION_ROOT / '消融结果汇总.md'}")
        return

    rows = collect_existing_results()
    selected_categories = _selected_categories(args.category)
    selected_datasets = _selected_datasets(args.dataset)
    runners = {
        "modules": run_module_ablations,
        "km": run_km_sensitivity,
        "losses": run_loss_ablations,
        "subdivision": run_subdivision_ablations,
    }

    for category in selected_categories:
        for dataset in selected_datasets:
            _write_dataset_runner(category, dataset)
            if category == "losses":
                new_rows = run_selected_loss_ablations(dataset, args.loss_variant)
            else:
                new_rows = runners[category](dataset)
            rows = [
                row
                for row in rows
                if not (row.category == category and row.dataset == dataset)
            ]
            rows.extend(new_rows)
            write_summary(rows)

    print(f"Summary: {ABLATION_ROOT / '消融结果汇总.md'}")


if __name__ == "__main__":
    main()
