from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from functions.common.io import ensure_dir, load_json, load_pickle, save_json
from functions.data.data_build import build_data_module
from functions.data.npz_dataset import SignalDataset
from functions.methods.fusion import (
    apply_score_calibration,
    fit_score_calibration,
    fuse_unknown_score,
    prototype_distance_unknown_score,
)
from functions.methods.leave_class_out import (
    LeaveClassOutFold,
    aggregate_transfer_quantiles,
    balanced_pseudo_indices,
    build_leave_class_out_folds,
    fold_class_coverage,
    evaluate_leave_class_out_candidate,
    restore_class_thresholds,
    search_leave_class_out_candidates,
    subset_and_remap,
)
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes
from functions.model.closed_set import ClosedSetTrainer
from functions.pipeline import (
    evaluate_open_set_artifacts,
    fit_openmax_artifacts,
    generate_pseudo_unknown_artifacts,
    mine_boundary_artifacts,
    train_closed_set,
)
from run_oracle import build_config


ROOT = Path(__file__).resolve().parent
METRIC_KEYS = ("known_accuracy", "unknown_recall", "macro_f1", "auroc")


def materialize_fold_data(
    source_root: str | Path,
    target_root: str | Path,
    fold: LeaveClassOutFold,
) -> dict[str, Any]:
    source_root = Path(source_root)
    target_root = ensure_dir(target_root)
    train = np.load(source_root / "train_known.npz")
    val = np.load(source_root / "val_known.npz")

    train_x, train_y, train_global_y = subset_and_remap(
        train["x"], train["y"], fold.known_classes
    )
    val_x, val_y, val_global_y = subset_and_remap(
        val["x"], val["y"], fold.known_classes
    )
    heldout_mask = np.isin(val["y"], fold.held_out_classes)
    heldout_x = np.asarray(val["x"][heldout_mask])
    heldout_global_y = np.asarray(val["y"][heldout_mask], dtype=np.int64)
    local_unknown_label = len(fold.known_classes)
    heldout_y = np.full(len(heldout_x), local_unknown_label, dtype=np.int64)

    np.savez_compressed(
        target_root / "train_known.npz",
        x=train_x,
        y=train_y,
        global_y=train_global_y,
    )
    np.savez_compressed(
        target_root / "val_known.npz",
        x=val_x,
        y=val_y,
        global_y=val_global_y,
    )
    np.savez_compressed(
        target_root / "test_known.npz",
        x=val_x,
        y=val_y,
        global_y=val_global_y,
    )
    np.savez_compressed(
        target_root / "test_unknown.npz",
        x=heldout_x,
        y=heldout_y,
        global_y=heldout_global_y,
    )
    manifest = {
        "fold_index": fold.fold_index,
        "known_classes": list(fold.known_classes),
        "held_out_classes": list(fold.held_out_classes),
        "local_to_global": {
            str(local): int(global_class)
            for local, global_class in enumerate(fold.known_classes)
        },
        "num_train_known": int(len(train_y)),
        "num_val_known": int(len(val_y)),
        "num_simulated_unknown": int(len(heldout_y)),
        "uses_real_unknown_for_calibration": False,
        "calibration_sources": [
            "train_known retained classes",
            "val_known retained classes",
            "val_known held-out classes",
            "feature-level pseudo unknown",
        ],
    }
    save_json(target_root / "fold_data_manifest.json", manifest)
    return manifest


def build_metric_comparison(
    *,
    old_manual: dict[str, Any],
    current_auto: dict[str, Any],
    leave_class_out: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    def direction(value: float) -> str:
        if value > 1e-12:
            return "提高"
        if value < -1e-12:
            return "降低"
        return "不变"

    comparison = {}
    for key in METRIC_KEYS:
        old_value = float(old_manual[key])
        auto_value = float(current_auto[key])
        new_value = float(leave_class_out[key])
        vs_old = round((new_value - old_value) * 100.0, 6)
        vs_auto = round((new_value - auto_value) * 100.0, 6)
        comparison[key] = {
            "old_manual": old_value,
            "current_auto": auto_value,
            "leave_class_out": new_value,
            "vs_old_manual_pp": vs_old,
            "vs_old_manual_direction": direction(vs_old),
            "vs_current_auto_pp": vs_auto,
            "vs_current_auto_direction": direction(vs_auto),
        }
    return comparison


def _fold_config(
    base_config: dict[str, Any],
    fold: LeaveClassOutFold,
    fold_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["project"]["name"] = f"oracle_lco_fold_{fold.fold_index}"
    config["project"]["output_dir"] = str((fold_dir / "artifacts").resolve())
    config["data"]["root"] = str((fold_dir / "data").resolve())
    lco = base_config["fusion"]["leave_class_out"]
    config["train"]["seed"] = int(lco["base_seed"]) + fold.fold_index
    config["unknown_subdivision"]["enabled"] = False
    config["reporting"]["write_root_summaries"] = False
    return config


def _unknown_scores(
    *,
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    openmax: OpenMaxCalibrator,
    mu: np.ndarray,
    sigma: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, _, distances = predict_with_prototypes(embeddings, prototypes, temperature)
    q_om = openmax.predict(activations_from_distances(distances))["unknown_prob"]
    q_pd = prototype_distance_unknown_score(distances, pred, mu, sigma)
    return pred, q_om, q_pd


def calibrate_fold(
    config: dict[str, Any],
    fold: LeaveClassOutFold,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(config["project"]["output_dir"])
    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config,
        datamodule.bundle.num_known_classes,
        datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(output_dir / "best_closed_set.pt")
    val_payload = trainer.extract_embeddings(datamodule.val_known_dataloader())

    heldout_file = np.load(Path(config["data"]["root"]) / "test_unknown.npz")
    heldout_loader = DataLoader(
        SignalDataset(
            heldout_file["x"],
            heldout_file["y"],
            normalize=str(config["data"].get("normalize", "none")),
        ),
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    heldout_payload = trainer.extract_embeddings(heldout_loader)
    pseudo_file = np.load(output_dir / "pseudo_unknown.npz", allow_pickle=True)
    stats = np.load(output_dir / "distance_stats.npz")
    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    prototypes = val_payload["prototypes"]
    temperature = float(config["model"]["temperature"])

    val_pred, val_q_om, val_q_pd = _unknown_scores(
        embeddings=val_payload["embeddings"],
        prototypes=prototypes,
        openmax=openmax,
        mu=stats["mu"],
        sigma=stats["sigma"],
        temperature=temperature,
    )
    heldout_pred, heldout_q_om, heldout_q_pd = _unknown_scores(
        embeddings=heldout_payload["embeddings"],
        prototypes=prototypes,
        openmax=openmax,
        mu=stats["mu"],
        sigma=stats["sigma"],
        temperature=temperature,
    )
    lco = config["fusion"]["leave_class_out"]
    pseudo_indices = balanced_pseudo_indices(
        pseudo_file["source_classes"],
        pseudo_file["pseudo_kind"],
        max_samples=int(lco["pseudo_max_samples"]),
        seed=int(config["train"]["seed"]),
    )
    pseudo_embeddings = pseudo_file["pseudo_embeddings"][pseudo_indices]
    pseudo_pred, pseudo_q_om, pseudo_q_pd = _unknown_scores(
        embeddings=pseudo_embeddings,
        prototypes=prototypes,
        openmax=openmax,
        mu=stats["mu"],
        sigma=stats["sigma"],
        temperature=temperature,
    )
    candidates = search_leave_class_out_candidates(
        known_labels=val_payload["labels"],
        known_pred=val_pred,
        known_q_om=val_q_om,
        known_q_pd=val_q_pd,
        heldout_pred=heldout_pred,
        heldout_q_om=heldout_q_om,
        heldout_q_pd=heldout_q_pd,
        pseudo_pred=pseudo_pred,
        pseudo_q_om=pseudo_q_om,
        pseudo_q_pd=pseudo_q_pd,
        num_classes=len(fold.known_classes),
        lambda_grid=config["fusion"]["lambda_grid"],
        threshold_grid=config["fusion"]["threshold_grid"],
        known_penalty_grid=config["fusion"]["classwise_known_penalty_grid"],
        min_known_accuracy=float(lco["min_known_accuracy"]),
        selection_weights=lco["selection_weights"],
        score_calibration_mode=str(config["fusion"]["score_calibration"]),
        fusion_mode=str(config["fusion"].get("mode", "linear")),
    )
    np.savez_compressed(
        output_dir / "fold_scores.npz",
        known_labels=val_payload["labels"],
        known_pred=val_pred,
        known_q_om=val_q_om,
        known_q_pd=val_q_pd,
        heldout_pred=heldout_pred,
        heldout_q_om=heldout_q_om,
        heldout_q_pd=heldout_q_pd,
        pseudo_pred=pseudo_pred,
        pseudo_q_om=pseudo_q_om,
        pseudo_q_pd=pseudo_q_pd,
    )
    summary = {
        "fold_index": fold.fold_index,
        "known_classes": list(fold.known_classes),
        "held_out_classes": list(fold.held_out_classes),
        "local_to_global": manifest["local_to_global"],
        "num_pseudo_available": int(len(pseudo_file["pseudo_embeddings"])),
        "num_pseudo_selected": int(len(pseudo_indices)),
        "uses_real_unknown_for_calibration": False,
        "candidates": candidates,
    }
    save_json(output_dir / "fold_calibration.json", summary)
    return summary


def run_fold(
    base_config: dict[str, Any],
    fold: LeaveClassOutFold,
    experiment_root: Path,
) -> dict[str, Any]:
    fold_dir = ensure_dir(experiment_root / "folds" / f"fold_{fold.fold_index}")
    manifest = materialize_fold_data(
        base_config["data"]["root"],
        fold_dir / "data",
        fold,
    )
    config = _fold_config(base_config, fold, fold_dir)
    output_dir = Path(config["project"]["output_dir"])
    checkpoint = output_dir / "best_closed_set.pt"
    if not checkpoint.exists():
        print(
            f"\n[LCO] 训练 fold {fold.fold_index + 1}/5，"
            f"留出类={fold.held_out_classes}，训练类={fold.known_classes}"
        )
        train_closed_set(config)
    else:
        print(f"\n[LCO] fold {fold.fold_index + 1}/5 已有模型，复用 {checkpoint}")
    mine_boundary_artifacts(config, ckpt_path=checkpoint)
    generate_pseudo_unknown_artifacts(config)
    fit_openmax_artifacts(config, ckpt_path=checkpoint)
    return calibrate_fold(config, fold, manifest)


def _candidate_for_lambda(
    candidates: list[dict[str, Any]],
    fusion_lambda: float,
) -> dict[str, Any]:
    return next(
        candidate
        for candidate in candidates
        if float(candidate["fusion_lambda"]) == float(fusion_lambda)
    )


def _quantile_rows_for_lambda(
    folds: list[LeaveClassOutFold],
    fold_summaries: list[dict[str, Any]],
    fusion_lambda: float,
) -> list[dict[str, Any]]:
    rows = []
    for fold, summary in zip(folds, fold_summaries):
        candidate = _candidate_for_lambda(summary["candidates"], fusion_lambda)
        for local_class, global_class in enumerate(fold.known_classes):
            rows.append(
                {
                    "fold_index": fold.fold_index,
                    "local_class": local_class,
                    "global_class": global_class,
                    "quantile": float(
                        candidate["threshold_quantiles_per_class"][local_class]
                    ),
                }
            )
    return rows


def _mean_metrics(metrics_per_fold: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics_per_fold[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metrics_per_fold]))
        for key in keys
    }


def _nested_transfer_selection(
    base_config: dict[str, Any],
    folds: list[LeaveClassOutFold],
    fold_summaries: list[dict[str, Any]],
    experiment_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    lco = base_config["fusion"]["leave_class_out"]
    modes = (
        "class_median",
        "class_lower_quartile",
        "global_median",
        "global_lower_quartile",
    )
    strategy_summaries = []
    for fusion_lambda in base_config["fusion"]["lambda_grid"]:
        rows = _quantile_rows_for_lambda(folds, fold_summaries, fusion_lambda)
        for mode in modes:
            fold_metrics = []
            fold_quantiles = []
            for fold in folds:
                score_file = np.load(
                    experiment_root
                    / "folds"
                    / f"fold_{fold.fold_index}"
                    / "artifacts"
                    / "fold_scores.npz"
                )
                known_raw = fuse_unknown_score(
                    score_file["known_q_om"],
                    score_file["known_q_pd"],
                    fusion_lambda,
                    mode=str(base_config["fusion"].get("mode", "linear")),
                )
                heldout_raw = fuse_unknown_score(
                    score_file["heldout_q_om"],
                    score_file["heldout_q_pd"],
                    fusion_lambda,
                    mode=str(base_config["fusion"].get("mode", "linear")),
                )
                pseudo_raw = fuse_unknown_score(
                    score_file["pseudo_q_om"],
                    score_file["pseudo_q_pd"],
                    fusion_lambda,
                    mode=str(base_config["fusion"].get("mode", "linear")),
                )
                calibration = fit_score_calibration(
                    known_raw,
                    score_file["known_pred"],
                    len(fold.known_classes),
                    str(base_config["fusion"]["score_calibration"]),
                )
                known_scores = apply_score_calibration(
                    known_raw, score_file["known_pred"], calibration
                )
                heldout_scores = apply_score_calibration(
                    heldout_raw, score_file["heldout_pred"], calibration
                )
                pseudo_scores = apply_score_calibration(
                    pseudo_raw, score_file["pseudo_pred"], calibration
                )
                quantiles = aggregate_transfer_quantiles(
                    rows=rows,
                    target_global_classes=fold.known_classes,
                    excluded_fold_index=fold.fold_index,
                    mode=mode,
                )
                thresholds = restore_class_thresholds(
                    known_scores=known_scores,
                    known_pred=score_file["known_pred"],
                    class_quantiles=quantiles,
                    num_classes=len(fold.known_classes),
                )
                metrics = evaluate_leave_class_out_candidate(
                    known_labels=score_file["known_labels"],
                    known_pred=score_file["known_pred"],
                    heldout_pred=score_file["heldout_pred"],
                    pseudo_pred=score_file["pseudo_pred"],
                    known_scores=known_scores,
                    heldout_scores=heldout_scores,
                    pseudo_scores=pseudo_scores,
                    thresholds=np.asarray(thresholds),
                    unknown_label=len(fold.known_classes),
                    selection_weights=lco["selection_weights"],
                )
                fold_metrics.append(metrics)
                fold_quantiles.append(quantiles)
            mean_metrics = _mean_metrics(fold_metrics)
            strategy_summaries.append(
                {
                    "fusion_lambda": float(fusion_lambda),
                    "aggregation_mode": mode,
                    "metrics": mean_metrics,
                    "fold_metrics": fold_metrics,
                    "fold_quantiles": fold_quantiles,
                    "all_folds_feasible": all(
                        metrics["known_accuracy"] >= float(lco["min_known_accuracy"])
                        for metrics in fold_metrics
                    ),
                    "feasible_fold_count": int(
                        sum(
                            metrics["known_accuracy"] >= float(lco["min_known_accuracy"])
                            for metrics in fold_metrics
                        )
                    ),
                }
            )
    feasible = [
        summary for summary in strategy_summaries
        if summary["all_folds_feasible"]
    ]
    pool = feasible or strategy_summaries
    selected = max(
        pool,
        key=lambda item: (
            item["feasible_fold_count"],
            item["metrics"]["selection_score"],
            item["metrics"]["known_accuracy"],
            -item["fusion_lambda"],
        ),
    )
    selected_rows = _quantile_rows_for_lambda(
        folds,
        fold_summaries,
        selected["fusion_lambda"],
    )
    class_quantiles = aggregate_transfer_quantiles(
        rows=selected_rows,
        target_global_classes=range(10),
        excluded_fold_index=None,
        mode=selected["aggregation_mode"],
    )
    save_json(
        experiment_root / "nested_transfer_search.json",
        {
            "selected": selected,
            "strategies": strategy_summaries,
            "uses_real_unknown_for_calibration": False,
        },
    )
    return selected, selected_rows, class_quantiles


def _build_formal_fusion(
    base_config: dict[str, Any],
    selected: dict[str, Any],
    class_quantiles: list[float],
) -> dict[str, Any]:
    source_output = Path(base_config["project"]["output_dir"])
    datamodule = build_data_module(base_config)
    trainer = ClosedSetTrainer(
        base_config,
        datamodule.bundle.num_known_classes,
        datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(source_output / "best_closed_set.pt")
    payload = trainer.extract_embeddings(datamodule.val_known_dataloader())
    stats = np.load(source_output / "distance_stats.npz")
    openmax = OpenMaxCalibrator.from_state_dict(
        load_pickle(source_output / "openmax.pkl")
    )
    pred, q_om, q_pd = _unknown_scores(
        embeddings=payload["embeddings"],
        prototypes=payload["prototypes"],
        openmax=openmax,
        mu=stats["mu"],
        sigma=stats["sigma"],
        temperature=float(base_config["model"]["temperature"]),
    )
    fusion_lambda = float(selected["fusion_lambda"])
    raw_scores = fuse_unknown_score(
        q_om,
        q_pd,
        fusion_lambda,
        mode=str(base_config["fusion"].get("mode", "linear")),
    )
    score_calibration = fit_score_calibration(
        raw_scores,
        pred,
        datamodule.bundle.num_known_classes,
        str(base_config["fusion"]["score_calibration"]),
    )
    calibrated_scores = apply_score_calibration(raw_scores, pred, score_calibration)
    thresholds = restore_class_thresholds(
        known_scores=calibrated_scores,
        known_pred=pred,
        class_quantiles=class_quantiles,
        num_classes=datamodule.bundle.num_known_classes,
    )
    return {
        "fusion_lambda": fusion_lambda,
        "threshold": None,
        "thresholds_per_class": thresholds,
        "threshold_mode": "leave_class_out_quantile_transfer",
        "threshold_quantile": None,
        "threshold_quantiles_per_class": class_quantiles,
        "fusion_mode": str(base_config["fusion"].get("mode", "linear")),
        "score_calibration": score_calibration or {"mode": "none"},
        "known_rescue": {"enabled": False},
        "metrics": selected["metrics"],
        "calibration_provenance": {
            "num_folds": 5,
            "aggregation_mode": selected["aggregation_mode"],
            "uses_real_unknown_for_calibration": False,
            "simulated_unknown_source": "held-out classes from val_known",
            "pseudo_unknown_source": "feature-level boundary extrapolation",
        },
    }


def _copy_formal_artifacts(source_output: Path, final_output: Path) -> None:
    ensure_dir(final_output)
    for filename in ("best_closed_set.pt", "openmax.pkl", "distance_stats.npz"):
        shutil.copy2(source_output / filename, final_output / filename)


def _write_comparison_markdown(
    path: Path,
    comparison: dict[str, dict[str, Any]],
) -> None:
    labels = {
        "known_accuracy": "Known Acc.",
        "unknown_recall": "Unknown Recall",
        "macro_f1": "Macro F1",
        "auroc": "AUROC",
    }
    lines = [
        "# Oracle 五折留类校准结果对比",
        "",
        "真实六个未知类未参与模型训练、OpenMax拟合、融合权重搜索或阈值搜索，只用于最终测试。",
        "",
        "| 指标 | 旧手动版 | 当前伪未知自动版 | 五折留类新版 | 新版-旧版(百分点) | 新版-当前版(百分点) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        row = comparison[key]
        lines.append(
            f"| {labels[key]} | {row['old_manual']:.4%} | "
            f"{row['current_auto']:.4%} | {row['leave_class_out']:.4%} | "
            f"{row['vs_old_manual_pp']:+.4f}（{row['vs_old_manual_direction']}） | "
            f"{row['vs_current_auto_pp']:+.4f}（{row['vs_current_auto_direction']}） |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_config = build_config()
    lco = base_config["fusion"]["leave_class_out"]
    if not bool(lco["enabled"]):
        raise RuntimeError("Oracle leave-class-out calibration is disabled.")
    if bool(lco.get("uses_real_unknown_for_calibration", True)):
        raise RuntimeError("Real unknown data must not be used for calibration.")

    experiment_root = ensure_dir(lco["output_dir"])
    source_output = Path(base_config["project"]["output_dir"])
    current_auto = load_json(source_output / "open_set_metrics.json")
    baseline_file = ROOT / "outputs" / "auto_fusion_calibration_comparison" / "baseline_metrics.json"
    old_manual = load_json(baseline_file)["oracle"]
    save_json(
        experiment_root / "baseline_snapshot.json",
        {
            "old_manual": old_manual,
            "current_auto": current_auto,
            "old_manual_source": str(baseline_file),
            "current_auto_source": str(source_output / "open_set_metrics.json"),
        },
    )

    folds = build_leave_class_out_folds(
        num_classes=10,
        num_folds=int(lco["num_folds"]),
    )
    coverage = fold_class_coverage(folds, num_classes=10)
    expected_heldout = {str(cls): 1 for cls in range(10)}
    expected_known = {str(cls): 4 for cls in range(10)}
    if coverage["held_out_counts"] != expected_heldout:
        raise RuntimeError(f"Invalid held-out coverage: {coverage['held_out_counts']}")
    if coverage["known_counts"] != expected_known:
        raise RuntimeError(f"Invalid known coverage: {coverage['known_counts']}")

    fold_summaries = [
        run_fold(base_config, fold, experiment_root)
        for fold in folds
    ]
    selected, quantile_rows, class_quantiles = _nested_transfer_selection(
        base_config,
        folds,
        fold_summaries,
        experiment_root,
    )
    aggregate_summary = {
        "selected_lambda": selected["fusion_lambda"],
        "aggregation_mode": selected["aggregation_mode"],
        "mean_fold_metrics": selected["metrics"],
        "all_folds_feasible": selected["all_folds_feasible"],
        "feasible_fold_count": selected["feasible_fold_count"],
        "class_quantiles": class_quantiles,
        "quantile_rows": quantile_rows,
        "coverage": coverage,
        "uses_real_unknown_for_calibration": False,
    }
    save_json(experiment_root / "aggregate_calibration.json", aggregate_summary)

    final_output = experiment_root / "final"
    _copy_formal_artifacts(source_output, final_output)
    formal_fusion = _build_formal_fusion(base_config, selected, class_quantiles)
    save_json(final_output / "fusion.json", formal_fusion)
    final_config = copy.deepcopy(base_config)
    final_config["project"]["name"] = "oracle_leave_class_out_calibration"
    final_config["project"]["output_dir"] = str(final_output.resolve())
    final_config["unknown_subdivision"]["enabled"] = False
    final_config["reporting"]["write_root_summaries"] = False
    final_metrics = evaluate_open_set_artifacts(
        final_config,
        ckpt_path=final_output / "best_closed_set.pt",
    )

    comparison = build_metric_comparison(
        old_manual=old_manual,
        current_auto=current_auto,
        leave_class_out=final_metrics,
    )
    save_json(
        experiment_root / "comparison.json",
        {
            "metrics": comparison,
            "old_manual": old_manual,
            "current_auto": current_auto,
            "leave_class_out": final_metrics,
            "uses_real_unknown_for_calibration": False,
        },
    )
    _write_comparison_markdown(experiment_root / "comparison.md", comparison)

    print("\nOracle 五折留类校准完成")
    print(f"selected_lambda: {selected['fusion_lambda']:.6f}")
    print(f"aggregation_mode: {selected['aggregation_mode']}")
    for key in METRIC_KEYS:
        row = comparison[key]
        print(
            f"{key}: {row['leave_class_out']:.6f} | "
            f"vs old {row['vs_old_manual_pp']:+.4f} pp | "
            f"vs current {row['vs_current_auto_pp']:+.4f} pp"
        )


if __name__ == "__main__":
    main()
