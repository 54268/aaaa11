from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from functions.common.io import ensure_dir, load_json, load_pickle, save_json
from functions.data.data_build import build_data_module
from functions.methods.leave_class_out import balanced_pseudo_indices, build_leave_class_out_folds
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes
from functions.methods.fusion import prototype_distance_unknown_score
from functions.methods.supervised_calibrator import (
    build_calibrator_features,
    choose_classwise_thresholds_with_pseudo_guard,
    evaluate_calibrator_candidate,
    save_calibrator,
    thresholds_from_quantile,
    train_calibrator,
)
from functions.model.closed_set import ClosedSetTrainer
from functions.pipeline import evaluate_open_set_artifacts
from run_oracle import build_config
from run_oracle_leave_class_out import (
    _copy_formal_artifacts,
    build_metric_comparison,
)


ROOT = Path(__file__).resolve().parent
METRIC_KEYS = ("known_accuracy", "unknown_recall", "macro_f1", "auroc")


def rescale_pseudo_embeddings(
    source_embeddings: np.ndarray,
    pseudo_embeddings: np.ndarray,
    scale: float,
) -> np.ndarray:
    source_embeddings = np.asarray(source_embeddings, dtype=np.float32)
    pseudo_embeddings = np.asarray(pseudo_embeddings, dtype=np.float32)
    if source_embeddings.shape != pseudo_embeddings.shape:
        raise ValueError("source and pseudo embeddings must have equal shapes.")
    return source_embeddings + float(scale) * (pseudo_embeddings - source_embeddings)


def select_cross_fold_candidate(
    candidates: list[dict[str, Any]],
    min_known_accuracy: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty.")
    enriched = []
    for candidate in candidates:
        fold_metrics = candidate["fold_metrics"]
        mean_known = float(np.mean([m["known_accuracy"] for m in fold_metrics]))
        mean_score = float(np.mean([m["selection_score"] for m in fold_metrics]))
        worst_known = float(min(m["known_accuracy"] for m in fold_metrics))
        enriched.append(
            {
                **candidate,
                "mean_known_accuracy": mean_known,
                "mean_selection_score": mean_score,
                "worst_fold_known_accuracy": worst_known,
                "mean_metrics": {
                    key: float(np.mean([m[key] for m in fold_metrics]))
                    for key in fold_metrics[0]
                    if isinstance(fold_metrics[0][key], (int, float, bool, np.bool_))
                },
            }
        )
    feasible = [
        item
        for item in enriched
        if item["mean_known_accuracy"] >= float(min_known_accuracy)
    ]
    pool = feasible or enriched
    return max(
        pool,
        key=lambda item: (
            item["mean_selection_score"],
            item["worst_fold_known_accuracy"],
            -int(item["seed"]),
        ),
    )


def _score_embeddings(
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


def _load_fold_payload(
    base_config: dict[str, Any],
    fold_index: int,
    pseudo_scale: float,
    pseudo_max_samples: int,
) -> dict[str, np.ndarray]:
    fold_root = (
        Path(base_config["fusion"]["leave_class_out"]["output_dir"])
        / "folds"
        / f"fold_{fold_index}"
    )
    artifact_root = fold_root / "artifacts"
    score_file = np.load(artifact_root / "fold_scores.npz")
    boundary = np.load(artifact_root / "boundary_mining.npz", allow_pickle=True)
    pseudo = np.load(artifact_root / "pseudo_unknown.npz", allow_pickle=True)
    fold_manifest = load_json(artifact_root / "fold_calibration.json")
    num_classes = len(fold_manifest["known_classes"])

    fold_config = copy.deepcopy(base_config)
    fold_config["project"]["output_dir"] = str(artifact_root)
    fold_config["data"]["root"] = str(fold_root / "data")
    fold_config["train"]["seed"] = (
        int(base_config["fusion"]["leave_class_out"]["base_seed"]) + fold_index
    )
    datamodule = build_data_module(fold_config)
    trainer = ClosedSetTrainer(
        fold_config,
        num_classes,
        datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(artifact_root / "best_closed_set.pt")
    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(artifact_root / "openmax.pkl"))
    stats = np.load(artifact_root / "distance_stats.npz")

    indices = balanced_pseudo_indices(
        pseudo["source_classes"],
        pseudo["pseudo_kind"],
        max_samples=pseudo_max_samples,
        seed=fold_config["train"]["seed"],
    )
    pseudo_embeddings = pseudo["pseudo_embeddings"][indices]
    source_embeddings = boundary["embeddings"][pseudo["source_indices"][indices]]
    scaled = rescale_pseudo_embeddings(source_embeddings, pseudo_embeddings, pseudo_scale)
    pseudo_pred, pseudo_q_om, pseudo_q_pd = _score_embeddings(
        scaled,
        stats["prototypes"],
        openmax,
        stats["mu"],
        stats["sigma"],
        float(fold_config["model"]["temperature"]),
    )
    return {
        "known_labels": score_file["known_labels"],
        "known_pred": score_file["known_pred"],
        "known_q_om": score_file["known_q_om"],
        "known_q_pd": score_file["known_q_pd"],
        "heldout_pred": score_file["heldout_pred"],
        "heldout_q_om": score_file["heldout_q_om"],
        "heldout_q_pd": score_file["heldout_q_pd"],
        "pseudo_pred": pseudo_pred,
        "pseudo_q_om": pseudo_q_om,
        "pseudo_q_pd": pseudo_q_pd,
        "num_classes": np.asarray(num_classes),
    }


def _search_five_folds(
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    calibrator_cfg = config["fusion"]["supervised_calibrator"]
    fold_cache: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    candidates = []
    for pseudo_scale in calibrator_cfg["pseudo_scale_grid"]:
        for fold_index in range(5):
            fold_cache[(fold_index, float(pseudo_scale))] = _load_fold_payload(
                config,
                fold_index,
                float(pseudo_scale),
                int(calibrator_cfg["pseudo_max_samples"]),
            )
        for fusion_lambda in calibrator_cfg["lambda_grid"]:
            for epochs in calibrator_cfg["epoch_grid"]:
                for seed in calibrator_cfg["seed_grid"]:
                    trained = []
                    for fold_index in range(5):
                        payload = fold_cache[(fold_index, float(pseudo_scale))]
                        known_features = build_calibrator_features(
                            payload["known_q_om"],
                            payload["known_q_pd"],
                            fusion_lambda,
                        )
                        pseudo_features = build_calibrator_features(
                            payload["pseudo_q_om"],
                            payload["pseudo_q_pd"],
                            fusion_lambda,
                        )
                        result = train_calibrator(
                            known_features,
                            pseudo_features,
                            seed=int(seed),
                            epochs=int(epochs),
                            lr=float(calibrator_cfg["learning_rate"]),
                            hidden_dim=int(calibrator_cfg["hidden_dim"]),
                        )
                        known_scores = result.model.predict_proba(known_features)
                        heldout_features = build_calibrator_features(
                            payload["heldout_q_om"],
                            payload["heldout_q_pd"],
                            fusion_lambda,
                        )
                        heldout_scores = result.model.predict_proba(heldout_features)
                        trained.append((payload, known_scores, heldout_scores))
                    for quantile in calibrator_cfg["threshold_quantile_grid"]:
                        fold_metrics = []
                        for payload, known_scores, heldout_scores in trained:
                            thresholds = thresholds_from_quantile(
                                known_scores,
                                payload["known_pred"],
                                int(payload["num_classes"]),
                                float(quantile),
                            )
                            fold_metrics.append(
                                evaluate_calibrator_candidate(
                                    known_labels=payload["known_labels"],
                                    known_pred=payload["known_pred"],
                                    known_scores=known_scores,
                                    heldout_pred=payload["heldout_pred"],
                                    heldout_scores=heldout_scores,
                                    thresholds=thresholds,
                                    unknown_label=int(payload["num_classes"]),
                                    min_known_accuracy=float(
                                        calibrator_cfg["min_known_accuracy"]
                                    ),
                                )
                            )
                        candidates.append(
                            {
                                "key": (
                                    f"s{pseudo_scale}_l{fusion_lambda}_"
                                    f"e{epochs}_r{seed}_q{quantile}"
                                ),
                                "pseudo_scale": float(pseudo_scale),
                                "fusion_lambda": float(fusion_lambda),
                                "epochs": int(epochs),
                                "seed": int(seed),
                                "threshold_quantile": float(quantile),
                                "fold_metrics": fold_metrics,
                                "uses_real_unknown_for_selection": False,
                            }
                        )
    selected = select_cross_fold_candidate(
        candidates,
        min_known_accuracy=float(calibrator_cfg["min_known_accuracy"]),
    )
    save_json(
        output_root / "five_fold_selection.json",
        {
            "selected": selected,
            "candidates": candidates,
            "uses_real_unknown_for_selection": False,
        },
    )
    return selected


def _formal_training_payload(
    config: dict[str, Any],
    selected: dict[str, Any],
) -> tuple[dict[str, np.ndarray], ClosedSetTrainer]:
    source_output = Path(config["project"]["output_dir"])
    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config,
        datamodule.bundle.num_known_classes,
        datamodule.bundle.signal_length,
    )
    trainer.load_checkpoint(source_output / "best_closed_set.pt")
    val = trainer.extract_embeddings(datamodule.val_known_dataloader())
    stats = np.load(source_output / "distance_stats.npz")
    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(source_output / "openmax.pkl"))
    known_pred, known_q_om, known_q_pd = _score_embeddings(
        val["embeddings"],
        stats["prototypes"],
        openmax,
        stats["mu"],
        stats["sigma"],
        float(config["model"]["temperature"]),
    )
    boundary = np.load(source_output / "boundary_mining.npz", allow_pickle=True)
    pseudo = np.load(source_output / "pseudo_unknown.npz", allow_pickle=True)
    calibrator_cfg = config["fusion"]["supervised_calibrator"]
    indices = balanced_pseudo_indices(
        pseudo["source_classes"],
        pseudo["pseudo_kind"],
        int(calibrator_cfg["pseudo_max_samples"]),
        int(selected["seed"]),
    )
    sources = boundary["embeddings"][pseudo["source_indices"][indices]]
    scaled = rescale_pseudo_embeddings(
        sources,
        pseudo["pseudo_embeddings"][indices],
        float(selected["pseudo_scale"]),
    )
    pseudo_pred, pseudo_q_om, pseudo_q_pd = _score_embeddings(
        scaled,
        stats["prototypes"],
        openmax,
        stats["mu"],
        stats["sigma"],
        float(config["model"]["temperature"]),
    )
    return {
        "known_labels": val["labels"],
        "known_pred": known_pred,
        "known_q_om": known_q_om,
        "known_q_pd": known_q_pd,
        "pseudo_pred": pseudo_pred,
        "pseudo_q_om": pseudo_q_om,
        "pseudo_q_pd": pseudo_q_pd,
    }, trainer


def _write_comparison(
    path: Path,
    rows: dict[str, dict[str, float]],
) -> None:
    labels = {
        "known_accuracy": "Known Acc.",
        "unknown_recall": "Unknown Recall",
        "macro_f1": "Macro F1",
        "auroc": "AUROC",
    }
    lines = [
        "# Oracle 伪未知监督校准器结果",
        "",
        "| 指标 | 旧手动版 | 伪未知阈值搜索版 | 五折阈值迁移版 | 可学习校准器版 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        row = rows[key]
        lines.append(
            f"| {labels[key]} | {row['old_manual']:.4%} | "
            f"{row['current_auto']:.4%} | {row['leave_class_out']:.4%} | "
            f"{row['supervised_calibrator']:.4%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    config = build_config()
    calibrator_cfg = config["fusion"]["supervised_calibrator"]
    if bool(calibrator_cfg.get("uses_real_unknown_for_selection", True)):
        raise RuntimeError("Real unknowns must not be used for calibrator selection.")
    output_root = ensure_dir(calibrator_cfg["output_dir"])
    selection_path = output_root / "five_fold_selection.json"
    if selection_path.exists():
        selected = load_json(selection_path)["selected"]
        print(f"[REUSE] five-fold selection: {selected['key']}")
    else:
        selected = _search_five_folds(config, output_root)

    payload, _ = _formal_training_payload(config, selected)
    known_features = build_calibrator_features(
        payload["known_q_om"],
        payload["known_q_pd"],
        selected["fusion_lambda"],
    )
    pseudo_features = build_calibrator_features(
        payload["pseudo_q_om"],
        payload["pseudo_q_pd"],
        selected["fusion_lambda"],
    )
    result = train_calibrator(
        known_features,
        pseudo_features,
        seed=int(selected["seed"]),
        epochs=int(selected["epochs"]),
        lr=float(calibrator_cfg["learning_rate"]),
        hidden_dim=int(calibrator_cfg["hidden_dim"]),
    )
    known_scores = result.model.predict_proba(known_features)
    pseudo_scores = result.model.predict_proba(pseudo_features)
    threshold_safety = choose_classwise_thresholds_with_pseudo_guard(
        known_labels=payload["known_labels"],
        known_pred=payload["known_pred"],
        known_scores=known_scores,
        pseudo_pred=payload["pseudo_pred"],
        pseudo_scores=pseudo_scores,
        num_classes=10,
        start_quantile=float(selected["threshold_quantile"]),
        min_known_accuracy=float(
            calibrator_cfg["formal_known_accuracy_target"]
        ),
        quantile_grid=[
            selected["threshold_quantile"],
            0.965,
            0.97,
            0.975,
            0.98,
            0.985,
            0.99,
            0.995,
            0.999,
            1.0,
        ],
    )
    thresholds = threshold_safety["thresholds"]

    final_output = output_root / "final"
    _copy_formal_artifacts(Path(config["project"]["output_dir"]), final_output)
    calibrator_path = final_output / "supervised_calibrator.pt"
    save_calibrator(calibrator_path, result)
    selection_manifest = {
        "selected": selected,
        "formal_training": {
            "num_known": result.num_known,
            "num_pseudo": result.num_pseudo,
            "pseudo_scale": selected["pseudo_scale"],
            "fusion_lambda": selected["fusion_lambda"],
            "threshold_quantile": selected["threshold_quantile"],
            "formal_val_known_accuracy": threshold_safety["known_accuracy"],
            "formal_threshold_quantiles_per_class": threshold_safety[
                "quantiles_per_class"
            ],
            "formal_pseudo_recall": threshold_safety["pseudo_recall"],
        },
        "uses_real_unknown_for_selection": False,
        "training_inputs": [
            "formal val_known scores",
            "formal feature-level pseudo_unknown scores",
        ],
    }
    fingerprint_source = json.dumps(selection_manifest, sort_keys=True).encode("utf-8")
    selection_manifest["config_fingerprint_sha256"] = hashlib.sha256(
        fingerprint_source
    ).hexdigest()
    save_json(output_root / "selection_manifest.json", selection_manifest)
    save_json(
        final_output / "supervised_calibrator_training.json",
        {
            "loss_history": result.loss_history,
            "num_known": result.num_known,
            "num_pseudo": result.num_pseudo,
            "seed": result.seed,
            "epochs": result.epochs,
            "lr": result.lr,
        },
    )
    save_json(
        final_output / "fusion.json",
        {
            "fusion_lambda": selected["fusion_lambda"],
            "threshold": None,
            "thresholds_per_class": thresholds,
            "threshold_mode": "supervised_calibrator_classwise",
            "threshold_quantile": None,
            "threshold_quantiles_per_class": threshold_safety[
                "quantiles_per_class"
            ],
            "fusion_mode": "linear",
            "score_calibration": {
                "mode": "supervised_calibrator",
                "path": str(calibrator_path.resolve()),
            },
            "known_rescue": {"enabled": False},
            "calibration_provenance": {
                "pseudo_unknown_direct_training": True,
                "formal_val_known_accuracy": threshold_safety["known_accuracy"],
                "formal_known_accuracy_target": calibrator_cfg[
                    "formal_known_accuracy_target"
                ],
                "uses_real_unknown_for_selection": False,
            },
        },
    )

    final_config = copy.deepcopy(config)
    final_config["project"]["name"] = "oracle_supervised_calibrator"
    final_config["project"]["output_dir"] = str(final_output.resolve())
    final_config["reporting"]["write_root_summaries"] = False
    final_config["unknown_subdivision"]["enabled"] = False
    final_metrics = evaluate_open_set_artifacts(
        final_config,
        ckpt_path=final_output / "best_closed_set.pt",
    )

    baseline = load_json(
        ROOT / "outputs" / "auto_fusion_calibration_comparison" / "baseline_metrics.json"
    )["oracle"]
    current = load_json(
        Path(config["project"]["output_dir"]) / "open_set_metrics.json"
    )
    lco = load_json(
        Path(config["fusion"]["leave_class_out"]["output_dir"])
        / "final"
        / "open_set_metrics.json"
    )
    comparison = {}
    for key in METRIC_KEYS:
        comparison[key] = {
            "old_manual": float(baseline[key]),
            "current_auto": float(current[key]),
            "leave_class_out": float(lco[key]),
            "supervised_calibrator": float(final_metrics[key]),
            "vs_old_manual_pp": float(
                (final_metrics[key] - baseline[key]) * 100.0
            ),
            "vs_current_auto_pp": float(
                (final_metrics[key] - current[key]) * 100.0
            ),
        }
    save_json(
        output_root / "comparison.json",
        {
            "metrics": comparison,
            "selected": selected,
            "final_metrics": final_metrics,
            "uses_real_unknown_for_selection": False,
        },
    )
    _write_comparison(output_root / "comparison.md", comparison)
    print("\nOracle 伪未知监督校准器训练完成")
    print(f"selected={selected['key']}")
    for key in METRIC_KEYS:
        print(f"{key}: {final_metrics[key]:.6f}")


if __name__ == "__main__":
    main()
