from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from functions.common.io import ensure_dir, load_json, load_pickle, save_json
from functions.common.metrics import evaluate_open_set
from functions.data.data_build import build_data_module
from functions.methods.fusion import apply_unknown_rejection, prototype_distance_unknown_score
from functions.methods.leave_class_out import balanced_pseudo_indices
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes
from functions.methods.pseudo_unknown import generate_hybrid_pseudo_unknown
from functions.methods.supervised_calibrator import (
    build_calibrator_features,
    choose_classwise_thresholds_with_pseudo_guard,
    save_calibrator,
    train_calibrator,
)
from functions.model.closed_set import ClosedSetTrainer
from functions.pipeline import evaluate_open_set_artifacts, fit_openmax_artifacts, mine_boundary_artifacts
from functions.subdivision_pipeline import run_unknown_subdivision
from run_wisig import build_config


ROOT = Path(__file__).resolve().parent
SOURCE_EXPERIMENT = "wisig_singleday_osr_k16_u12"
OUTPUT_EXPERIMENT = "wisig_supervised_calibrator_formal"

PSEUDO_SCALE_GRID = [0.75, 1.0, 1.25, 1.5, 2.0]
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPOCH_GRID = [100, 250]
SEED_GRID = [42, 43, 44]
THRESHOLD_QUANTILE_GRID = [0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]
FORMAL_KNOWN_ACCURACY_TARGET = 0.995
PSEUDO_MAX_SAMPLES = 3200
LEARNING_RATE = 0.01
HIDDEN_DIM = 8


def rescale_pseudo_embeddings(
    source_embeddings: np.ndarray,
    pseudo_embeddings: np.ndarray,
    scale: float,
) -> np.ndarray:
    source_embeddings = np.asarray(source_embeddings, dtype=np.float32)
    pseudo_embeddings = np.asarray(pseudo_embeddings, dtype=np.float32)
    return source_embeddings + float(scale) * (pseudo_embeddings - source_embeddings)


def score_embeddings(
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


def copy_formal_artifacts(source_output: Path, final_output: Path) -> None:
    ensure_dir(final_output)
    for name in [
        "best_closed_set.pt",
        "distance_stats.npz",
        "openmax.pkl",
        "boundary_mining.npz",
        "boundary_summary.json",
        "openmax_summary.json",
        "train_summary.json",
    ]:
        source = source_output / name
        if source.exists():
            shutil.copy2(source, final_output / name)


def generate_current_pseudo_unknown(
    config: dict[str, Any],
    source_output: Path,
) -> dict[str, np.ndarray]:
    boundary_file = np.load(source_output / "boundary_mining.npz", allow_pickle=True)
    boundary = {key: boundary_file[key] for key in boundary_file.files}
    pseudo_cfg = config["pseudo_unknown"]
    return generate_hybrid_pseudo_unknown(
        np.asarray(boundary["embeddings"], dtype=np.float32),
        np.asarray(boundary["labels"], dtype=np.int64),
        np.asarray(boundary["prototypes"], dtype=np.float32),
        boundary,
        ordinary_eta=float(pseudo_cfg["ordinary_eta"]),
        critical_eta=float(pseudo_cfg["critical_eta"]),
        critical_beta=float(pseudo_cfg["critical_beta"]),
        ordinary_variations=int(pseudo_cfg["ordinary_variations"]),
        critical_variations=int(pseudo_cfg["critical_variations"]),
        jitter=float(pseudo_cfg["jitter"]),
        use_critical_boundary=True,
        seed=int(config["train"]["seed"]),
    )


def save_current_pseudo_unknown(
    config: dict[str, Any],
    source_output: Path,
    final_output: Path,
) -> None:
    pseudo = generate_current_pseudo_unknown(config, source_output)
    np.savez_compressed(
        final_output / "pseudo_unknown.npz",
        pseudo_embeddings=pseudo["pseudo_embeddings"],
        pseudo_labels=pseudo["pseudo_labels"],
        source_indices=pseudo["source_indices"],
        source_classes=pseudo["source_classes"],
        pseudo_kind=pseudo["pseudo_kind"],
        summary=pseudo["summary"],
        embedding_space="feature",
    )
    save_json(final_output / "pseudo_unknown_summary.json", pseudo["summary"])


def artifact_fingerprint(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in [
        "best_closed_set.pt",
        "boundary_mining.npz",
        "distance_stats.npz",
        "openmax.pkl",
        "pseudo_unknown.npz",
    ]:
        path = output_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def score_selection_candidate(metrics: dict[str, Any]) -> float:
    return float(
        0.60 * metrics["known_accuracy"]
        + 0.35 * metrics["pseudo_unknown_recall"]
        + 0.05 * metrics["macro_f1"]
    )


def build_training_payload(
    config: dict[str, Any],
    source_output: Path,
    pseudo_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
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
    known_pred, known_q_om, known_q_pd = score_embeddings(
        val["embeddings"],
        stats["prototypes"],
        openmax,
        stats["mu"],
        stats["sigma"],
        float(config["model"]["temperature"]),
    )

    boundary = np.load(source_output / "boundary_mining.npz", allow_pickle=True)
    pseudo = generate_current_pseudo_unknown(config, source_output)
    indices = balanced_pseudo_indices(
        pseudo["source_classes"],
        pseudo["pseudo_kind"],
        PSEUDO_MAX_SAMPLES,
        int(seed),
    )
    sources = boundary["embeddings"][pseudo["source_indices"][indices]]
    scaled = rescale_pseudo_embeddings(
        sources,
        pseudo["pseudo_embeddings"][indices],
        float(pseudo_scale),
    )
    pseudo_pred, pseudo_q_om, pseudo_q_pd = score_embeddings(
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
        "num_classes": np.asarray(datamodule.bundle.num_known_classes),
    }


def evaluate_pseudo_guard(
    *,
    payload: dict[str, np.ndarray],
    known_scores: np.ndarray,
    pseudo_scores: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    num_classes = int(payload["num_classes"])
    known_y_pred = apply_unknown_rejection(
        payload["known_pred"],
        known_scores,
        num_classes,
        thresholds_per_class=thresholds,
    )
    pseudo_y_pred = apply_unknown_rejection(
        payload["pseudo_pred"],
        pseudo_scores,
        num_classes,
        thresholds_per_class=thresholds,
    )
    y_true = np.concatenate(
        [
            payload["known_labels"],
            np.full(len(pseudo_y_pred), num_classes, dtype=np.int64),
        ]
    )
    y_pred = np.concatenate([known_y_pred, pseudo_y_pred])
    scores = np.concatenate([known_scores, pseudo_scores])
    metrics = evaluate_open_set(y_true, y_pred, scores, num_classes)
    metrics["pseudo_unknown_recall"] = float(np.mean(pseudo_y_pred == num_classes))
    metrics["known_accuracy"] = float(np.mean(known_y_pred == payload["known_labels"]))
    metrics["selection_score"] = score_selection_candidate(metrics)
    return metrics


def search_calibrator(
    config: dict[str, Any],
    output_root: Path,
    artifact_hash: str,
) -> dict[str, Any]:
    source_output = Path(config["project"]["output_dir"])
    payload_cache: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    candidates: list[dict[str, Any]] = []
    for pseudo_scale in PSEUDO_SCALE_GRID:
        for seed in SEED_GRID:
            payload_cache[(float(pseudo_scale), int(seed))] = build_training_payload(
                config,
                source_output,
                float(pseudo_scale),
                int(seed),
            )
        for fusion_lambda in LAMBDA_GRID:
            for epochs in EPOCH_GRID:
                for seed in SEED_GRID:
                    payload = payload_cache[(float(pseudo_scale), int(seed))]
                    known_features = build_calibrator_features(
                        payload["known_q_om"],
                        payload["known_q_pd"],
                        float(fusion_lambda),
                    )
                    pseudo_features = build_calibrator_features(
                        payload["pseudo_q_om"],
                        payload["pseudo_q_pd"],
                        float(fusion_lambda),
                    )
                    result = train_calibrator(
                        known_features,
                        pseudo_features,
                        seed=int(seed),
                        epochs=int(epochs),
                        lr=LEARNING_RATE,
                        hidden_dim=HIDDEN_DIM,
                    )
                    known_scores = result.model.predict_proba(known_features)
                    pseudo_scores = result.model.predict_proba(pseudo_features)
                    for quantile in THRESHOLD_QUANTILE_GRID:
                        threshold_safety = choose_classwise_thresholds_with_pseudo_guard(
                            known_labels=payload["known_labels"],
                            known_pred=payload["known_pred"],
                            known_scores=known_scores,
                            pseudo_pred=payload["pseudo_pred"],
                            pseudo_scores=pseudo_scores,
                            num_classes=int(payload["num_classes"]),
                            start_quantile=float(quantile),
                            min_known_accuracy=FORMAL_KNOWN_ACCURACY_TARGET,
                            quantile_grid=THRESHOLD_QUANTILE_GRID + [0.995, 0.999, 1.0],
                        )
                        metrics = evaluate_pseudo_guard(
                            payload=payload,
                            known_scores=known_scores,
                            pseudo_scores=pseudo_scores,
                            thresholds=threshold_safety["thresholds"],
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
                                "threshold_safety": threshold_safety,
                                "validation_metrics": metrics,
                                "uses_real_unknown_for_selection": False,
                            }
                        )

    feasible = [
        item
        for item in candidates
        if item["validation_metrics"]["known_accuracy"] >= FORMAL_KNOWN_ACCURACY_TARGET
    ]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["validation_metrics"]["selection_score"],
            item["validation_metrics"]["known_accuracy"],
            item["validation_metrics"]["pseudo_unknown_recall"],
            -int(item["epochs"]),
            -int(item["seed"]),
        ),
    )
    save_json(
        output_root / "selection_candidates.json",
        {
            "selected": selected,
            "candidates": candidates,
            "uses_real_unknown_for_selection": False,
            "pseudo_source": "generated_from_boundary",
            "boundary_source": "recomputed_from_checkpoint",
            "artifact_fingerprint_sha256": artifact_hash,
        },
    )
    return selected


def train_selected_formal(
    config: dict[str, Any],
    selected: dict[str, Any],
    final_output: Path,
) -> dict[str, Any]:
    payload = build_training_payload(
        config,
        Path(config["project"]["output_dir"]),
        float(selected["pseudo_scale"]),
        int(selected["seed"]),
    )
    known_features = build_calibrator_features(
        payload["known_q_om"],
        payload["known_q_pd"],
        float(selected["fusion_lambda"]),
    )
    pseudo_features = build_calibrator_features(
        payload["pseudo_q_om"],
        payload["pseudo_q_pd"],
        float(selected["fusion_lambda"]),
    )
    result = train_calibrator(
        known_features,
        pseudo_features,
        seed=int(selected["seed"]),
        epochs=int(selected["epochs"]),
        lr=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
    )
    known_scores = result.model.predict_proba(known_features)
    pseudo_scores = result.model.predict_proba(pseudo_features)
    threshold_safety = choose_classwise_thresholds_with_pseudo_guard(
        known_labels=payload["known_labels"],
        known_pred=payload["known_pred"],
        known_scores=known_scores,
        pseudo_pred=payload["pseudo_pred"],
        pseudo_scores=pseudo_scores,
        num_classes=int(payload["num_classes"]),
        start_quantile=float(selected["threshold_quantile"]),
        min_known_accuracy=FORMAL_KNOWN_ACCURACY_TARGET,
        quantile_grid=THRESHOLD_QUANTILE_GRID + [0.995, 0.999, 1.0],
    )
    calibrator_path = final_output / "supervised_calibrator.pt"
    save_calibrator(calibrator_path, result)
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
    fusion_summary = {
        "fusion_lambda": float(selected["fusion_lambda"]),
        "threshold": None,
        "thresholds_per_class": threshold_safety["thresholds"],
        "threshold_mode": "supervised_calibrator_classwise",
        "threshold_quantile": None,
        "threshold_quantiles_per_class": threshold_safety["quantiles_per_class"],
        "fusion_mode": "linear",
        "score_calibration": {
            "mode": "supervised_calibrator",
            "path": str(calibrator_path.resolve()),
        },
        "known_rescue": {"enabled": False},
        "calibration_provenance": {
            "pseudo_unknown_direct_training": True,
            "formal_val_known_accuracy": threshold_safety["known_accuracy"],
            "formal_known_accuracy_target": FORMAL_KNOWN_ACCURACY_TARGET,
            "formal_pseudo_recall": threshold_safety["pseudo_recall"],
            "uses_real_unknown_for_selection": False,
        },
    }
    save_json(final_output / "fusion.json", fusion_summary)
    return {
        "training": {
            "num_known": result.num_known,
            "num_pseudo": result.num_pseudo,
            "seed": result.seed,
            "epochs": result.epochs,
            "lr": result.lr,
        },
        "threshold_safety": threshold_safety,
        "fusion_summary": fusion_summary,
    }


def write_comparison(output_root: Path, calibrated: dict[str, Any]) -> None:
    keys = ["known_accuracy", "unknown_recall", "macro_f1", "oscr", "auroc", "fpr95"]
    rows = [{"metric": key, "supervised_calibrator": float(calibrated[key])} for key in keys]
    save_json(output_root / "comparison.json", rows)
    lines = [
        "# WiSig supervised calibrator metrics",
        "",
        "| metric | supervised_calibrator |",
        "| --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['supervised_calibrator'] * 100:.2f} |")
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_formal_wisig_subdivision_config(config: dict[str, Any], predictions_path: Path) -> None:
    subdivision = config["unknown_subdivision"]
    subdivision["enabled"] = True
    subdivision["reuse_open_set_predictions"] = True
    subdivision["open_set_predictions_path"] = str(predictions_path)
    subdivision["output_subdir"] = "unknown_subdivision_true_auto_k_sample_unified_k2_20_merge"
    subdivision["feature_mode"] = "embedding_iq_stats"
    subdivision["pca_dim"] = 96
    subdivision["k_min"] = 2
    subdivision["k_max"] = 20
    subdivision["clustering_backend"] = "gmm_full_direct"
    subdivision["target_num_clusters"] = None
    subdivision["target_k_strength"] = 0.0
    subdivision["use_known_prototype_anchors"] = False
    subdivision["known_reject_margin"] = -1.0
    subdivision["k_selection_mode"] = "sample_unified"
    subdivision["overcluster_extra_clusters"] = 0
    subdivision["overcluster_extra_candidates"] = [0]
    subdivision["m_selection_mode"] = "unsupervised"
    subdivision["m_selection_min_quality_gain"] = 0.01
    subdivision["merge_extra_clusters_to_target"] = False
    subdivision["auto_merge_small_clusters"] = True
    subdivision["auto_merge_max_source_mean_ratio"] = 0.65
    subdivision["auto_merge_min_balance_gain"] = 0.15
    subdivision["direct_confidence_quantile"] = 0.0
    subdivision["direct_min_cluster_size"] = 0


def main() -> None:
    torch.set_num_threads(1)
    config = build_config()
    formal_candidates = sorted((ROOT / "ablations").glob("04_*/wisig/full_subdivision"))
    source_output = formal_candidates[0] if formal_candidates else ROOT / "outputs" / SOURCE_EXPERIMENT
    output_root = ensure_dir(ROOT / "outputs" / OUTPUT_EXPERIMENT)
    final_output = ensure_dir(output_root / "final")

    copy_formal_artifacts(source_output, final_output)
    config["project"]["output_dir"] = str(final_output.resolve())
    config["reporting"]["write_root_summaries"] = False
    mine_boundary_artifacts(config, ckpt_path=final_output / "best_closed_set.pt")
    fit_openmax_artifacts(config, ckpt_path=final_output / "best_closed_set.pt")
    save_current_pseudo_unknown(config, final_output, final_output)
    current_artifact_hash = artifact_fingerprint(final_output)

    selected_path = output_root / "selection_candidates.json"
    selected = None
    if selected_path.exists():
        cached_selection = load_json(selected_path)
        cached_selected = cached_selection["selected"]
        cached_known = float(
            cached_selected.get("threshold_safety", {}).get("known_accuracy", 0.0)
        )
        cached_source = str(cached_selection.get("pseudo_source", ""))
        cached_boundary = str(cached_selection.get("boundary_source", ""))
        cached_hash = str(cached_selection.get("artifact_fingerprint_sha256", ""))
        if (
            cached_known >= FORMAL_KNOWN_ACCURACY_TARGET
            and cached_source == "generated_from_boundary"
            and cached_boundary == "recomputed_from_checkpoint"
            and cached_hash == current_artifact_hash
        ):
            selected = cached_selected
            print(f"[REUSE] selected={selected['key']}")
        else:
            print(
                "[RESELECT] cached WiSig calibrator target "
                f"{cached_known:.4f}, source={cached_source or 'legacy'}, "
                f"boundary={cached_boundary or 'legacy'}, "
                f"fingerprint={'match' if cached_hash == current_artifact_hash else 'changed'}"
            )
    if selected is None:
        selected = search_calibrator(config, output_root, current_artifact_hash)
        print(f"[SELECT] selected={selected['key']}")

    training_manifest = train_selected_formal(config, selected, final_output)
    selection_manifest = {
        "selected": selected,
        "formal_training": training_manifest,
        "uses_real_unknown_for_selection": False,
    }
    fingerprint_source = json.dumps(selection_manifest, sort_keys=True).encode("utf-8")
    selection_manifest["config_fingerprint_sha256"] = hashlib.sha256(
        fingerprint_source
    ).hexdigest()
    save_json(output_root / "selection_manifest.json", selection_manifest)

    final_config = copy.deepcopy(config)
    final_config["project"]["name"] = OUTPUT_EXPERIMENT
    final_config["project"]["output_dir"] = str(final_output.resolve())
    final_config["reporting"]["write_root_summaries"] = False
    final_metrics = evaluate_open_set_artifacts(
        final_config,
        ckpt_path=final_output / "best_closed_set.pt",
    )
    subdivision_config = copy.deepcopy(final_config)
    apply_formal_wisig_subdivision_config(
        subdivision_config,
        final_output / "open_set_predictions.csv",
    )
    subdivision_metrics = run_unknown_subdivision(
        subdivision_config,
        ckpt_path=final_output / "best_closed_set.pt",
    )

    write_comparison(output_root, final_metrics)
    save_json(
        output_root / "final_summary.json",
        {
            "supervised_calibrator_metrics": final_metrics,
            "subdivision_metrics": subdivision_metrics,
            "selected": selected,
            "uses_real_unknown_for_selection": False,
        },
    )
    for key in ["known_accuracy", "unknown_recall", "macro_f1", "oscr", "auroc", "fpr95"]:
        print(f"{key}: {final_metrics[key]:.6f}")
    for key in ["nmi", "ari", "hungarian_accuracy", "coverage_of_total_test_unknown"]:
        print(f"subdivision_{key}: {subdivision_metrics[key]:.6f}")


if __name__ == "__main__":
    main()
