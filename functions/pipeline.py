from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.fusion import (
    apply_unknown_rejection,
    apply_known_rescue,
    apply_score_calibration,
    FusionResult,
    fit_score_calibration,
    fuse_unknown_score,
    prototype_distance_unknown_score,
    search_fusion_params,
)
from functions.data.data_build import build_data_module
from functions.data.prep_oracle import prepare_oracle_sigmf
from functions.data.prep_wisig import prepare_wisig_compact
from functions.methods.boundary_mining import mine_boundary_samples
from functions.methods.pseudo_unknown import generate_hybrid_pseudo_unknown
from functions.methods.prototype_utils import activations_from_distances, collect_distance_stats, predict_with_prototypes
from functions.model.closed_set import ClosedSetTrainer
from functions.common.io import ensure_dir, load_json, load_pickle, save_json, save_npz, save_pickle
from functions.common.metrics import (
    evaluate_open_set as compute_open_set_metrics,
    save_confusion_matrix,
    save_prediction_csv,
)
from functions.common.reporting import dataset_summary_path, write_final_report, write_summary_index
from functions.common.seed import set_seed
from functions.common.visualization import generate_open_set_figures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_config(config_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config_or_path, dict):
        raise TypeError("请在根目录入口文件顶部直接修改 CONFIG 字典，不再使用 YAML 配置文件。")
    return config_or_path


def checkpoint_path_for(config: dict[str, Any], ckpt_path: str | Path | None = None) -> Path:
    if ckpt_path:
        return Path(ckpt_path).resolve()
    return Path(config["project"]["output_dir"]) / "best_closed_set.pt"


def format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def should_run_unknown_subdivision(config: dict[str, Any], requested: bool) -> bool:
    if not requested:
        return False
    return bool(config.get("unknown_subdivision", {}).get("enabled", False))


def prepare_data(config_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    prep_kind = config.get("prep", {}).get("kind", "")
    if prep_kind == "wisig_compact":
        return prepare_wisig_compact(config)
    if prep_kind == "oracle_sigmf":
        return prepare_oracle_sigmf(config)
    return {"skipped": True, "reason": f"unsupported prep.kind: {prep_kind}"}


def train_closed_set(config_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    set_seed(int(config["train"]["seed"]))
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(
        config=config,
        num_classes=datamodule.bundle.num_known_classes,
        signal_length=datamodule.bundle.signal_length,
    )
    artifacts = trainer.fit(datamodule.train_dataloader(), datamodule.val_known_dataloader(), output_dir)
    summary = {
        "checkpoint_path": artifacts.checkpoint_path,
        "best_val_acc": artifacts.best_val_acc,
        "num_known_classes": datamodule.bundle.num_known_classes,
        "signal_length": datamodule.bundle.signal_length,
    }
    save_json(output_dir / "train_summary.json", summary)
    return summary


def mine_boundary_artifacts(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, ckpt_path))

    payload = trainer.extract_embeddings(datamodule.train_dataloader())
    boundary = mine_boundary_samples(
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        k=int(config["boundary"]["k"]),
        beta=float(config["boundary"].get("beta", 1.0)),
        alpha=float(config["boundary"]["alpha"]),
        top_m=int(config["boundary"]["top_m"]),
        ordinary_edge_ratio=float(config["boundary"]["ordinary_edge_ratio"]),
    )
    save_npz(
        output_dir / "boundary_mining.npz",
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        embedding_space="original",
        scores=boundary["scores"],
        local_edge=boundary["local_edge"],
        prototype_deviation=boundary["prototype_deviation"],
        local_sparsity=boundary["local_sparsity"],
        local_marginality=boundary["local_marginality"],
        gap=boundary["gap"],
        competition_distance=boundary["competition_distance"],
        local_scale=boundary["local_scale"],
        nearest_foreign=boundary["nearest_foreign"],
        marginal_mask=boundary["marginal_mask"].astype("int64"),
        critical_mask=boundary["critical_mask"].astype("int64"),
        ordinary_edge_mask=boundary["ordinary_edge_mask"].astype("int64"),
        noise_mask=boundary["noise_mask"].astype("int64"),
    )
    save_json(output_dir / "boundary_summary.json", boundary["summary"])
    return boundary["summary"]


def generate_pseudo_unknown_artifacts(config_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    output_dir = ensure_dir(config["project"]["output_dir"])
    boundary_file = np.load(output_dir / "boundary_mining.npz", allow_pickle=True)

    boundary = {
        "scores": boundary_file["scores"],
        "local_scale": boundary_file["local_scale"],
        "nearest_foreign": boundary_file["nearest_foreign"],
        "critical_mask": boundary_file["critical_mask"].astype(bool),
        "ordinary_edge_mask": boundary_file["ordinary_edge_mask"].astype(bool),
    }
    pseudo = generate_hybrid_pseudo_unknown(
        embeddings=boundary_file["embeddings"],
        labels=boundary_file["labels"],
        prototypes=boundary_file["prototypes"],
        boundary_result=boundary,
        ordinary_eta=float(config["pseudo_unknown"]["ordinary_eta"]),
        critical_eta=float(config["pseudo_unknown"]["critical_eta"]),
        critical_beta=float(config["pseudo_unknown"]["critical_beta"]),
        ordinary_variations=int(config["pseudo_unknown"]["ordinary_variations"]),
        critical_variations=int(config["pseudo_unknown"]["critical_variations"]),
        jitter=float(config["pseudo_unknown"]["jitter"]),
        enable_conflict_protection=bool(config["pseudo_unknown"].get("enable_conflict_protection", True)),
        seed=int(config["train"]["seed"]),
    )
    pseudo["embedding_space"] = str(boundary_file["embedding_space"]) if "embedding_space" in boundary_file.files else "original"
    save_npz(output_dir / "pseudo_unknown.npz", **pseudo)
    save_json(output_dir / "pseudo_unknown_summary.json", pseudo["summary"])
    return pseudo["summary"]


def fit_openmax_artifacts(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, ckpt_path))
    payload = trainer.extract_embeddings(datamodule.val_known_dataloader())

    pred, _, distances = predict_with_prototypes(
        payload["embeddings"],
        payload["prototypes"],
        float(config["model"]["temperature"]),
    )
    openmax = OpenMaxCalibrator(
        alpha_rank=int(config["openmax"]["alpha_rank"]),
        tail_size=int(config["openmax"]["tail_size"]),
        backend=str(config["openmax"].get("backend", "native")),
        distance_type=str(config["openmax"].get("distance_type", "eucl")),
        euclid_weight=float(config["openmax"].get("euclid_weight", 1.0)),
    )
    openmax.fit(activations_from_distances(distances), payload["labels"], pred)
    save_pickle(output_dir / "openmax.pkl", openmax.state_dict())

    stats = collect_distance_stats(
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        prototypes=payload["prototypes"],
        temperature=float(config["model"]["temperature"]),
    )
    save_npz(
        output_dir / "distance_stats.npz",
        mu=stats["mu"],
        sigma=stats["sigma"],
        prototypes=payload["prototypes"],
    )
    summary = {
        "alpha_rank": int(config["openmax"]["alpha_rank"]),
        "tail_size": int(config["openmax"]["tail_size"]),
        "backend": str(config["openmax"].get("backend", "native")),
        "num_val_samples": int(len(payload["labels"])),
    }
    save_json(output_dir / "openmax_summary.json", summary)
    return summary


def calibrate_fusion_artifacts(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, ckpt_path))

    val_payload = trainer.extract_embeddings(datamodule.val_known_dataloader())
    pseudo_file = np.load(output_dir / "pseudo_unknown.npz", allow_pickle=True)
    stats_file = np.load(output_dir / "distance_stats.npz")
    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    unknown_label = datamodule.bundle.num_known_classes

    val_pred, _, val_dist = predict_with_prototypes(
        val_payload["embeddings"],
        val_payload["prototypes"],
        float(config["model"]["temperature"]),
    )
    val_openmax = openmax.predict(activations_from_distances(val_dist))
    val_q_om = val_openmax["unknown_prob"]
    val_q_pd = prototype_distance_unknown_score(val_dist, val_pred, stats_file["mu"], stats_file["sigma"])

    pseudo_embeddings = pseudo_file["pseudo_embeddings"]
    pseudo_pred, _, pseudo_dist = predict_with_prototypes(
        pseudo_embeddings,
        val_payload["prototypes"],
        float(config["model"]["temperature"]),
    )
    pseudo_openmax = openmax.predict(activations_from_distances(pseudo_dist))
    pseudo_q_om = pseudo_openmax["unknown_prob"]
    pseudo_q_pd = prototype_distance_unknown_score(pseudo_dist, pseudo_pred, stats_file["mu"], stats_file["sigma"])

    y_true = np.concatenate(
        [
            val_payload["labels"],
            np.full(len(pseudo_embeddings), unknown_label, dtype=np.int64),
        ]
    )
    known_pred = np.concatenate([val_pred, pseudo_pred])
    q_om = np.concatenate([val_q_om, pseudo_q_om])
    q_pd = np.concatenate([val_q_pd, pseudo_q_pd])
    all_distances = np.concatenate([val_dist, pseudo_dist], axis=0)

    fusion_mode = str(config["fusion"].get("mode", "linear"))
    score_calibration_mode = str(config["fusion"].get("score_calibration", "none"))
    score_calibration = None
    known_rescue_cfg = config["fusion"].get("known_rescue")
    manual_thresholds = config["fusion"].get("manual_thresholds_per_class")
    manual_threshold = config["fusion"].get("manual_threshold")
    if manual_thresholds is not None:
        fusion_lambda = float(config["fusion"].get("manual_fusion_lambda", list(config["fusion"]["lambda_grid"])[0]))
        q_u_raw = fuse_unknown_score(q_om, q_pd, fusion_lambda, mode=fusion_mode)
        val_q_u_raw = fuse_unknown_score(val_q_om, val_q_pd, fusion_lambda, mode=fusion_mode)
        score_calibration = fit_score_calibration(val_q_u_raw, val_pred, unknown_label, score_calibration_mode)
        q_u = apply_score_calibration(q_u_raw, known_pred, score_calibration)
        y_pred = apply_unknown_rejection(
            known_pred=known_pred,
            q_u=q_u,
            unknown_label=unknown_label,
            thresholds_per_class=manual_thresholds,
        )
        y_pred = apply_known_rescue(y_pred, known_pred, q_u, all_distances, unknown_label, known_rescue_cfg)
        result = FusionResult(
            fusion_lambda=fusion_lambda,
            threshold=None,
            thresholds_per_class=[float(value) for value in manual_thresholds],
            threshold_mode="manual_classwise",
            threshold_quantile=None,
            metrics=compute_open_set_metrics(y_true, y_pred, q_u, unknown_label),
        )
    elif manual_threshold is not None:
        fusion_lambda = float(config["fusion"].get("manual_fusion_lambda", list(config["fusion"]["lambda_grid"])[0]))
        q_u_raw = fuse_unknown_score(q_om, q_pd, fusion_lambda, mode=fusion_mode)
        val_q_u_raw = fuse_unknown_score(val_q_om, val_q_pd, fusion_lambda, mode=fusion_mode)
        score_calibration = fit_score_calibration(val_q_u_raw, val_pred, unknown_label, score_calibration_mode)
        q_u = apply_score_calibration(q_u_raw, known_pred, score_calibration)
        y_pred = apply_unknown_rejection(
            known_pred=known_pred,
            q_u=q_u,
            unknown_label=unknown_label,
            threshold=float(manual_threshold),
        )
        y_pred = apply_known_rescue(y_pred, known_pred, q_u, all_distances, unknown_label, known_rescue_cfg)
        result = FusionResult(
            fusion_lambda=fusion_lambda,
            threshold=float(manual_threshold),
            thresholds_per_class=None,
            threshold_mode="manual_global",
            threshold_quantile=None,
            metrics=compute_open_set_metrics(y_true, y_pred, q_u, unknown_label),
        )
    else:
        result = search_fusion_params(
            y_true=y_true,
            known_pred=known_pred,
            q_om=q_om,
            q_pd=q_pd,
            unknown_label=unknown_label,
            lambda_grid=config["fusion"]["lambda_grid"],
            threshold_grid=config["fusion"]["threshold_grid"],
            selection_weights=config["fusion"].get("selection_weights"),
            known_quantile_floor=config["fusion"].get("known_quantile_floor"),
            min_known_accuracy=config["fusion"].get("min_known_accuracy"),
            threshold_mode=str(config["fusion"].get("threshold_mode", "global")),
            classwise_quantile_grid=config["fusion"].get("classwise_quantile_grid"),
            classwise_known_weight=float(config["fusion"].get("classwise_known_weight", 0.55)),
            classwise_unknown_weight=float(config["fusion"].get("classwise_unknown_weight", 0.45)),
            classwise_min_known_accept=config["fusion"].get("classwise_min_known_accept"),
            fusion_mode=fusion_mode,
        )
    summary = {
        "fusion_lambda": result.fusion_lambda,
        "threshold": result.threshold,
        "thresholds_per_class": result.thresholds_per_class,
        "threshold_mode": result.threshold_mode,
        "threshold_quantile": result.threshold_quantile,
        "fusion_mode": fusion_mode,
        "score_calibration": score_calibration or {"mode": "none"},
        "known_rescue": known_rescue_cfg or {"enabled": False},
        "metrics": result.metrics,
    }
    save_json(output_dir / "fusion.json", summary)
    return summary


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def _build_experiment_metadata(config: dict[str, Any], fusion_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    data_root = Path(config["data"]["root"])
    dataset_summary = _load_json_if_exists(data_root / "dataset_summary.json")
    split_meta = _load_json_if_exists(config.get("prep", {}).get("split_file"))

    known_tx = split_meta.get("known_classes", dataset_summary.get("known_classes", []))
    unknown_tx = split_meta.get("unknown_classes", dataset_summary.get("unknown_classes", []))
    rx_used = None
    for source in (split_meta, dataset_summary, config.get("prep", {})):
        for key in ("include_rx", "rx_used", "fixed_rx"):
            if key in source and source[key] is not None:
                rx_used = source[key]
                break
        if rx_used is not None:
            break

    rx_applicable = rx_used is not None
    if isinstance(rx_used, str):
        rx_used = [rx_used]
    elif isinstance(rx_used, (tuple, set)):
        rx_used = list(rx_used)
    elif not isinstance(rx_used, list):
        rx_used = ["N/A"] if not rx_applicable else [rx_used]

    if not rx_applicable:
        number_of_rx_used: int | str = "N/A"
        rx_mode = "not_applicable"
    else:
        number_of_rx_used = int(len(rx_used))
        rx_mode = "fixed" if len(rx_used) == 1 else "mixed" if len(rx_used) > 1 else "unspecified"
    threshold_mode = str(fusion_cfg.get("threshold_mode", config.get("fusion", {}).get("threshold_mode", "global")))

    metadata = {
        "threshold_strategy_used": threshold_mode,
        "threshold_mode": threshold_mode,
        "threshold": fusion_cfg.get("threshold"),
        "threshold_quantile": fusion_cfg.get("threshold_quantile"),
        "score_calibration_mode": fusion_cfg.get("score_calibration", {}).get("mode", "none"),
        "known_rescue_enabled": bool(fusion_cfg.get("known_rescue", {}).get("enabled", False)),
        "number_of_tx": int(len(known_tx) + len(unknown_tx)),
        "number_of_rx_used": number_of_rx_used,
        "rx_mode": rx_mode,
        "rx_used": rx_used,
        "known_tx_list": known_tx,
        "unknown_tx_list": unknown_tx,
        "known_classes": int(len(known_tx)),
        "unknown_classes": int(len(unknown_tx)),
        "train_sample_count": int(dataset_summary.get("train_sample_count", dataset_summary.get("num_train_known", 0))),
        "val_sample_count": int(dataset_summary.get("val_sample_count", dataset_summary.get("num_val_known", 0))),
        "test_known_sample_count": int(dataset_summary.get("test_known_sample_count", dataset_summary.get("num_test_known", 0))),
        "test_unknown_sample_count": int(dataset_summary.get("test_unknown_sample_count", dataset_summary.get("num_test_unknown", 0))),
        "split_file": str(config.get("prep", {}).get("split_file", "")),
        "output_dir": str(output_dir),
    }
    if fusion_cfg.get("thresholds_per_class") is not None:
        metadata["thresholds_per_class"] = fusion_cfg.get("thresholds_per_class")
    return metadata


def evaluate_open_set_artifacts(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    output_dir = ensure_dir(config["project"]["output_dir"])

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, ckpt_path))

    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    fusion_cfg = load_json(output_dir / "fusion.json")
    stats_file = np.load(output_dir / "distance_stats.npz")

    test_known = trainer.extract_embeddings(datamodule.test_known_dataloader())
    test_unknown = trainer.extract_embeddings(datamodule.test_unknown_dataloader())
    prototypes = test_known["prototypes"]
    np.save(output_dir / "known_prototypes.npy", prototypes)
    unknown_label = datamodule.bundle.num_known_classes

    all_embeddings = np.concatenate([test_known["embeddings"], test_unknown["embeddings"]], axis=0)
    all_labels = np.concatenate(
        [
            test_known["labels"],
            np.full(len(test_unknown["labels"]), unknown_label, dtype=np.int64),
        ]
    )
    known_pred, _, distances = predict_with_prototypes(
        all_embeddings,
        prototypes,
        float(config["model"]["temperature"]),
    )
    openmax_out = openmax.predict(activations_from_distances(distances))
    q_om = openmax_out["unknown_prob"]
    q_pd = prototype_distance_unknown_score(distances, known_pred, stats_file["mu"], stats_file["sigma"])
    q_u = fuse_unknown_score(
        q_om,
        q_pd,
        float(fusion_cfg["fusion_lambda"]),
        mode=str(fusion_cfg.get("fusion_mode", config["fusion"].get("mode", "linear"))),
    )
    q_u = apply_score_calibration(q_u, known_pred, fusion_cfg.get("score_calibration"))
    y_pred = apply_unknown_rejection(
        known_pred=known_pred,
        q_u=q_u,
        unknown_label=unknown_label,
        threshold=fusion_cfg.get("threshold"),
        thresholds_per_class=fusion_cfg.get("thresholds_per_class"),
    )
    y_pred = apply_known_rescue(
        y_pred=y_pred,
        known_pred=known_pred,
        q_u=q_u,
        distances=distances,
        unknown_label=unknown_label,
        rescue_config=fusion_cfg.get("known_rescue", config["fusion"].get("known_rescue")),
    )

    metrics = compute_open_set_metrics(all_labels, y_pred, q_u, unknown_label)
    metrics.update(_build_experiment_metadata(config, fusion_cfg, output_dir))
    d_min = distances[np.arange(len(distances)), known_pred]
    save_json(output_dir / "open_set_metrics.json", metrics)
    save_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        all_labels,
        y_pred,
        labels=list(range(datamodule.bundle.num_known_classes)) + [unknown_label],
    )
    if bool(config["eval"].get("save_predictions", True)):
        save_prediction_csv(output_dir / "open_set_predictions.csv", all_labels, y_pred, q_u, q_om, q_pd, d_min)

    dataset_name = config.get("prep", {}).get("kind", config["data"]["mode"])
    figure_dir = generate_open_set_figures(
        config=config,
        output_dir=output_dir,
        y_true=all_labels,
        y_pred=y_pred,
        unknown_score=q_u,
        unknown_label=unknown_label,
        threshold=float(fusion_cfg["threshold"]) if fusion_cfg.get("threshold") is not None else None,
    )

    notes = [f"图表目录：{figure_dir}"]
    split_file = config.get("prep", {}).get("split_file")
    if split_file:
        notes.append(f"划分文件：{split_file}")
    if config["openmax"].get("backend", "native") != "native":
        notes.append(f"OpenMax 后端：{config['openmax']['backend']}")

    write_final_report(
        path=output_dir / "final_report.md",
        metrics=metrics,
        config_path=str(config["_config_path"]),
        output_dir=str(output_dir),
        dataset_name=str(dataset_name),
        extra_notes=notes,
    )
    if bool(config.get("reporting", {}).get("write_root_summaries", True)):
        write_final_report(
            path=dataset_summary_path(PROJECT_ROOT, str(dataset_name), str(output_dir)),
            metrics=metrics,
            config_path=str(config["_config_path"]),
            output_dir=str(output_dir),
            dataset_name=str(dataset_name),
            extra_notes=notes,
        )
        write_summary_index(
            path=PROJECT_ROOT / "outputs" / "summaries" / "RESULT_SUMMARY.md",
            entries=[
                {"label": "WiSig 结果", "path": "RESULT_SUMMARY_WISIG.md"},
                {"label": "Oracle 结果", "path": "RESULT_SUMMARY_ORACLE.md"},
            ],
            latest_dataset=str(dataset_name),
            latest_output_dir=str(output_dir),
            latest_config_path=str(config["_config_path"]),
        )
    return metrics


def run_post_training_open_set_steps(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    mine_boundary_artifacts(config, ckpt_path=ckpt_path)
    generate_pseudo_unknown_artifacts(config)
    fit_openmax_artifacts(config, ckpt_path=ckpt_path)
    calibrate_fusion_artifacts(config, ckpt_path=ckpt_path)
    return evaluate_open_set_artifacts(config, ckpt_path=ckpt_path)


def run_osr_pipeline(
    config_or_path: str | Path | dict[str, Any],
    *,
    skip_prepare: bool = False,
    skip_training: bool = False,
    ckpt_path: str | Path | None = None,
    with_unknown_subdivision: bool = False,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    if not skip_prepare:
        prepare_data(config)

    effective_ckpt = ckpt_path
    if skip_training:
        expected_ckpt = checkpoint_path_for(config, ckpt_path)
        if not expected_ckpt.exists():
            raise FileNotFoundError(f"未找到闭集模型权重：{expected_ckpt}")
        effective_ckpt = expected_ckpt
    else:
        train_closed_set(config)

    metrics = run_post_training_open_set_steps(config, ckpt_path=effective_ckpt)
    if should_run_unknown_subdivision(config, with_unknown_subdivision):
        from functions.subdivision_pipeline import run_unknown_subdivision

        run_unknown_subdivision(config, ckpt_path=effective_ckpt)
    elif with_unknown_subdivision:
        print("[SKIP] unknown_subdivision.enabled=false；本次只运行拒识评估。")
    return metrics



