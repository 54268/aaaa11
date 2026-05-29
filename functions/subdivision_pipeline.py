from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.fusion import apply_unknown_rejection, fuse_unknown_score, prototype_distance_unknown_score
from functions.data.data_build import build_data_module
from functions.methods.unknown_subdivision import (
    evaluate_unknown_subdivision,
    fit_feature_preprocessor,
    run_ofscil_subdivision,
)
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes, squared_euclidean
from functions.pipeline import checkpoint_path_for, load_pipeline_config
from functions.model.closed_set import ClosedSetTrainer
from functions.common.io import ensure_dir, load_json, load_pickle, save_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _npz_label_names(path: Path, fallback_prefix: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if "label_name" in payload.files:
        return payload["label_name"].astype(str)
    labels = payload["y"].astype(str)
    return np.asarray([f"{fallback_prefix}_{label}" for label in labels], dtype=str)


def build_cluster_features(
    mode: str,
    embeddings: np.ndarray,
    distances: np.ndarray,
    q_om: np.ndarray,
    q_pd: np.ndarray,
    q_u: np.ndarray,
    known_pred: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    if mode == "embedding":
        return embeddings
    if mode == "embedding_distance":
        return np.concatenate([embeddings, distances], axis=1)
    if mode == "prototype_distance":
        return distances
    if mode == "score_distance":
        return np.concatenate([distances, q_om[:, None], q_pd[:, None], q_u[:, None]], axis=1)
    if mode == "prototype_residual":
        return embeddings - prototypes[known_pred]
    if mode == "residual_distance":
        return np.concatenate([embeddings - prototypes[known_pred], distances], axis=1)
    raise ValueError(f"Unsupported unknown_subdivision.feature_mode: {mode}")


def _known_anchor_features(mode: str, prototypes: np.ndarray) -> np.ndarray:
    distances = np.sqrt(squared_euclidean(prototypes, prototypes))
    zeros = np.zeros(len(prototypes), dtype=np.float32)
    known_pred = np.arange(len(prototypes), dtype=np.int64)
    return build_cluster_features(
        mode,
        prototypes,
        distances,
        zeros,
        zeros,
        zeros,
        known_pred,
        prototypes,
    )


def _write_confusion(path: Path, true_unknown_names: np.ndarray, cluster_labels: np.ndarray) -> None:
    ensure_dir(path.parent)
    class_names = sorted(set(np.asarray(true_unknown_names).astype(str).tolist()))
    cluster_ids = sorted(set(int(x) for x in cluster_labels.tolist()))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["真实未知类", *[f"细分类_{idx}" for idx in cluster_ids]])
        for class_name in class_names:
            mask = true_unknown_names.astype(str) == class_name
            counts = [int((cluster_labels[mask] == cluster_id).sum()) for cluster_id in cluster_ids]
            writer.writerow([class_name, *counts])


def _write_assignments(
    path: Path,
    selected_indices: np.ndarray,
    source_split: np.ndarray,
    source_index: np.ndarray,
    label_names: np.ndarray,
    true_open_labels: np.ndarray,
    unknown_score: np.ndarray,
    labels: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "全局样本索引",
                "来源划分",
                "来源内索引",
                "真实标签名_仅离线评估使用",
                "真实开放集标签_仅离线评估使用",
                "未知分数",
                "未知细分类标签",
                "是否不确定样本",
            ]
        )
        for row in zip(selected_indices, source_split, source_index, label_names, true_open_labels, unknown_score, labels):
            writer.writerow([*row[:-1], int(row[-1]), bool(int(row[-1]) == -1)])


def _cluster_size_stats(labels: np.ndarray) -> dict[str, Any]:
    valid = labels[labels != -1]
    if len(valid) == 0:
        return {"cluster_size_min": 0, "cluster_size_max": 0, "cluster_size_mean": 0.0}
    _, counts = np.unique(valid, return_counts=True)
    return {
        "cluster_size_min": int(counts.min()),
        "cluster_size_max": int(counts.max()),
        "cluster_size_mean": float(counts.mean()),
    }


def _nearest_known_distance(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    if len(prototypes) == 0:
        return np.zeros(len(features), dtype=np.float32)
    return np.sqrt(squared_euclidean(features, prototypes)).min(axis=1)


def _write_report(path: Path, metrics: dict[str, Any], dataset_name: str = "") -> None:
    ensure_dir(path.parent)
    rows = [
        ("method", "聚类协议"),
        ("clustering_backend", "聚类后端"),
        ("feature_mode", "聚类特征"),
        ("resolved_num_clusters", "自动确定的未知细分类数"),
        ("selected_unknown_cache_size", "进入 unknown cache 的样本数"),
        ("uncertain_size", "未分配/不确定样本数"),
        ("uncertain_ratio", "未分配/不确定样本比例"),
        ("cluster_size_min", "最小细分类样本数"),
        ("cluster_size_max", "最大细分类样本数"),
        ("cluster_size_mean", "平均细分类样本数"),
        ("nearest_known_proto_distance_mean", "到最近已知原型的平均距离"),
        ("nearest_known_proto_distance_min", "到最近已知原型的最小距离"),
        ("nmi", "归一化互信息，越高表示聚类与真实未知类越一致"),
        ("ari", "调整兰德指数，越高表示聚类与真实未知类越一致"),
        ("purity", "纯度，每个聚类中主导真实类的占比"),
        ("hungarian_accuracy", "匈牙利匹配后的聚类准确率"),
        ("unknown_cache_precision", "unknown cache 中真实未知样本占比"),
        ("unknown_cache_recall", "真实未知样本进入 unknown cache 的比例"),
        ("coverage_of_total_test_unknown", "完成细分的真实未知样本覆盖率"),
    ]
    title_suffix = f"{dataset_name} " if dataset_name else ""
    lines = [
        f"# {title_suffix}未知类细分结果",
        "",
        "当前协议：已知原型引导的半监督聚类（支持 KMeans / Agglomerative / GMM 后端）。",
        "",
        "| 指标键 | 中文说明 | 数值 |",
        "| --- | --- | ---: |",
    ]
    for key, cn_name in rows:
        value = metrics.get(key)
        value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {cn_name} | {value_str} |")
    lines.append("")
    lines.append("真实未知标签只用于离线 NMI、ARI、纯度、匈牙利准确率和混淆分析，不参与训练或在线判别。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_unknown_subdivision(
    config_or_path: str | Path | dict[str, Any],
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_or_path)
    cfg = config.get("unknown_subdivision", {})
    if cfg.get("enabled", True) is False:
        print("unknown_subdivision.enabled=false；跳过未知类细分。")
        return {}

    output_dir = ensure_dir(config["project"]["output_dir"])
    subdivision_dir = ensure_dir(output_dir / str(cfg.get("output_subdir", "unknown_subdivision")))

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, ckpt_path))

    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    fusion_cfg = load_json(output_dir / "fusion.json")
    stats_file = np.load(output_dir / "distance_stats.npz")

    test_known = trainer.extract_embeddings(datamodule.test_known_dataloader())
    test_unknown = trainer.extract_embeddings(datamodule.test_unknown_dataloader())
    prototypes = test_known["prototypes"]
    unknown_label = datamodule.bundle.num_known_classes

    all_embeddings = np.concatenate([test_known["embeddings"], test_unknown["embeddings"]], axis=0)
    true_open_labels = np.concatenate(
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
    y_pred = apply_unknown_rejection(
        known_pred=known_pred,
        q_u=q_u,
        unknown_label=unknown_label,
        threshold=fusion_cfg.get("threshold"),
        thresholds_per_class=fusion_cfg.get("thresholds_per_class"),
    )
    selected_mask = y_pred == unknown_label

    data_root = Path(config["data"]["root"])
    known_names = _npz_label_names(data_root / "test_known.npz", "known")
    unknown_names = _npz_label_names(data_root / "test_unknown.npz", "unknown")
    all_names = np.concatenate([known_names, unknown_names], axis=0)
    source_split = np.concatenate(
        [
            np.full(len(known_names), "known", dtype="<U7"),
            np.full(len(unknown_names), "unknown", dtype="<U7"),
        ]
    )
    source_index = np.concatenate([np.arange(len(known_names)), np.arange(len(unknown_names))], axis=0)

    selected_indices = np.where(selected_mask)[0]
    selected_embeddings = all_embeddings[selected_mask]
    selected_distances = distances[selected_mask]
    selected_known_pred = known_pred[selected_mask]
    selected_q_om = q_om[selected_mask]
    selected_q_pd = q_pd[selected_mask]
    selected_q_u = q_u[selected_mask]
    selected_true_open = true_open_labels[selected_mask]
    selected_names = all_names[selected_mask]

    feature_mode = str(cfg.get("feature_mode", "embedding_distance"))
    sample_features = build_cluster_features(
        feature_mode,
        selected_embeddings,
        selected_distances,
        selected_q_om,
        selected_q_pd,
        selected_q_u,
        selected_known_pred,
        prototypes,
    )
    anchor_features = _known_anchor_features(feature_mode, prototypes)
    preprocessor = fit_feature_preprocessor(
        np.vstack([sample_features, anchor_features]),
        pca_dim=int(cfg.get("pca_dim", 16)),
    )
    prepared_samples = preprocessor.transform(sample_features)
    prepared_anchors = preprocessor.transform(anchor_features)

    target_num_clusters = cfg.get("target_num_clusters")
    if target_num_clusters is not None:
        target_num_clusters = int(target_num_clusters)
    merge_similarity_threshold = cfg.get("merge_similarity_threshold")
    if merge_similarity_threshold is not None:
        merge_similarity_threshold = float(merge_similarity_threshold)

    result = run_ofscil_subdivision(
        prepared_samples,
        prepared_anchors,
        k_min=int(cfg.get("k_min", 2)),
        k_max=int(cfg.get("k_max", 15)),
        seed=int(config.get("train", {}).get("seed", 42)),
        auto_sample_size=int(cfg.get("auto_sample_size", 3000)),
        assignment_margin=float(cfg.get("assignment_margin", 0.0)),
        known_reject_margin=float(cfg.get("known_reject_margin", 0.0)),
        backend=str(cfg.get("clustering_backend", "kmeans")),
        target_num_clusters=target_num_clusters,
        target_k_strength=float(cfg.get("target_k_strength", 0.10)),
        uncertain_penalty=float(cfg.get("uncertain_penalty", 0.15)),
        stability_weight=float(cfg.get("stability_weight", 0.30)),
        db_weight=float(cfg.get("db_weight", 0.25)),
        ch_weight=float(cfg.get("ch_weight", 0.30)),
        n_init=int(cfg.get("n_init", 30)),
        merge_similarity_threshold=merge_similarity_threshold,
        agg_sample_size=int(cfg.get("agg_sample_size", 8000)),
    )

    true_unknown_mask = selected_true_open == unknown_label
    eval_mask = true_unknown_mask & (result.labels != -1)
    metrics = evaluate_unknown_subdivision(selected_names[eval_mask], result.labels[eval_mask])
    nearest_known = _nearest_known_distance(selected_embeddings, prototypes)
    assigned_known = nearest_known[result.labels != -1] if np.any(result.labels != -1) else nearest_known
    metrics.update(
        {
            "method": f"prototype_guided_{cfg.get('clustering_backend', 'kmeans')}",
            "feature_mode": feature_mode,
            "clustering_backend": str(cfg.get("clustering_backend", "kmeans")),
            "resolved_num_clusters": int(result.resolved_k),
            "selected_unknown_cache_size": int(len(selected_indices)),
            "uncertain_size": int((result.labels == -1).sum()),
            "uncertain_ratio": float((result.labels == -1).mean()) if len(result.labels) else 0.0,
            "nearest_known_proto_distance_mean": float(assigned_known.mean()) if len(assigned_known) else 0.0,
            "nearest_known_proto_distance_min": float(assigned_known.min()) if len(assigned_known) else 0.0,
            "unknown_cache_precision": float(true_unknown_mask.mean()) if len(true_unknown_mask) else 0.0,
            "unknown_cache_recall": float(true_unknown_mask.sum() / max(len(test_unknown["labels"]), 1)),
            "coverage_of_selected_true_unknown": float(eval_mask.sum() / max(int(true_unknown_mask.sum()), 1)),
            "coverage_of_total_test_unknown": float(eval_mask.sum() / max(len(test_unknown["labels"]), 1)),
            "suspected_known_noise_size": int(result.suspected_known_mask.sum()),
            **_cluster_size_stats(result.labels),
        }
    )

    np.save(subdivision_dir / "unknown_subdivision_labels.npy", result.labels)
    np.save(subdivision_dir / "unknown_subdivision_centers.npy", result.centers)
    save_json(subdivision_dir / "k_search_history.json", _jsonable(result.k_search_history))
    save_json(subdivision_dir / "unknown_subdivision_metrics.json", _jsonable(metrics))
    dataset_label = str(config.get("prep", {}).get("kind", "")).replace("_compact", "").replace("_sigmf", "")
    dataset_label = dataset_label.capitalize() if dataset_label else ""
    _write_report(subdivision_dir / "unknown_subdivision_report.md", metrics, dataset_name=dataset_label)
    _write_confusion(subdivision_dir / "true_unknown_confusion.csv", selected_names[true_unknown_mask], result.labels[true_unknown_mask])
    _write_assignments(
        subdivision_dir / "unknown_subdivision_assignments.csv",
        selected_indices,
        source_split[selected_mask],
        source_index[selected_mask],
        selected_names,
        selected_true_open,
        selected_q_u,
        result.labels,
    )

    print(f"未知类细分报告：{subdivision_dir / 'unknown_subdivision_report.md'}")
    for key in ["nmi", "ari", "purity", "hungarian_accuracy", "resolved_num_clusters", "uncertain_size"]:
        value = metrics.get(key)
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return metrics



