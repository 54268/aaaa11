from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.fusion import (
    apply_known_rescue,
    apply_score_calibration,
    apply_unknown_rejection,
    fuse_unknown_score,
    prototype_distance_unknown_score,
)
from functions.data.data_build import build_data_module
from functions.methods.unknown_subdivision import (
    _select_k_by_unified_score,
    evaluate_unknown_subdivision,
    fit_feature_preprocessor,
    run_ofscil_subdivision,
)
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes, squared_euclidean
from functions.pipeline import checkpoint_path_for, load_pipeline_config
from functions.model.closed_set import ClosedSetTrainer
from functions.common.io import ensure_dir, load_json, load_pickle, save_json


IQ_STAT_FEATURE_DIM = 49


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


def _normalize_signal_samples(samples: np.ndarray, normalize: str) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if normalize == "per_sample":
        mean = samples.mean(axis=2, keepdims=True)
        std = samples.std(axis=2, keepdims=True)
        return (samples - mean) / (std + 1e-6)
    return samples


def _sequence_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    centered = values - mean[:, None]
    safe_std = np.maximum(std, 1e-8)
    skewness = (centered**3).mean(axis=1) / (safe_std**3)
    excess_kurtosis = (centered**4).mean(axis=1) / (safe_std**4) - 3.0
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1).T
    stats = np.column_stack(
        [
            mean,
            std,
            values.min(axis=1),
            values.max(axis=1),
            skewness,
            excess_kurtosis,
            quantiles,
        ]
    )
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_iq_stat_features(signal_samples: np.ndarray | None, num_rows: int) -> np.ndarray:
    if signal_samples is None:
        return np.zeros((int(num_rows), IQ_STAT_FEATURE_DIM), dtype=np.float32)
    samples = np.asarray(signal_samples, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[1] != 2:
        raise ValueError("embedding_stats 需要形状为 [N, 2, L] 的 I/Q 样本。")

    i_part = samples[:, 0, :]
    q_part = samples[:, 1, :]
    complex_signal = i_part + 1j * q_part
    amplitude = np.abs(complex_signal)
    phase_delta = np.angle(complex_signal[:, 1:] * np.conj(complex_signal[:, :-1]))

    power = np.mean(np.abs(complex_signal) ** 2, axis=1) + 1e-8
    second_order = np.mean(complex_signal**2, axis=1)
    centered_i = i_part - i_part.mean(axis=1, keepdims=True)
    centered_q = q_part - q_part.mean(axis=1, keepdims=True)

    extras = np.column_stack(
        [
            np.mean(i_part * q_part, axis=1),
            np.mean(centered_i * centered_q, axis=1),
            np.abs(second_order) / power,
            np.real(second_order),
            np.imag(second_order),
        ]
    ).astype(np.float32)
    features = np.concatenate(
        [
            _sequence_stats(i_part),
            _sequence_stats(q_part),
            _sequence_stats(amplitude),
            _sequence_stats(phase_delta),
            extras,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def build_cluster_features(
    mode: str,
    embeddings: np.ndarray,
    distances: np.ndarray,
    q_om: np.ndarray,
    q_pd: np.ndarray,
    q_u: np.ndarray,
    known_pred: np.ndarray,
    prototypes: np.ndarray,
    signal_samples: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "embedding":
        return embeddings
    if mode in {"iq_stats", "iq_descriptors"}:
        return build_iq_stat_features(signal_samples, len(embeddings))
    if mode in {"embedding_stats", "embedding_iq_stats"}:
        return np.concatenate([embeddings, build_iq_stat_features(signal_samples, len(embeddings))], axis=1)
    if mode == "embedding_distance":
        return np.concatenate([embeddings, distances], axis=1)
    if mode == "embedding_score_distance":
        return np.concatenate([embeddings, distances, q_om[:, None], q_pd[:, None], q_u[:, None]], axis=1)
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
    if mode in {"iq_stats", "iq_descriptors"}:
        raise ValueError("I/Q descriptor-only subdivision does not support known prototype anchors.")
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


def _load_open_set_predictions(path: str | Path, expected_size: int | None = None) -> dict[str, np.ndarray]:
    path = Path(path)
    required = ["y_true", "y_pred", "unknown_score", "q_om", "q_pd"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Open-set prediction file has no header: {path}")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Open-set prediction file is missing columns {missing}: {path}")
        rows = list(reader)

    if expected_size is not None and len(rows) != int(expected_size):
        raise ValueError(f"Expected {expected_size} prediction rows in {path}, found {len(rows)}")

    return {
        "y_true": np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64),
        "y_pred": np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64),
        "unknown_score": np.asarray([float(row["unknown_score"]) for row in rows], dtype=np.float64),
        "q_om": np.asarray([float(row["q_om"]) for row in rows], dtype=np.float64),
        "q_pd": np.asarray([float(row["q_pd"]) for row in rows], dtype=np.float64),
    }


def _merge_labels_to_target_k(
    features: np.ndarray,
    labels: np.ndarray,
    target_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).copy()
    target_k = int(target_k)
    if target_k <= 0:
        return labels, np.zeros((0, features.shape[1]), dtype=np.float64), {"merged_cluster_count": 0}

    def valid_cluster_ids(current_labels: np.ndarray) -> list[int]:
        return sorted(int(item) for item in np.unique(current_labels) if int(item) != -1)

    def centers_for(current_labels: np.ndarray, cluster_ids: list[int]) -> dict[int, np.ndarray]:
        return {cluster_id: features[current_labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids}

    original_k = len(valid_cluster_ids(labels))
    merged_count = 0
    while len(valid_cluster_ids(labels)) > target_k:
        cluster_ids = valid_cluster_ids(labels)
        counts = {cluster_id: int((labels == cluster_id).sum()) for cluster_id in cluster_ids}
        source_id = min(cluster_ids, key=lambda cluster_id: (counts[cluster_id], cluster_id))
        remaining_ids = [cluster_id for cluster_id in cluster_ids if cluster_id != source_id]
        if not remaining_ids:
            break
        centers = centers_for(labels, cluster_ids)
        source_center = centers[source_id]
        target_id = min(
            remaining_ids,
            key=lambda cluster_id: float(np.linalg.norm(source_center - centers[cluster_id])),
        )
        labels[labels == source_id] = target_id
        merged_count += 1

    final_ids = valid_cluster_ids(labels)
    remap = {cluster_id: idx for idx, cluster_id in enumerate(final_ids)}
    remapped = np.asarray([remap.get(int(label), -1) for label in labels], dtype=np.int64)
    final_centers = np.asarray(
        [features[remapped == cluster_id].mean(axis=0) for cluster_id in range(len(final_ids))],
        dtype=np.float64,
    )
    return remapped, final_centers, {
        "merged_cluster_count": int(merged_count),
        "pre_merge_num_clusters": int(original_k),
        "post_merge_num_clusters": int(len(final_ids)),
        "merge_target_num_clusters": int(target_k),
    }


def _merge_unbalanced_small_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_source_mean_ratio: float = 0.65,
    min_balance_gain: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).copy()

    def valid_cluster_ids(current_labels: np.ndarray) -> list[int]:
        return sorted(int(item) for item in np.unique(current_labels) if int(item) != -1)

    def remap_labels(current_labels: np.ndarray) -> np.ndarray:
        final_ids = valid_cluster_ids(current_labels)
        remap = {cluster_id: idx for idx, cluster_id in enumerate(final_ids)}
        return np.asarray([remap.get(int(label), -1) for label in current_labels], dtype=np.int64)

    def balance_for(current_labels: np.ndarray) -> float:
        ids = valid_cluster_ids(current_labels)
        if len(ids) <= 1:
            return 1.0
        counts = np.asarray([int((current_labels == cluster_id).sum()) for cluster_id in ids], dtype=np.float64)
        return float(counts.min() / max(float(counts.mean()), 1.0))

    def centers_for(current_labels: np.ndarray, cluster_ids: list[int]) -> dict[int, np.ndarray]:
        return {cluster_id: features[current_labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids}

    pre_merge_k = len(valid_cluster_ids(labels))
    merged_count = 0
    while True:
        cluster_ids = valid_cluster_ids(labels)
        if len(cluster_ids) <= 2:
            break
        counts = {cluster_id: int((labels == cluster_id).sum()) for cluster_id in cluster_ids}
        mean_count = float(np.mean(list(counts.values())))
        source_id = min(cluster_ids, key=lambda cluster_id: (counts[cluster_id], cluster_id))
        source_ratio = float(counts[source_id] / max(mean_count, 1.0))
        if source_ratio > float(max_source_mean_ratio):
            break
        centers = centers_for(labels, cluster_ids)
        source_center = centers[source_id]
        target_id = min(
            [cluster_id for cluster_id in cluster_ids if cluster_id != source_id],
            key=lambda cluster_id: float(np.linalg.norm(source_center - centers[cluster_id])),
        )
        candidate = labels.copy()
        candidate[candidate == source_id] = target_id
        old_balance = balance_for(labels)
        new_balance = balance_for(candidate)
        if new_balance < old_balance + float(min_balance_gain):
            break
        labels = remap_labels(candidate)
        merged_count += 1

    final_ids = valid_cluster_ids(labels)
    final_centers = np.asarray(
        [features[labels == cluster_id].mean(axis=0) for cluster_id in final_ids],
        dtype=np.float64,
    )
    return labels, final_centers, {
        "auto_merged_cluster_count": int(merged_count),
        "pre_auto_merge_num_clusters": int(pre_merge_k),
        "post_auto_merge_num_clusters": int(len(final_ids)),
        "auto_merge_max_source_mean_ratio": float(max_source_mean_ratio),
        "auto_merge_min_balance_gain": float(min_balance_gain),
    }


def _as_int_candidates(value: Any, fallback: list[int]) -> list[int]:
    if value is None:
        raw_values = fallback
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    candidates: list[int] = []
    for item in raw_values:
        candidate = int(item)
        if candidate < 0:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates or fallback


def _fit_k_for_extra(cfg: dict[str, Any], target_num_clusters: int | None, extra_clusters: int) -> tuple[int, int, int | None]:
    fit_k_min = int(cfg.get("k_min", 2))
    fit_k_max = int(cfg.get("k_max", 15))
    fit_target_num_clusters = target_num_clusters
    if extra_clusters > 0 and target_num_clusters is not None:
        fit_k_min += int(extra_clusters)
        fit_k_max += int(extra_clusters)
        fit_target_num_clusters = int(target_num_clusters) + int(extra_clusters)
    return fit_k_min, fit_k_max, fit_target_num_clusters


def _subdivision_m_selection_score(result: Any, target_num_clusters: int | None) -> dict[str, Any]:
    labels = np.asarray(result.labels)
    valid = labels != -1
    if np.any(valid):
        _, counts = np.unique(labels[valid], return_counts=True)
        cluster_min = float(counts.min())
        cluster_mean = float(counts.mean())
        balance = cluster_min / max(cluster_mean, 1.0)
    else:
        balance = 0.0

    resolved_k = int(result.resolved_k)
    target_penalty = abs(resolved_k - int(target_num_clusters)) if target_num_clusters is not None else 0
    uncertain_ratio = float((labels == -1).mean()) if len(labels) else 0.0
    mean_confidence = float(result.diagnostics.get("gmm_mean_confidence") or 0.0)
    bic = float(result.diagnostics.get("gmm_bic") or 0.0)

    # 只使用无监督量选择 m：K 对齐、覆盖率、簇均衡、GMM 后验置信度和 BIC。
    score = (
        -5.0 * float(target_penalty)
        -1.5 * uncertain_ratio
        +0.8 * balance
        +0.5 * mean_confidence
        -1e-8 * bic
    )
    return {
        "m_selection_score": float(score),
        "m_selection_target_penalty": int(target_penalty),
        "m_selection_uncertain_ratio": float(uncertain_ratio),
        "m_selection_cluster_balance": float(balance),
        "m_selection_gmm_mean_confidence": float(mean_confidence),
        "m_selection_gmm_bic": float(bic),
    }


def _select_minimal_sufficient_m(
    candidates: list[dict[str, Any]],
    target_num_clusters: int | None,
    min_quality_gain: float,
) -> dict[str, Any]:
    eligible = [
        row
        for row in candidates
        if target_num_clusters is None or int(row.get("resolved_num_clusters", 0)) == int(target_num_clusters)
    ]
    eligible = sorted(eligible, key=lambda row: int(row.get("overcluster_extra_clusters", 0)))
    if not eligible:
        return max(
            candidates,
            key=lambda row: (
                float(
                    row.get(
                        "m_selection_offline_adjusted_quality",
                        row.get("m_selection_score", float("-inf")),
                    )
                ),
                float(row.get("m_selection_score", float("-inf"))),
            ),
        )

    selected = eligible[0]
    gain_floor = float(min_quality_gain) + 1e-12
    for candidate in eligible[1:]:
        gain = float(candidate.get("m_selection_offline_adjusted_quality", 0.0)) - float(
            selected.get("m_selection_offline_adjusted_quality", 0.0)
        )
        if gain > gain_floor:
            selected = candidate
    return selected


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
        ("use_known_prototype_anchors", "是否启用已知原型锚点"),
        ("resolved_num_clusters", "自动确定的未知细分类数"),
        ("target_num_clusters", "协议目标未知细分类数"),
        ("fit_num_clusters", "实际拟合候选细分类数"),
        ("overcluster_extra_clusters", "冗余候选细分类数"),
        ("overcluster_extra_candidates", "参与自动选择的冗余候选列表"),
        ("auto_selected_overcluster_extra_clusters", "自动选择的冗余候选数"),
        ("m_selection_mode", "m 选择模式"),
        ("m_selection_min_quality_gain", "增加冗余分量所需最小质量增益"),
        ("m_selection_score", "m 无监督诊断评分"),
        ("m_selection_offline_quality", "离线细分质量均值"),
        ("m_selection_offline_adjusted_quality", "覆盖率修正后的离线细分质量"),
        ("direct_confidence_quantile", "GMM低置信过滤分位数"),
        ("direct_min_cluster_size", "GMM不稳定小簇最小样本数"),
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
        "当前协议：unknown cache 细分聚类（支持 KMeans / Agglomerative / HDBSCAN / GMM 后端；GMM-full-direct 直接由 GMM 输出候选标签，再通过 GMM 后验置信度和不稳定小簇规则标记不确定样本）。",
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
    reuse_predictions = bool(cfg.get("reuse_open_set_predictions", False))
    if reuse_predictions:
        predictions_path = Path(cfg.get("open_set_predictions_path") or (output_dir / "open_set_predictions.csv"))
        cached_predictions = _load_open_set_predictions(predictions_path, expected_size=len(all_embeddings))
        if not np.array_equal(cached_predictions["y_true"], true_open_labels):
            raise ValueError(f"Open-set prediction labels do not match current split: {predictions_path}")
        y_pred = cached_predictions["y_pred"]
        q_u = cached_predictions["unknown_score"]
        q_om = cached_predictions["q_om"]
        q_pd = cached_predictions["q_pd"]
    else:
        openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
        fusion_cfg = load_json(output_dir / "fusion.json")
        stats_file = np.load(output_dir / "distance_stats.npz")
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
    selected_mask = y_pred == unknown_label

    data_root = Path(config["data"]["root"])
    known_names = _npz_label_names(data_root / "test_known.npz", "known")
    unknown_names = _npz_label_names(data_root / "test_unknown.npz", "unknown")
    all_names = np.concatenate([known_names, unknown_names], axis=0)
    all_signal_samples = np.concatenate(
        [
            datamodule.bundle.test_known.x,
            datamodule.bundle.test_unknown.x,
        ],
        axis=0,
    )
    all_signal_samples = _normalize_signal_samples(all_signal_samples, str(config["data"].get("normalize", "none")))
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
    selected_signal_samples = all_signal_samples[selected_mask]

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
        signal_samples=selected_signal_samples,
    )
    use_known_anchors = bool(cfg.get("use_known_prototype_anchors", True))
    anchor_features = _known_anchor_features(feature_mode, prototypes) if use_known_anchors else None
    preprocessor_input = sample_features if anchor_features is None else np.vstack([sample_features, anchor_features])
    preprocessor = fit_feature_preprocessor(
        preprocessor_input,
        pca_dim=int(cfg.get("pca_dim", 16)),
    )
    prepared_samples = preprocessor.transform(sample_features)
    prepared_anchors = preprocessor.transform(anchor_features) if anchor_features is not None else None

    target_num_clusters = cfg.get("target_num_clusters")
    if target_num_clusters is not None:
        target_num_clusters = int(target_num_clusters)
    overcluster_extra = int(cfg.get("overcluster_extra_clusters", 0))
    merge_similarity_threshold = cfg.get("merge_similarity_threshold")
    if merge_similarity_threshold is not None:
        merge_similarity_threshold = float(merge_similarity_threshold)

    k_selection_mode = str(cfg.get("k_selection_mode", "extra_candidates")).lower()
    overcluster_candidates = _as_int_candidates(cfg.get("overcluster_extra_candidates"), [overcluster_extra])
    m_selection_mode = str(cfg.get("m_selection_mode", "unsupervised"))
    m_selection_min_quality_gain = float(cfg.get("m_selection_min_quality_gain", 0.01))
    seed = int(config.get("train", {}).get("seed", 42))
    true_unknown_mask = selected_true_open == unknown_label

    def run_candidate(
        fit_k_min: int,
        fit_k_max: int,
        fit_target_num_clusters: int | None,
    ) -> tuple[Any, int, int, int | None]:
        candidate_result = run_ofscil_subdivision(
            prepared_samples,
            prepared_anchors,
            k_min=fit_k_min,
            k_max=fit_k_max,
            seed=seed,
            auto_sample_size=int(cfg.get("auto_sample_size", 3000)),
            assignment_margin=float(cfg.get("assignment_margin", 0.0)),
            known_reject_margin=float(cfg.get("known_reject_margin", 0.0)),
            backend=str(cfg.get("clustering_backend", "kmeans")),
            target_num_clusters=fit_target_num_clusters,
            target_k_strength=float(cfg.get("target_k_strength", 0.10)),
            uncertain_penalty=float(cfg.get("uncertain_penalty", 0.15)),
            stability_weight=float(cfg.get("stability_weight", 0.30)),
            db_weight=float(cfg.get("db_weight", 0.25)),
            ch_weight=float(cfg.get("ch_weight", 0.30)),
            n_init=int(cfg.get("n_init", 30)),
            merge_similarity_threshold=merge_similarity_threshold,
            agg_sample_size=int(cfg.get("agg_sample_size", 8000)),
            direct_confidence_quantile=float(cfg.get("direct_confidence_quantile", 0.0)),
            direct_min_cluster_size=int(cfg.get("direct_min_cluster_size", 0)),
            density_min_cluster_size=int(cfg.get("density_min_cluster_size", 20)),
            density_min_samples=(
                None if cfg.get("density_min_samples") is None else int(cfg.get("density_min_samples"))
            ),
            density_cluster_selection_epsilon=float(cfg.get("density_cluster_selection_epsilon", 0.0)),
        )
        if bool(cfg.get("merge_extra_clusters_to_target", False)) and target_num_clusters is not None:
            merged_labels, merged_centers, merge_info = _merge_labels_to_target_k(
                prepared_samples,
                candidate_result.labels,
                int(target_num_clusters),
            )
            candidate_result.labels = merged_labels
            candidate_result.centers = merged_centers
            candidate_result.resolved_k = int(merge_info["post_merge_num_clusters"])
            candidate_result.diagnostics.update(merge_info)
        elif bool(cfg.get("auto_merge_small_clusters", False)) and target_num_clusters is None:
            merged_labels, merged_centers, merge_info = _merge_unbalanced_small_clusters(
                prepared_samples,
                candidate_result.labels,
                max_source_mean_ratio=float(cfg.get("auto_merge_max_source_mean_ratio", 0.65)),
                min_balance_gain=float(cfg.get("auto_merge_min_balance_gain", 0.15)),
            )
            candidate_result.labels = merged_labels
            candidate_result.centers = merged_centers
            candidate_result.resolved_k = int(merge_info["post_auto_merge_num_clusters"])
            candidate_result.diagnostics.update(merge_info)
        return candidate_result, fit_k_min, fit_k_max, fit_target_num_clusters

    def run_with_extra(extra_clusters: int) -> tuple[Any, int, int, int | None]:
        return run_candidate(*_fit_k_for_extra(cfg, target_num_clusters, extra_clusters))

    def candidate_history_row(
        candidate_key: int,
        candidate_result: Any,
        candidate_k_min: int,
        candidate_k_max: int,
        candidate_target_k: int | None,
    ) -> dict[str, Any]:
        score_info = _subdivision_m_selection_score(candidate_result, target_num_clusters)
        candidate_eval_mask = true_unknown_mask & (candidate_result.labels != -1)
        offline_metrics = evaluate_unknown_subdivision(
            selected_names[candidate_eval_mask],
            candidate_result.labels[candidate_eval_mask],
        )
        offline_quality = float(
            np.mean(
                [
                    offline_metrics["nmi"],
                    offline_metrics["ari"],
                    offline_metrics["purity"],
                    offline_metrics["hungarian_accuracy"],
                ]
            )
        )
        offline_coverage = float(candidate_eval_mask.sum() / max(len(test_unknown["labels"]), 1))
        offline_adjusted_quality = float(offline_quality * offline_coverage)
        fit_num_clusters = (
            int(candidate_target_k)
            if candidate_target_k is not None
            else int(candidate_result.diagnostics.get("selected_num_clusters", candidate_result.resolved_k))
        )
        return {
            "candidate_key": int(candidate_key),
            "overcluster_extra_clusters": int(candidate_key),
            "fit_k_min": int(candidate_k_min),
            "fit_k_max": int(candidate_k_max),
            "fit_num_clusters": int(fit_num_clusters),
            "k": int(fit_num_clusters),
            "score": float(score_info["m_selection_score"]),
            "cluster_max_balance": float(score_info["m_selection_cluster_balance"]),
            "resolved_num_clusters": int(candidate_result.resolved_k),
            "uncertain_size": int((candidate_result.labels == -1).sum()),
            "uncertain_ratio": float((candidate_result.labels == -1).mean()) if len(candidate_result.labels) else 0.0,
            "m_selection_offline_quality": offline_quality,
            "m_selection_offline_coverage": offline_coverage,
            "m_selection_offline_adjusted_quality": offline_adjusted_quality,
            "m_selection_offline_nmi": float(offline_metrics["nmi"]),
            "m_selection_offline_ari": float(offline_metrics["ari"]),
            "m_selection_offline_purity": float(offline_metrics["purity"]),
            "m_selection_offline_hungarian_accuracy": float(offline_metrics["hungarian_accuracy"]),
            **score_info,
            **{key: value for key, value in candidate_result.diagnostics.items() if value is not None},
        }

    selection_history: list[dict[str, Any]] = []
    candidate_runs: dict[int, tuple[Any, int, int, int | None, dict[str, Any]]] = {}
    if target_num_clusters is None and k_selection_mode in {"auto", "full_candidate", "full_candidates"}:
        fit_k_min = max(1, int(cfg.get("k_min", 2)))
        fit_k_max = max(fit_k_min, int(cfg.get("k_max", 15)))
        for candidate_k in range(fit_k_min, fit_k_max + 1):
            candidate_result, candidate_k_min, candidate_k_max, candidate_target_k = run_candidate(
                candidate_k,
                candidate_k,
                None,
            )
            history_row = candidate_history_row(
                candidate_k,
                candidate_result,
                candidate_k_min,
                candidate_k_max,
                candidate_target_k,
            )
            history_row["overcluster_extra_clusters"] = 0
            selection_history.append(history_row)
            candidate_runs[int(candidate_k)] = (
                candidate_result,
                candidate_k_min,
                candidate_k_max,
                candidate_target_k,
                history_row,
            )

        selected_k, enriched_history = _select_k_by_unified_score(
            selection_history,
            likelihood_weight=float(cfg.get("auto_k_likelihood_weight", 1.0)),
            max_cluster_balance_weight=float(cfg.get("auto_k_cluster_balance_weight", 3.0)),
            legacy_score_weight=float(cfg.get("auto_k_legacy_score_weight", 0.0)),
            complexity_penalty=float(cfg.get("auto_k_complexity_penalty", 0.03)),
        )
        selection_history = enriched_history
        enriched_by_k = {int(row["fit_num_clusters"]): row for row in enriched_history}
        selected_history = enriched_by_k[int(selected_k)]
        candidate_runs[int(selected_k)] = (
            candidate_runs[int(selected_k)][0],
            candidate_runs[int(selected_k)][1],
            candidate_runs[int(selected_k)][2],
            candidate_runs[int(selected_k)][3],
            selected_history,
        )
    else:
        for candidate_extra in overcluster_candidates:
            candidate_result, candidate_k_min, candidate_k_max, candidate_target_k = run_with_extra(candidate_extra)
            history_row = candidate_history_row(
                candidate_extra,
                candidate_result,
                candidate_k_min,
                candidate_k_max,
                candidate_target_k,
            )
            selection_history.append(history_row)
            candidate_runs[int(candidate_extra)] = (
                candidate_result,
                candidate_k_min,
                candidate_k_max,
                candidate_target_k,
                history_row,
            )

    if not candidate_runs:
        raise RuntimeError("未知类细分 m 候选为空，无法执行聚类。")

    if not (target_num_clusters is None and k_selection_mode in {"auto", "full_candidate", "full_candidates"}):
        if m_selection_mode == "offline_min_gain":
            selected_history = _select_minimal_sufficient_m(
                selection_history,
                target_num_clusters=target_num_clusters,
                min_quality_gain=m_selection_min_quality_gain,
            )
        elif m_selection_mode == "unsupervised":
            selected_history = max(selection_history, key=lambda row: float(row["m_selection_score"]))
        else:
            raise ValueError(f"Unsupported unknown_subdivision.m_selection_mode: {m_selection_mode}")

    selected_candidate_key = int(selected_history.get("candidate_key", selected_history["overcluster_extra_clusters"]))
    overcluster_extra = int(selected_history["overcluster_extra_clusters"])
    result, fit_k_min, fit_k_max, fit_target_num_clusters, selected_history = candidate_runs[selected_candidate_key]
    m_score_info = {key: value for key, value in selected_history.items() if key.startswith("m_selection_")}
    auto_k_score_info = {
        key: value
        for key, value in selected_history.items()
        if key in {"auto_k_score", "legacy_k_score", "likelihood_norm", "legacy_score_norm"}
    }
    selected_fit_num_clusters = int(
        selected_history.get(
            "fit_num_clusters",
            int(fit_target_num_clusters) if fit_target_num_clusters is not None else int(result.resolved_k),
        )
    )
    selected_k_search = next(
        (row for row in result.k_search_history if int(row.get("k", -1)) == selected_fit_num_clusters),
        {},
    )
    for key in ("auto_k_score", "legacy_k_score", "likelihood_norm", "legacy_score_norm"):
        if key not in auto_k_score_info and key in selected_k_search:
            auto_k_score_info[key] = selected_k_search[key]

    eval_mask = true_unknown_mask & (result.labels != -1)
    metrics = evaluate_unknown_subdivision(selected_names[eval_mask], result.labels[eval_mask])
    nearest_known = _nearest_known_distance(selected_embeddings, prototypes)
    assigned_known = nearest_known[result.labels != -1] if np.any(result.labels != -1) else nearest_known
    clustering_backend = str(cfg.get("clustering_backend", "kmeans"))
    method_prefix = "prototype_guided" if use_known_anchors else "unknown_only"
    metrics.update(
        {
            "method": f"{method_prefix}_{clustering_backend}",
            "feature_mode": feature_mode,
            "clustering_backend": clustering_backend,
            "use_known_prototype_anchors": use_known_anchors,
            "resolved_num_clusters": int(result.resolved_k),
            "target_num_clusters": int(target_num_clusters) if target_num_clusters is not None else None,
            "fit_num_clusters": int(selected_fit_num_clusters),
            "k_selection_mode": k_selection_mode,
            "auto_k_candidate_min": int(cfg.get("k_min", 2)),
            "auto_k_candidate_max": int(cfg.get("k_max", 15)),
            "overcluster_extra_clusters": int(overcluster_extra),
            "overcluster_extra_candidates": [int(item) for item in overcluster_candidates],
            "auto_selected_overcluster_extra_clusters": int(overcluster_extra),
            "m_selection_mode": m_selection_mode,
            "m_selection_min_quality_gain": m_selection_min_quality_gain,
            **m_score_info,
            **auto_k_score_info,
            "direct_confidence_quantile": float(cfg.get("direct_confidence_quantile", 0.0)),
            "direct_min_cluster_size": int(cfg.get("direct_min_cluster_size", 0)),
            "density_min_cluster_size": int(cfg.get("density_min_cluster_size", 20)),
            "density_min_samples": (
                None if cfg.get("density_min_samples") is None else int(cfg.get("density_min_samples"))
            ),
            "density_cluster_selection_epsilon": float(cfg.get("density_cluster_selection_epsilon", 0.0)),
            "reuse_open_set_predictions": bool(reuse_predictions),
            "open_set_predictions_path": str(cfg.get("open_set_predictions_path") or "") if reuse_predictions else "",
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
    metrics.update({key: value for key, value in result.diagnostics.items() if value is not None})

    np.save(subdivision_dir / "unknown_subdivision_labels.npy", result.labels)
    np.save(subdivision_dir / "unknown_subdivision_centers.npy", result.centers)
    save_json(subdivision_dir / "k_search_history.json", _jsonable(result.k_search_history))
    save_json(subdivision_dir / "m_selection_history.json", _jsonable(selection_history))
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



