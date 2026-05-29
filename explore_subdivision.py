"""探索未知类细分的不同配置组合，离线评估 NMI / ARI / 纯度。

使用方式：

- `python explore_subdivision.py oracle`：在 Oracle 上对比多种 feature_mode + backend + target_k。
- `python explore_subdivision.py wisig`：在 WiSig 上对比多种配置。

脚本会读取对应输出目录里现成的 `openmax.pkl`、`fusion.json`、`distance_stats.npz`，因此只需要
确保已经跑过对应的开放集拒识流程。结果以 JSON 形式写入项目根目录的 `_explore_subdivision_<数据集>.json`。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from functions.common.io import load_json, load_pickle
from functions.data.data_build import build_data_module
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from functions.methods.fusion import apply_unknown_rejection, fuse_unknown_score, prototype_distance_unknown_score
from functions.methods.prototype_utils import activations_from_distances, predict_with_prototypes, squared_euclidean
from functions.methods.unknown_subdivision import (
    evaluate_unknown_subdivision,
    fit_feature_preprocessor,
    run_ofscil_subdivision,
)
from functions.model.closed_set import ClosedSetTrainer
from functions.pipeline import checkpoint_path_for
from functions.subdivision_pipeline import build_cluster_features, _known_anchor_features
from settings import default_oracle_config, default_wisig_config


def _npz_label_names(path: Path, fallback_prefix: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if "label_name" in payload.files:
        return payload["label_name"].astype(str)
    labels = payload["y"].astype(str)
    return np.asarray([f"{fallback_prefix}_{label}" for label in labels], dtype=str)


def _prepare_unknown_cache(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["project"]["output_dir"])
    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(checkpoint_path_for(config, None))

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
    return {
        "selected_mask": selected_mask,
        "selected_embeddings": all_embeddings[selected_mask],
        "selected_distances": distances[selected_mask],
        "selected_known_pred": known_pred[selected_mask],
        "selected_q_om": q_om[selected_mask],
        "selected_q_pd": q_pd[selected_mask],
        "selected_q_u": q_u[selected_mask],
        "selected_true_open": true_open_labels[selected_mask],
        "selected_names": all_names[selected_mask],
        "prototypes": prototypes,
        "unknown_label": unknown_label,
        "num_total_test_unknown": len(test_unknown["labels"]),
    }


def _run_clustering_for_config(
    cache: dict[str, Any],
    feature_mode: str,
    pca_dim: int,
    backend: str,
    target_k: int,
    seed: int,
) -> dict[str, Any]:
    sample_features = build_cluster_features(
        feature_mode,
        cache["selected_embeddings"],
        cache["selected_distances"],
        cache["selected_q_om"],
        cache["selected_q_pd"],
        cache["selected_q_u"],
        cache["selected_known_pred"],
        cache["prototypes"],
    )
    anchor_features = _known_anchor_features(feature_mode, cache["prototypes"])
    preprocessor = fit_feature_preprocessor(
        np.vstack([sample_features, anchor_features]),
        pca_dim=int(pca_dim),
    )
    prepared_samples = preprocessor.transform(sample_features)
    prepared_anchors = preprocessor.transform(anchor_features)

    result = run_ofscil_subdivision(
        prepared_samples,
        prepared_anchors,
        k_min=int(target_k),
        k_max=int(target_k),
        seed=int(seed),
        auto_sample_size=3000,
        assignment_margin=0.0,
        known_reject_margin=0.0,
        backend=backend,
        target_num_clusters=int(target_k),
        target_k_strength=1.0,
        n_init=30,
        agg_sample_size=8000,
    )

    unknown_label = cache["unknown_label"]
    selected_true_open = cache["selected_true_open"]
    true_unknown_mask = selected_true_open == unknown_label
    eval_mask = true_unknown_mask & (result.labels != -1)
    metrics = evaluate_unknown_subdivision(cache["selected_names"][eval_mask], result.labels[eval_mask])
    metrics["uncertain_ratio"] = float((result.labels == -1).mean())
    metrics["resolved_k"] = int(result.resolved_k)
    return metrics


def explore(dataset: str) -> None:
    root = Path(__file__).resolve().parent
    if dataset == "oracle":
        config = default_oracle_config(root)
        target_ks = [4, 5, 6, 7, 8]
    elif dataset == "wisig":
        config = default_wisig_config(root)
        target_ks = [8, 10, 12, 13, 14]
    else:
        raise ValueError(dataset)

    print(f"准备 {dataset} 的 unknown cache ...")
    cache = _prepare_unknown_cache(config)
    print(
        f"unknown cache size = {int(cache['selected_mask'].sum())}, "
        f"真实未知样本数 = {int((cache['selected_true_open'] == cache['unknown_label']).sum())}, "
        f"已知噪声 = {int((cache['selected_true_open'] != cache['unknown_label']).sum())}"
    )

    feature_modes = ["embedding", "embedding_distance", "score_distance"]
    backends = ["kmeans", "agglomerative_cosine", "gmm"]
    pca_dims = {
        "embedding": 32,
        "embedding_distance": 32,
        "score_distance": 13,
        "prototype_residual": 16,
        "residual_distance": 16,
    }

    rows: list[dict[str, Any]] = []
    for feature_mode, backend, target_k in itertools.product(feature_modes, backends, target_ks):
        pca_dim = pca_dims[feature_mode]
        try:
            metrics = _run_clustering_for_config(
                cache,
                feature_mode=feature_mode,
                pca_dim=pca_dim,
                backend=backend,
                target_k=target_k,
                seed=int(config.get("train", {}).get("seed", 42)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"feature={feature_mode}, backend={backend}, k={target_k}: ERROR {exc}")
            continue
        row = {
            "feature_mode": feature_mode,
            "backend": backend,
            "target_k": target_k,
            **metrics,
        }
        rows.append(row)
        print(
            f"feature={feature_mode:<20s} backend={backend:<22s} k={target_k:>2d} "
            f"nmi={metrics['nmi']:.4f} ari={metrics['ari']:.4f} "
            f"purity={metrics['purity']:.4f} hung_acc={metrics['hungarian_accuracy']:.4f} "
            f"uncertain={metrics['uncertain_ratio']:.3f}"
        )

    rows.sort(key=lambda r: r["nmi"], reverse=True)
    print("\nTop 10 by NMI:")
    for row in rows[:10]:
        print(
            f"feature={row['feature_mode']:<20s} backend={row['backend']:<22s} k={row['target_k']:>2d} "
            f"nmi={row['nmi']:.4f} ari={row['ari']:.4f} purity={row['purity']:.4f}"
        )

    out_path = Path(__file__).resolve().parent / f"_explore_subdivision_{dataset}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"探索结果已写入 {out_path}")


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    explore(dataset)
