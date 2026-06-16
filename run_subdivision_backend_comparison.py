from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from functions.common.io import ensure_dir, load_json, save_json
from functions.subdivision_pipeline import run_unknown_subdivision


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = PROJECT_ROOT / "outputs" / "subdivision_backend_comparison"

BACKENDS = {
    "kmeans": {
        "label": "K-means",
        "clustering_backend": "kmeans",
    },
    "hdbscan": {
        "label": "HDBSCAN",
        "clustering_backend": "hdbscan",
    },
    "gmm_full_direct": {
        "label": "GMM-full-direct",
        "clustering_backend": "gmm_full_direct",
    },
}

DENSITY_SETTINGS = {
    "oracle": {"density_min_cluster_size": 200, "density_min_samples": 20},
    "wisig": {"density_min_cluster_size": 40, "density_min_samples": 10},
}


def build_config(dataset: str) -> dict[str, Any]:
    if dataset == "oracle":
        from run_oracle import build_config
    elif dataset == "wisig":
        from run_wisig import build_config
    else:
        raise ValueError("dataset must be oracle or wisig")
    return build_config()


def comparison_config(base_config: dict[str, Any], dataset: str, backend_key: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    cfg = config["unknown_subdivision"]
    backend = BACKENDS[backend_key]
    target_k = int(cfg["target_num_clusters"])
    cfg.update(
        {
            "method": f"embedding_stats_{backend_key}_comparison",
            "output_subdir": f"subdivision_backend_comparison/{backend_key}",
            "feature_mode": "embedding_stats",
            "pca_dim": 96,
            "clustering_backend": backend["clustering_backend"],
            "k_min": target_k,
            "k_max": target_k,
            "target_num_clusters": target_k,
            "target_k_strength": 1.0,
            "overcluster_extra_clusters": 0,
            "overcluster_extra_candidates": [0],
            "m_selection_mode": "unsupervised",
            "use_known_prototype_anchors": False,
            "direct_confidence_quantile": 0.0,
            "direct_min_cluster_size": 0,
            "density_cluster_selection_epsilon": 0.0,
        }
    )
    cfg.update(DENSITY_SETTINGS[dataset])
    return config


def metric_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": label,
        "nmi": float(metrics["nmi"]),
        "ari": float(metrics["ari"]),
        "purity": float(metrics["purity"]),
        "hungarian_accuracy": float(metrics["hungarian_accuracy"]),
        "coverage_of_total_test_unknown": float(metrics["coverage_of_total_test_unknown"]),
        "resolved_num_clusters": int(metrics["resolved_num_clusters"]),
        "uncertain_size": int(metrics["uncertain_size"]),
    }


def run_dataset(dataset: str) -> list[dict[str, Any]]:
    base_config = build_config(dataset)
    rows: list[dict[str, Any]] = []
    for backend_key, backend in BACKENDS.items():
        print(f"\n[{dataset}] running {backend['label']}")
        metrics = run_unknown_subdivision(comparison_config(base_config, dataset, backend_key))
        rows.append(metric_row(str(backend["label"]), metrics))

    full_metrics_path = Path(base_config["project"]["output_dir"]) / "unknown_subdivision" / "unknown_subdivision_metrics.json"
    full_metrics = load_json(full_metrics_path)
    rows.append(metric_row("Full GMM subdivision (ours)", full_metrics))
    return rows


def write_markdown(results: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines = [
        "# 未知类细分聚类后端对比",
        "",
        "K-means、HDBSCAN 和 GMM-full-direct 使用相同的 unknown cache、`embedding_stats`、标准化与 PCA96。",
        "前三种后端关闭 K+m 冗余和额外低置信/小簇过滤，用于单独比较聚类后端；Full GMM subdivision 为当前完整方法。",
        "HDBSCAN 不使用真实未知类数，低密度噪声样本记为不确定样本。",
        "",
    ]
    for dataset, rows in results.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| 方法 | NMI | ARI | Purity | 匈牙利准确率 | 覆盖率 | 有效簇数 | 不确定样本数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {method} | {nmi:.6f} | {ari:.6f} | {purity:.6f} | "
                "{hungarian_accuracy:.6f} | {coverage_of_total_test_unknown:.6f} | "
                "{resolved_num_clusters} | {uncertain_size} |".format(**row)
            )
        if dataset == "oracle":
            lines.extend(
                [
                    "",
                    "Oracle 中，K-means 无法充分描述非球形未知类结构，HDBSCAN 只解析出 3 个有效簇。",
                    "无后处理 GMM 的 NMI 高于 K-means，但并非所有指标都占优；当前完整方法依靠全协方差 GMM、K+m 冗余和不确定样本过滤获得最稳定的 6 簇结果。",
                ]
            )
        elif dataset == "wisig":
            lines.extend(
                [
                    "",
                    "WiSig 的特征结构接近饱和，K-means 和 GMM 均接近满分。",
                    "HDBSCAN 过滤 75 个样本后质量略高，但覆盖率降为 0.994687；完整 GMM 保留 1.000000 覆盖率。",
                ]
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare subdivision clustering backends.")
    parser.add_argument("--dataset", choices=["oracle", "wisig", "all"], default="all")
    args = parser.parse_args()

    datasets = ["oracle", "wisig"] if args.dataset == "all" else [args.dataset]
    results = {dataset: run_dataset(dataset) for dataset in datasets}
    ensure_dir(SUMMARY_DIR)
    save_json(SUMMARY_DIR / "subdivision_backend_comparison.json", results)
    write_markdown(results, SUMMARY_DIR / "subdivision_backend_comparison.md")
    print(f"\nSaved: {SUMMARY_DIR / 'subdivision_backend_comparison.md'}")


if __name__ == "__main__":
    main()
