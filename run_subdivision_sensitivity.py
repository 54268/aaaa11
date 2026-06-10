from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from functions.common.io import ensure_dir, save_json
from functions.subdivision_pipeline import _select_minimal_sufficient_m, run_unknown_subdivision
from run_oracle import CHECKPOINT_PATH as ORACLE_CHECKPOINT_PATH
from run_oracle import build_config as build_oracle_config
from run_wisig import CHECKPOINT_PATH as WISIG_CHECKPOINT_PATH
from run_wisig import build_config as build_wisig_config


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "subdivision_sensitivity"
M_CANDIDATES = [0, 1, 2, 3]


def _selection_score(metrics: dict) -> float:
    """无监督选择分数，不使用 NMI/ARI/Purity/Hungarian 等真实未知细分类标签指标。"""
    target_k = metrics.get("target_num_clusters")
    resolved_k = int(metrics.get("resolved_num_clusters") or 0)
    target_penalty = abs(resolved_k - int(target_k)) if target_k is not None else 0
    uncertain_ratio = float(metrics.get("uncertain_ratio") or 0.0)
    cluster_min = float(metrics.get("cluster_size_min") or 0.0)
    cluster_mean = float(metrics.get("cluster_size_mean") or 1.0)
    balance = cluster_min / max(cluster_mean, 1.0)
    mean_conf = float(metrics.get("gmm_mean_confidence") or 0.0)
    bic = float(metrics.get("gmm_bic") or 0.0)

    # BIC 数值量级较大，只用于同一数据集候选之间的轻量归一化辅助项，主约束仍是 K 对齐、覆盖和簇均衡。
    bic_term = -bic * 1e-8
    return float(
        -5.0 * target_penalty
        -1.5 * uncertain_ratio
        +0.8 * balance
        +0.5 * mean_conf
        +bic_term
    )


def _offline_adjusted_quality(metrics: dict) -> float:
    quality = sum(
        float(metrics.get(key) or 0.0)
        for key in ["nmi", "ari", "purity", "hungarian_accuracy"]
    ) / 4.0
    coverage = float(metrics.get("coverage_of_total_test_unknown") or 0.0)
    return float(quality * coverage)


def _run_dataset(name: str, build_config, checkpoint_path: str | Path | None) -> list[dict]:
    rows = []
    for m in M_CANDIDATES:
        config = deepcopy(build_config())
        config["unknown_subdivision"]["overcluster_extra_clusters"] = int(m)
        config["unknown_subdivision"]["overcluster_extra_candidates"] = [int(m)]
        config["unknown_subdivision"]["output_subdir"] = f"unknown_subdivision_sensitivity_m{m}"
        metrics = run_unknown_subdivision(config, ckpt_path=checkpoint_path)
        row = {
            "dataset": name,
            "m": int(m),
            "selection_score": _selection_score(metrics),
            "overcluster_extra_clusters": int(m),
            "m_selection_offline_adjusted_quality": _offline_adjusted_quality(metrics),
            **metrics,
        }
        rows.append(row)
    return rows


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# 未知类细分 m 敏感性分析",
        "",
        "本报告扫描 `m=0,1,2,3`。`selection_score` 是不使用真实标签的无监督诊断分数；正式离线实验采用最小充分冗余规则：最终有效簇数必须等于 K，且覆盖率修正后的 NMI/ARI/Purity/Hungarian 均值提升超过 1 个百分点，才接受更大的 m。",
        "",
        "该选择依赖真实未知标签，只用于离线敏感性分析和固定正式实验配置，不应描述为在线无标签决策。",
        "",
    ]
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        target_k = dataset_rows[0].get("target_num_clusters")
        best = _select_minimal_sufficient_m(dataset_rows, target_k, min_quality_gain=0.01)
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"最小充分冗余规则选择的 m：`{best['m']}`；覆盖率修正质量：`{best['m_selection_offline_adjusted_quality']:.6f}`。",
                "",
                "| m | selection_score | adjusted_quality | fit_num_clusters | resolved_num_clusters | uncertain_ratio | coverage | NMI | ARI | Purity | Hungarian Accuracy | GMM BIC |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_rows:
            lines.append(
                "| {m} | {selection_score:.6f} | {adjusted_quality:.6f} | {fit_k} | {resolved_k} | {uncertain:.6f} | {coverage:.6f} | {nmi:.6f} | {ari:.6f} | {purity:.6f} | {hacc:.6f} | {bic:.3f} |".format(
                    m=int(row.get("m", 0)),
                    selection_score=float(row.get("selection_score", 0.0)),
                    adjusted_quality=float(row.get("m_selection_offline_adjusted_quality", 0.0)),
                    fit_k=row.get("fit_num_clusters", ""),
                    resolved_k=row.get("resolved_num_clusters", ""),
                    uncertain=float(row.get("uncertain_ratio") or 0.0),
                    coverage=float(row.get("coverage_of_total_test_unknown") or 0.0),
                    nmi=float(row.get("nmi") or 0.0),
                    ari=float(row.get("ari") or 0.0),
                    purity=float(row.get("purity") or 0.0),
                    hacc=float(row.get("hungarian_accuracy") or 0.0),
                    bic=float(row.get("gmm_bic") or 0.0),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULT_DIR)
    rows = []
    rows.extend(_run_dataset("oracle_kri16_demod", build_oracle_config, ORACLE_CHECKPOINT_PATH))
    rows.extend(_run_dataset("wisig_singleday_osr_k16_u12", build_wisig_config, WISIG_CHECKPOINT_PATH))
    save_json(RESULT_DIR / "subdivision_m_sensitivity.json", rows)
    _write_markdown(RESULT_DIR / "subdivision_m_sensitivity.md", rows)
    print(f"敏感性分析报告：{RESULT_DIR / 'subdivision_m_sensitivity.md'}")
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        target_k = dataset_rows[0].get("target_num_clusters")
        best = _select_minimal_sufficient_m(dataset_rows, target_k, min_quality_gain=0.01)
        print(
            f"{dataset}: selected m={best['m']} "
            f"adjusted_quality={best['m_selection_offline_adjusted_quality']:.6f}"
        )


if __name__ == "__main__":
    main()
