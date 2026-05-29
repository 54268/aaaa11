from __future__ import annotations

import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
RAW_WISIG_ROOT = PROJECT_ROOT / "data" / "raw" / "wisig"
SEEDS = list(range(10))

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_ROOT))

from src.clustering_baselines import BASELINES, BaselineSpec, run_baseline
from src.metrics import confusion_after_hungarian
from src.reporting import (
    dataframe_to_markdown,
    plot_confusion,
    plot_metric_bar,
    summarize_results,
    write_json,
    write_markdown_table,
)
from src.wisig_loader import (
    LoadedSubset,
    balanced_sample,
    find_subset_candidates,
    load_subset,
    subset_inventory_row,
    tx_samples,
)


def ensure_inside_test(path: Path) -> Path:
    path = path.resolve()
    root = TEST_ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"拒绝写入 test_wisig 之外的路径：{path}")
    forbidden = [
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "settings",
        PROJECT_ROOT / "functions",
        PROJECT_ROOT / "data" / "raw",
    ]
    for item in forbidden:
        try:
            if path == item.resolve() or item.resolve() in path.parents:
                raise RuntimeError(f"拒绝写入受保护目录：{path}")
        except FileNotFoundError:
            continue
    return path


def mkdir(path: Path) -> Path:
    path = ensure_inside_test(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


RESULTS = mkdir(TEST_ROOT / "results")
LOGS = mkdir(TEST_ROOT / "logs")
CACHE = mkdir(TEST_ROOT / "cache")
CONFIG = mkdir(TEST_ROOT / "config")
FIGURES = mkdir(RESULTS / "figures")
ERROR_LOG = LOGS / "errors.log"
RUN_LOG = LOGS / "run.log"


def log(message: str) -> None:
    print(message, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def log_error(context: str, exc: BaseException) -> None:
    text = f"\n[{context}]\n{repr(exc)}\n{traceback.format_exc()}\n"
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(text)
    log(f"[ERROR] {context}: {exc}")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path = ensure_inside_test(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def check_environment() -> dict[str, Any]:
    deps = ["numpy", "pandas", "sklearn", "matplotlib", "scipy"]
    rows = []
    ok = True
    for name in deps:
        try:
            module = importlib.import_module(name)
            rows.append({"package": name, "available": True, "version": getattr(module, "__version__", "unknown")})
        except Exception as exc:
            ok = False
            rows.append({"package": name, "available": False, "version": "", "error": repr(exc)})
    lines = ["# 环境依赖检查", ""]
    for row in rows:
        if row["available"]:
            lines.append(f"- {row['package']}: OK, version={row['version']}")
        else:
            lines.append(f"- {row['package']}: 缺失，错误={row.get('error', '')}")
    if not ok:
        lines.extend(
            [
                "",
                "缺少依赖时可在 pytorch 环境中安装：",
                "conda install numpy pandas scikit-learn matplotlib scipy",
            ]
        )
    (TEST_ROOT / "environment_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ok": ok, "dependencies": rows}


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def detect_project_protocol() -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "config_sources_found": [],
        "warnings": [],
    }
    try:
        from settings import default_wisig_config

        cfg = default_wisig_config(PROJECT_ROOT)
        protocol["config_sources_found"].append("settings.default_wisig_config")
        protocol["settings_config"] = {
            "output_dir": cfg["project"]["output_dir"],
            "raw_path": cfg["prep"].get("raw_path"),
            "split_file": cfg["prep"].get("split_file"),
            "processed_root": cfg["prep"].get("processed_root"),
            "signal_length": cfg["data"].get("signal_length"),
            "normalize": cfg["data"].get("normalize"),
            "batch_size": cfg["data"].get("batch_size"),
        }
    except Exception as exc:
        protocol["warnings"].append(f"无法导入 settings.default_wisig_config: {exc}")
        cfg = None

    split_path = Path(cfg["prep"]["split_file"]) if cfg else PROJECT_ROOT / "data" / "splits" / "wisig" / "single_day_rx1_eq0" / "wisig_single_day_rx1_eq0_k16_u12_seed42.json"
    split_payload = safe_load_json(split_path)
    if split_payload:
        protocol["config_sources_found"].append(str(split_path))
        protocol.update(
            {
                "subset": "SingleDay",
                "known_tx_list": split_payload.get("known_classes", []),
                "unknown_tx_list": split_payload.get("unknown_classes", []),
                "include_rx": split_payload.get("include_rx"),
                "include_capture_dates": split_payload.get("include_capture_dates"),
                "include_equalized": split_payload.get("include_equalized"),
                "unknown_count": int(len(split_payload.get("unknown_classes", []))),
                "samples_per_class": split_payload.get("samples_per_class"),
                "split_file": str(split_path),
            }
        )
    else:
        protocol["warnings"].append(f"无法读取 split 文件：{split_path}")

    metrics_path = PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12" / "unknown_subdivision" / "unknown_subdivision_metrics.json"
    metrics = safe_load_json(metrics_path)
    if metrics:
        protocol["config_sources_found"].append(str(metrics_path))
        protocol["existing_wisig_unknown_subdivision"] = {
            "path": str(metrics_path),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "purity": metrics.get("purity"),
            "hungarian_accuracy": metrics.get("hungarian_accuracy"),
            "resolved_num_clusters": metrics.get("resolved_num_clusters"),
            "uncertain_size": metrics.get("uncertain_size"),
        }
    else:
        protocol["warnings"].append("未找到已有 WiSig 细分指标文件。")

    write_json(CONFIG / "detected_project_protocol.json", protocol)
    return protocol


def load_all_subsets() -> dict[str, LoadedSubset]:
    candidates = find_subset_candidates(RAW_WISIG_ROOT)
    loaded = {name: load_subset(name, path) for name, path in candidates.items()}
    rows = []
    for subset in loaded.values():
        rows.append(subset_inventory_row(subset))
    inv_dir = mkdir(RESULTS / "inventory")
    json_rows = []
    for row in rows:
        clean = dict(row)
        for key in ["tx_list", "rx_list", "day_list"]:
            if key in clean and not isinstance(clean[key], list):
                clean[key] = list(clean[key])
        json_rows.append(clean)
    write_json(inv_dir / "subset_inventory.json", json_rows)
    df = pd.DataFrame(rows)
    for col in ["tx_list", "rx_list", "day_list"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: ",".join(v) if isinstance(v, list) else v)
    save_csv(inv_dir / "subset_inventory.csv", df)
    write_markdown_table(inv_dir / "subset_inventory.md", "WiSig subset 扫描结果", df)
    return loaded


def preferred_equalized(payload: dict[str, Any]) -> list[int] | None:
    eq = list(payload.get("equalized_list", []))
    if 0 in eq:
        return [0]
    if eq:
        return [eq[0]]
    return None


def run_rows_for_sample(
    x: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    seed: int,
    subset: str,
    protocol_name: str,
    selected_tx_ids: list[str],
    samples_per_class: int,
    out_confusion_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for spec in BASELINES:
        metrics, pred, pca_dim = run_baseline(x, y, n_clusters, spec, seed)
        if out_confusion_path is not None and spec.feature_type == "raw_iq" and spec.clustering_method == "kmeans":
            matrix, true_labels, pred_labels = confusion_after_hungarian(y, pred)
            plot_confusion(matrix, true_labels, pred_labels, out_confusion_path, "SingleDay 当前协议 Raw IQ + PCA + K-Means 混淆矩阵")
        row = {
            "subset": subset,
            "protocol": protocol_name,
            "method": spec.name,
            "feature_type": spec.feature_type,
            "clustering_method": spec.clustering_method,
            "seed": seed,
            "selected_tx_ids": ";".join(selected_tx_ids),
            "samples_per_class": int(samples_per_class),
            "input_total_samples": int(len(y)),
            "n_clusters": int(n_clusters),
            "pca_dim": int(pca_dim),
            **metrics,
        }
        rows.append(row)
    return rows


def run_current_singleday(protocol: dict[str, Any], loaded: dict[str, LoadedSubset]) -> pd.DataFrame:
    out_dir = mkdir(RESULTS / "singleday_current_protocol")
    subset = loaded.get("SingleDay")
    if subset is None or subset.payload is None:
        (out_dir / "report.md").write_text("# 当前 SingleDay 协议\n\nSingleDay 无法加载，跳过。\n", encoding="utf-8")
        return pd.DataFrame()

    unknown_tx = list(protocol.get("unknown_tx_list") or [])
    if not unknown_tx:
        (out_dir / "report.md").write_text("# 当前 SingleDay 协议\n\n无法识别当前 unknown Tx 列表，跳过。\n", encoding="utf-8")
        return pd.DataFrame()
    samples = tx_samples(
        subset.payload,
        tx_ids=unknown_tx,
        include_rx=protocol.get("include_rx"),
        include_days=protocol.get("include_capture_dates"),
        include_equalized=protocol.get("include_equalized"),
    )
    rows = []
    sample_protocol = {
        "subset": "SingleDay",
        "unknown_tx_list": unknown_tx,
        "include_rx": protocol.get("include_rx"),
        "include_capture_dates": protocol.get("include_capture_dates"),
        "include_equalized": protocol.get("include_equalized"),
        "max_per_class": 500,
        "seed_samples": {},
    }
    for seed in SEEDS:
        try:
            x, y, info = balanced_sample(samples, unknown_tx, max_per_class=500, seed=seed)
            sample_protocol["seed_samples"][str(seed)] = info
            confusion_path = FIGURES / "singleday_current_protocol_rawiq_kmeans_confusion.png" if seed == 0 else None
            rows.extend(
                run_rows_for_sample(
                    x=x,
                    y=y,
                    n_clusters=len(unknown_tx),
                    seed=seed,
                    subset="SingleDay",
                    protocol_name="singleday_current_protocol",
                    selected_tx_ids=unknown_tx,
                    samples_per_class=info["samples_per_class"],
                    out_confusion_path=confusion_path,
                )
            )
        except Exception as exc:
            log_error(f"singleday_current_protocol seed={seed}", exc)
    df = pd.DataFrame(rows)
    save_csv(out_dir / "per_seed_results.csv", df)
    summary = summarize_results(df, ["method"])
    save_csv(out_dir / "summary_results.csv", summary)
    write_json(out_dir / "sampled_protocol.json", sample_protocol)
    write_markdown_table(out_dir / "report.md", "当前 SingleDay unknown 协议直接聚类诊断", summary)
    plot_metric_bar(summary, FIGURES / "singleday_current_protocol_hungarian_accuracy.png", "SingleDay 当前协议 Hungarian Accuracy", "hungarian_accuracy")
    plot_metric_bar(summary, FIGURES / "singleday_current_protocol_nmi.png", "SingleDay 当前协议 NMI", "nmi")
    return df


def run_p1(loaded: dict[str, LoadedSubset]) -> pd.DataFrame:
    out_dir = mkdir(RESULTS / "all_subsets_p1")
    rows = []
    for name, subset in loaded.items():
        if subset.payload is None:
            continue
        tx_list = list(subset.payload.get("tx_list", []))
        if len(tx_list) < 6:
            continue
        eq = preferred_equalized(subset.payload)
        samples = tx_samples(subset.payload, include_equalized=eq)
        available_tx = [tx for tx in tx_list if tx in samples and len(samples[tx]) > 0]
        if len(available_tx) < 6:
            continue
        for seed in SEEDS:
            try:
                rng = np.random.default_rng(seed)
                selected = sorted(rng.choice(available_tx, size=6, replace=False).tolist())
                x, y, info = balanced_sample(samples, selected, max_per_class=300, seed=seed)
                rows.extend(run_rows_for_sample(x, y, 6, seed, name, "P1_6class_balanced", selected, info["samples_per_class"]))
            except Exception as exc:
                log_error(f"P1 subset={name} seed={seed}", exc)
    df = pd.DataFrame(rows)
    save_csv(out_dir / "per_seed_results.csv", df)
    summary = summarize_results(df, ["subset", "method"])
    save_csv(out_dir / "summary_results.csv", summary)
    write_markdown_table(out_dir / "report.md", "P1 四个官方子集 6 类平衡弱基线诊断", summary)
    plot_metric_bar(summary, FIGURES / "all_subsets_p1_hungarian_accuracy.png", "P1 各 subset Hungarian Accuracy", "hungarian_accuracy", category_col="subset")
    plot_metric_bar(summary, FIGURES / "all_subsets_p1_nmi.png", "P1 各 subset NMI", "nmi", category_col="subset")
    return df


def common_tx_groups(loaded: dict[str, LoadedSubset]) -> tuple[list[tuple[str, list[str], list[str]]], dict[str, Any]]:
    tx_sets = {name: set(subset.payload.get("tx_list", [])) for name, subset in loaded.items() if subset.payload is not None}
    inventory = {name: sorted(values) for name, values in tx_sets.items()}
    groups = []
    if len(tx_sets) >= 2:
        all_common = set.intersection(*tx_sets.values()) if tx_sets else set()
        inventory["all_four_common"] = sorted(all_common)
        if len(all_common) >= 4:
            groups.append(("all_successful_subsets", list(tx_sets.keys()), sorted(all_common)))
        else:
            priorities = [("SingleDay_ManySig", ["SingleDay", "ManySig"]), ("SingleDay_ManyRx", ["SingleDay", "ManyRx"])]
            seen = set()
            for group_name, names in priorities:
                if all(n in tx_sets for n in names):
                    common = set.intersection(*(tx_sets[n] for n in names))
                    inventory[f"{group_name}_common"] = sorted(common)
                    if len(common) >= 4:
                        groups.append((group_name, names, sorted(common)))
                        seen.add(tuple(names))
            if not groups:
                names = list(tx_sets.keys())
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        pair = [names[i], names[j]]
                        if tuple(pair) in seen:
                            continue
                        common = set.intersection(tx_sets[pair[0]], tx_sets[pair[1]])
                        inventory[f"{pair[0]}_{pair[1]}_common"] = sorted(common)
                        if len(common) >= 4:
                            groups.append((f"{pair[0]}_{pair[1]}", pair, sorted(common)))
                            break
                    if groups:
                        break
    return groups, inventory


def run_p2(loaded: dict[str, LoadedSubset]) -> pd.DataFrame:
    out_dir = mkdir(RESULTS / "shared_tx_p2")
    groups, inventory = common_tx_groups(loaded)
    write_json(out_dir / "shared_tx_inventory.json", inventory)
    rows = []
    if not groups:
        (out_dir / "report.md").write_text("# P2 Shared-Tx 对齐比较\n\n共同 Tx 数量不足，跳过。\n", encoding="utf-8")
        save_csv(out_dir / "per_seed_results.csv", pd.DataFrame())
        save_csv(out_dir / "summary_results.csv", pd.DataFrame())
        return pd.DataFrame()
    group_name, subset_names, common = groups[0]
    for seed in SEEDS:
        try:
            rng = np.random.default_rng(seed)
            k = min(6, len(common))
            selected = sorted(rng.choice(common, size=k, replace=False).tolist()) if len(common) > k else sorted(common)
            subset_samples = {}
            min_available = 300
            for subset_name in subset_names:
                payload = loaded[subset_name].payload
                eq = preferred_equalized(payload)
                samples = tx_samples(payload, tx_ids=selected, include_equalized=eq)
                subset_samples[subset_name] = samples
                min_available = min(min_available, *(len(samples[tx]) for tx in selected))
            max_per_class = int(min(300, min_available))
            for subset_name in subset_names:
                x, y, info = balanced_sample(subset_samples[subset_name], selected, max_per_class=max_per_class, seed=seed)
                rows.extend(run_rows_for_sample(x, y, k, seed, subset_name, f"P2_{group_name}", selected, info["samples_per_class"]))
        except Exception as exc:
            log_error(f"P2 group={group_name} seed={seed}", exc)
    df = pd.DataFrame(rows)
    save_csv(out_dir / "per_seed_results.csv", df)
    summary = summarize_results(df, ["protocol", "subset", "method"])
    save_csv(out_dir / "summary_results.csv", summary)
    write_markdown_table(out_dir / "report.md", "P2 共有 Tx 对齐弱基线诊断", summary)
    plot_metric_bar(summary, FIGURES / "shared_tx_p2_hungarian_accuracy.png", "P2 Shared-Tx Hungarian Accuracy", "hungarian_accuracy", category_col="subset")
    return df


def _run_condition(name: str, subset: LoadedSubset, selected: list[str], seed: int, include_rx: list[str] | None, include_days: list[str] | None, include_equalized: list[int] | None) -> list[dict[str, Any]]:
    samples = tx_samples(subset.payload, tx_ids=selected, include_rx=include_rx, include_days=include_days, include_equalized=include_equalized)
    x, y, info = balanced_sample(samples, selected, max_per_class=300, seed=seed)
    return run_rows_for_sample(x, y, len(selected), seed, subset.name, name, selected, info["samples_per_class"])


def run_p3(loaded: dict[str, LoadedSubset]) -> pd.DataFrame:
    out_dir = mkdir(RESULTS / "condition_sensitivity_p3")
    rows = []
    skipped = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        try:
            single = loaded.get("SingleDay")
            if single and single.payload is not None:
                tx_list = list(single.payload.get("tx_list", []))
                selected = sorted(rng.choice(tx_list, size=6, replace=False).tolist())
                rx_list = list(single.payload.get("rx_list", []))
                day_list = list(single.payload.get("capture_date_list", []))
                eq = preferred_equalized(single.payload)
                if rx_list:
                    rows.extend(_run_condition("P3-1_SingleDay_FixedRx", single, selected, seed, [rx_list[0]], day_list[:1] if day_list else None, eq))
                    rows.extend(_run_condition("P3-2_SingleDay_MixedRx", single, selected, seed, None, day_list[:1] if day_list else None, eq))
                else:
                    skipped.append({"protocol": "P3-1/P3-2", "reason": "SingleDay 无可靠 receiver 字段"})
            else:
                skipped.append({"protocol": "P3-1/P3-2", "reason": "SingleDay 无法加载"})
        except Exception as exc:
            log_error(f"P3 SingleDay seed={seed}", exc)

        try:
            manysig = loaded.get("ManySig")
            if manysig and manysig.payload is not None:
                tx_list = list(manysig.payload.get("tx_list", []))
                selected = sorted(rng.choice(tx_list, size=6, replace=False).tolist())
                rx_list = list(manysig.payload.get("rx_list", []))
                day_list = list(manysig.payload.get("capture_date_list", []))
                eq = preferred_equalized(manysig.payload)
                if rx_list and day_list:
                    rows.extend(_run_condition("P3-3_ManySig_FixedRxDay", manysig, selected, seed, [rx_list[0]], [day_list[0]], eq))
                    rows.extend(_run_condition("P3-4_ManySig_MixedDay", manysig, selected, seed, [rx_list[0]], None, eq))
                else:
                    skipped.append({"protocol": "P3-3/P3-4", "reason": "ManySig 无可靠 receiver/day 字段"})
            else:
                skipped.append({"protocol": "P3-3/P3-4", "reason": "ManySig 无法加载"})
        except Exception as exc:
            log_error(f"P3 ManySig seed={seed}", exc)

        try:
            manyrx = loaded.get("ManyRx")
            if manyrx and manyrx.payload is not None:
                tx_list = list(manyrx.payload.get("tx_list", []))
                selected = sorted(rng.choice(tx_list, size=6, replace=False).tolist())
                eq = preferred_equalized(manyrx.payload)
                rows.extend(_run_condition("P3-5_ManyRx_MixedRx", manyrx, selected, seed, None, None, eq))
            else:
                skipped.append({"protocol": "P3-5", "reason": "ManyRx 无法加载"})
        except Exception as exc:
            log_error(f"P3 ManyRx seed={seed}", exc)

    df = pd.DataFrame(rows)
    save_csv(out_dir / "per_seed_results.csv", df)
    summary = summarize_results(df, ["protocol", "subset", "method"])
    save_csv(out_dir / "summary_results.csv", summary)
    write_json(out_dir / "skipped_protocols.json", skipped)
    write_markdown_table(out_dir / "report.md", "P3 Receiver / Day 条件敏感性诊断", summary)
    if not summary.empty:
        plot_metric_bar(summary, FIGURES / "condition_sensitivity_p3_hungarian_accuracy.png", "P3 条件敏感性 Hungarian Accuracy", "hungarian_accuracy", category_col="protocol")
    return df


def read_oracle_reference() -> dict[str, Any] | None:
    out_dir = mkdir(RESULTS / "oracle_reference")
    path = PROJECT_ROOT / "outputs" / "oracle_kri16_demod_known_first" / "unknown_subdivision" / "unknown_subdivision_metrics.json"
    payload = safe_load_json(path)
    if not payload:
        (out_dir / "readme.md").write_text("# Oracle 已有结果旁注\n\n未找到可读取的 Oracle 细分结果。\n", encoding="utf-8")
        return None
    keys = ["nmi", "ari", "purity", "hungarian_accuracy", "resolved_num_clusters", "uncertain_size", "unknown_cache_precision", "unknown_cache_recall"]
    ref = {"source_path": str(path), **{key: payload.get(key) for key in keys}}
    write_json(out_dir / "oracle_existing_metrics.json", ref)
    lines = ["# Oracle 已有结果旁注", "", "该结果为当前项目已有输出的只读摘录，不是本次同协议弱基线实验。", ""]
    for key in keys:
        lines.append(f"- `{key}`: {ref.get(key)}")
    (out_dir / "readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ref


def fmt_mean_std(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}"


def compact_table(summary: pd.DataFrame, first_cols: list[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary.iterrows():
        out = {col: row[col] for col in first_cols if col in row}
        out.update(
            {
                "NMI mean±std": fmt_mean_std(row, "nmi"),
                "ARI mean±std": fmt_mean_std(row, "ari"),
                "Purity mean±std": fmt_mean_std(row, "purity"),
                "Hungarian Accuracy mean±std": fmt_mean_std(row, "hungarian_accuracy"),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def conclusion_text(single_summary: pd.DataFrame, p1_summary: pd.DataFrame, p3_summary: pd.DataFrame) -> tuple[list[str], bool]:
    lines = []
    supports_easy = False
    raw = single_summary[single_summary["method"] == "Raw IQ + PCA + K-Means"] if not single_summary.empty else pd.DataFrame()
    if not raw.empty and float(raw.iloc[0]["hungarian_accuracy_mean"]) >= 0.90:
        supports_easy = True
        lines.append("规则 A 触发：在当前 SingleDay 受控协议下，即使不使用深度特征、拒识模块和本文提出的细分方法，最简单的原始 I/Q 聚类基线也已达到 90% 以上。这说明该设置下未知类细分任务可能接近饱和，不适合作为证明复杂细分算法优势的唯一主要实验场景。")

    high_methods = 0
    if not single_summary.empty:
        high_methods = int((single_summary["hungarian_accuracy_mean"] >= 0.90).sum())
    mixed_drop = False
    if high_methods >= 2:
        candidate_frames = []
        if not p1_summary.empty:
            candidate_frames.append(p1_summary[p1_summary["subset"].isin(["ManyRx", "ManySig"])])
        if not p3_summary.empty:
            candidate_frames.append(p3_summary[p3_summary["protocol"].str.contains("Mixed", na=False)])
        candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        if not candidates.empty:
            single_best = float(single_summary["hungarian_accuracy_mean"].max())
            mixed_min = float(candidates["hungarian_accuracy_mean"].min())
            mixed_drop = (single_best - mixed_min) >= 0.10
    if high_methods >= 2 and mixed_drop:
        supports_easy = True
        lines.append("规则 B 触发：WiSig 在当前协议下的极高未知类细分结果主要与受控采集条件有关。引入 receiver 或 day 变化后，简单基线性能明显下降。论文中应将当前 SingleDay 设置定位为容易场景，将跨接收机或跨日期协议作为主要挑战设置。")

    if not single_summary.empty and bool((single_summary["hungarian_accuracy_mean"] < 0.80).all()):
        lines.append("规则 C 触发：当前方法在 WiSig 上的高细分性能不能简单归因于数据天然易分，后续应通过统一 unknown cache、统一特征输入与消融实验进一步确认方法贡献。")

    if p1_summary.empty and p3_summary.empty:
        lines.append("规则 D 触发：当前可获得的 compact subset 结构或标签信息不足以完成多 receiver / 多 day 严格比较。本次结论仅限于当前 SingleDay 协议的天然可分性诊断，不对 WiSig 全部采集条件作推广。")

    if not lines:
        lines.append("自动规则未形成强结论：当前结果需要结合 P1/P2/P3 表格人工判断。")
    return lines, supports_easy


def generate_final_report(protocol: dict[str, Any], inventory: pd.DataFrame, single_df: pd.DataFrame, p1_df: pd.DataFrame, p2_df: pd.DataFrame, p3_df: pd.DataFrame, oracle_ref: dict[str, Any] | None) -> None:
    single_summary = summarize_results(single_df, ["method"])
    p1_summary = summarize_results(p1_df, ["subset", "method"])
    p2_summary = summarize_results(p2_df, ["protocol", "subset", "method"])
    p3_summary = summarize_results(p3_df, ["protocol", "subset", "method"])
    conclusions, supports_easy = conclusion_text(single_summary, p1_summary, p3_summary)

    lines = [
        "# WiSig 数据集未知类天然可分性诊断实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验用于诊断 WiSig 当前受控协议是否天然容易进行未知类细分，不用于证明主方法性能。实验刻意不使用 CVCNN 深度特征、OpenMax、原型距离、伪未知样本、known prototype guided clustering 和当前 unknown cache，只对真实 unknown 类样本直接使用经典弱基线聚类。",
        "",
        "## 2. 目录隔离说明",
        "",
        "- 本实验所有新增内容均位于 `test_wisig/`。",
        "- 原项目代码、配置与数据仅以只读方式访问。",
        "- 原有 `outputs/` 未被覆盖或写入新的诊断结果。",
        "",
        "## 3. 数据扫描结果",
        "",
    ]
    lines.extend(dataframe_to_markdown(inventory.drop(columns=[c for c in ["keys", "tx_list", "rx_list", "day_list"] if c in inventory.columns], errors="ignore")))
    lines.extend(["", "## 4. 当前 SingleDay 主协议直接聚类结果", ""])
    lines.append(f"- 当前 unknown 类数量：{protocol.get('unknown_count')}")
    lines.append(f"- receiver 筛选：{protocol.get('include_rx')}")
    lines.append(f"- day 筛选：{protocol.get('include_capture_dates')}")
    lines.append(f"- equalized 筛选：{protocol.get('include_equalized')}")
    lines.append("- 说明：此处未经过任何拒识模块和深度特征提取。")
    lines.append("")
    lines.extend(dataframe_to_markdown(compact_table(single_summary, ["method"])))
    lines.extend(["", "## 5. 四个 subset 的 P1 统一 6 类结果", ""])
    lines.extend(dataframe_to_markdown(compact_table(p1_summary, ["subset", "method"])))
    lines.extend(["", "## 6. Shared-Tx 的 P2 对齐结果", ""])
    if p2_summary.empty:
        lines.append("P2 未能可靠运行，详见 `results/shared_tx_p2/shared_tx_inventory.json` 和 `report.md`。")
    else:
        lines.append("P2 使用共同 Tx 集合对齐 subset，能够减少不同 subset 随机抽到不同 Tx 难度所带来的偏差。")
        lines.append("")
        lines.extend(dataframe_to_markdown(compact_table(p2_summary, ["protocol", "subset", "method"])))
    lines.extend(["", "## 7. Receiver / Day 敏感性的 P3 结果", ""])
    if p3_summary.empty:
        lines.append("P3 未能可靠运行，详见 skipped_protocols.json。")
    else:
        lines.extend(dataframe_to_markdown(compact_table(p3_summary, ["protocol", "subset", "method"])))
    lines.extend(["", "## 8. Oracle 已有结果旁注", ""])
    if oracle_ref:
        lines.append("以下 Oracle 结果只是当前项目已有输出的只读摘录，不是本次同协议弱基线结果。")
        lines.append("")
        for key, value in oracle_ref.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("未安全读取到 Oracle 已有细分结果。")
    lines.extend(["", "## 9. 自动判断结论", ""])
    lines.extend(f"- {line}" for line in conclusions)
    lines.extend(["", "## 10. 论文实验建议", ""])
    if supports_easy:
        lines.extend(
            [
                "- WiSig 当前 SingleDay 受控协议建议作为易场景验证，不建议作为证明未知类细分优势的唯一主要场景。",
                "- 论文中应增加或突出跨 receiver、跨 day、ManyRx/ManySig 等更困难设置。",
                "- Oracle 更适合作为主要困难场景，用于展示方法在非饱和协议下的实际价值。",
                "- HyperRSI、OpenRFI、OFSCIL、EPD/IOWL 等复杂 baseline 后续值得在 Oracle 或更困难 WiSig 协议上运行。",
            ]
        )
    else:
        lines.extend(
            [
                "- WiSig 是否过易需要结合更多协议结果继续判断。",
                "- 建议保留 Oracle 作为主要困难场景，并继续补充跨 receiver/day 设置。",
            ]
        )
    lines.append("")
    (RESULTS / "final_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# WiSig 天然可分性诊断实验",
        "",
        "本目录完全隔离于主实验，用于诊断 WiSig 当前 unknown 类是否在原始 I/Q 或 FFT 弱特征下已经容易聚类。",
        "",
        "运行入口：",
        "",
        "```text",
        "python test_wisig/run_wisig_separability_diagnostic.py",
        "```",
        "",
        "输入数据只读来自 `data/raw/wisig/`，当前项目配置和已有结果只读用于记录协议和旁注。所有新增脚本、缓存、日志、CSV、JSON、图片和 Markdown 报告均写入 `test_wisig/`。",
        "",
        "主要输出：",
        "",
        "- `config/detected_project_protocol.json`：自动识别到的当前 WiSig 主协议。",
        "- `environment_check.txt`：依赖检查。",
        "- `results/inventory/`：四个 official compact subsets 扫描结果。",
        "- `results/singleday_current_protocol/`：当前 SingleDay unknown 协议直接聚类结果。",
        "- `results/all_subsets_p1/`：四个 subset 的统一 6 类弱基线结果。",
        "- `results/shared_tx_p2/`：共有 Tx 对齐比较。",
        "- `results/condition_sensitivity_p3/`：receiver/day 条件敏感性诊断。",
        "- `results/oracle_reference/`：当前已有 Oracle 细分结果旁注。",
        "- `results/final_diagnostic_report.md`：总报告。",
    ]
    (TEST_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUN_LOG.write_text("", encoding="utf-8")
    ERROR_LOG.write_text("", encoding="utf-8")
    random.seed(0)
    np.random.seed(0)
    write_readme()
    log("[1/8] 检查依赖")
    check_environment()
    log("[2/8] 只读识别当前项目 WiSig 协议")
    protocol = detect_project_protocol()
    log(f"配置来源：{protocol.get('config_sources_found')}")
    if protocol.get("warnings"):
        log(f"协议识别警告：{protocol['warnings']}")
    log("[3/8] 扫描并加载 WiSig 官方 compact subsets")
    loaded = load_all_subsets()
    inventory_df = pd.read_csv(RESULTS / "inventory" / "subset_inventory.csv", encoding="utf-8-sig")
    log("成功加载 subset：" + ", ".join(name for name, subset in loaded.items() if subset.payload is not None))
    log("[4/8] 运行当前 SingleDay unknown 协议直接聚类诊断")
    single_df = run_current_singleday(protocol, loaded)
    log("[5/8] 运行 P1 四个 subset 统一 6 类弱基线诊断")
    p1_df = run_p1(loaded)
    log("[6/8] 运行 P2 shared-Tx 对齐诊断")
    p2_df = run_p2(loaded)
    log("[7/8] 运行 P3 receiver/day 条件敏感性诊断")
    p3_df = run_p3(loaded)
    log("[8/8] 读取 Oracle 旁注并生成最终报告")
    oracle_ref = read_oracle_reference()
    generate_final_report(protocol, inventory_df, single_df, p1_df, p2_df, p3_df, oracle_ref)

    single_summary = summarize_results(single_df, ["method"])
    p1_summary = summarize_results(p1_df, ["subset", "method"])
    p2_summary = summarize_results(p2_df, ["protocol", "subset", "method"])
    p3_summary = summarize_results(p3_df, ["protocol", "subset", "method"])
    conclusions, supports_easy = conclusion_text(single_summary, p1_summary, p3_summary)
    log("")
    log("===== 诊断完成 =====")
    log("新建内容均位于 test_wisig/：README、入口脚本、src、config、cache、logs、results、environment_check.txt。")
    log("确认未写入 outputs/、settings/、functions/、data/raw/wisig/，未修改主实验入口。")
    log("成功识别到 subset：" + ", ".join(name for name, subset in loaded.items() if subset.payload is not None))
    log("当前 SingleDay unknown 协议三个 baseline 汇总：")
    for _, row in single_summary.iterrows():
        log(f"- {row['method']}: NMI={row['nmi_mean']:.6f}±{row['nmi_std']:.6f}, Hungarian={row['hungarian_accuracy_mean']:.6f}±{row['hungarian_accuracy_std']:.6f}")
    log(f"P1 成功运行：{not p1_summary.empty}；P2 成功运行：{not p2_summary.empty}；P3 成功运行：{not p3_summary.empty}")
    log("是否支持当前 WiSig SingleDay 细分设置过于容易：" + ("支持" if supports_easy else "未形成强支持"))
    for line in conclusions:
        log("- " + line)
    log(f"最终报告：{RESULTS / 'final_diagnostic_report.md'}")


if __name__ == "__main__":
    main()

