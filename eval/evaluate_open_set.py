from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.calibrators import OpenMaxCalibrator, apply_unknown_rejection, fuse_unknown_score, prototype_distance_unknown_score
from sei_osr.datasets import build_data_module
from sei_osr.modules.prototype_utils import activations_from_distances, predict_with_prototypes
from sei_osr.trainers import ClosedSetTrainer
from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, load_json, load_pickle, save_json
from sei_osr.utils.metrics import evaluate_open_set, save_confusion_matrix, save_prediction_csv
from sei_osr.utils.reporting import dataset_summary_path, write_final_report, write_summary_index
from sei_osr.utils.visualization import generate_open_set_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "oracle_osr_main.yaml"))
    parser.add_argument("--ckpt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_path = Path(args.ckpt) if args.ckpt else output_dir / "best_closed_set.pt"

    datamodule = build_data_module(config)
    trainer = ClosedSetTrainer(config, datamodule.bundle.num_known_classes, datamodule.bundle.signal_length)
    trainer.load_checkpoint(ckpt_path)

    openmax = OpenMaxCalibrator.from_state_dict(load_pickle(output_dir / "openmax.pkl"))
    fusion_cfg = load_json(output_dir / "fusion.json")
    stats_file = np.load(output_dir / "distance_stats.npz")

    test_known = trainer.extract_embeddings(datamodule.test_known_dataloader())
    test_unknown = trainer.extract_embeddings(datamodule.test_unknown_dataloader())
    prototypes = test_known["prototypes"]
    unknown_label = datamodule.bundle.num_known_classes

    all_embeddings = np.concatenate([test_known["embeddings"], test_unknown["embeddings"]], axis=0)
    all_labels = np.concatenate(
        [
            test_known["labels"],
            np.full(len(test_unknown["labels"]), unknown_label, dtype=np.int64),
        ]
    )
    pred_proto, _, distances = predict_with_prototypes(
        all_embeddings,
        prototypes,
        float(config["model"]["temperature"]),
    )
    openmax_out = openmax.predict(activations_from_distances(distances))
    q_om = openmax_out["unknown_prob"]
    q_pd = prototype_distance_unknown_score(distances, pred_proto, stats_file["mu"], stats_file["sigma"])
    q_u = fuse_unknown_score(
        q_om,
        q_pd,
        float(fusion_cfg["fusion_lambda"]),
        mode=str(fusion_cfg.get("fusion_mode", config["fusion"].get("mode", "linear"))),
    )
    y_pred = apply_unknown_rejection(
        known_pred=pred_proto,
        q_u=q_u,
        unknown_label=unknown_label,
        threshold=fusion_cfg.get("threshold"),
        thresholds_per_class=fusion_cfg.get("thresholds_per_class"),
    )

    metrics = evaluate_open_set(all_labels, y_pred, q_u, unknown_label)
    d_min = distances[np.arange(len(distances)), pred_proto]
    save_json(output_dir / "open_set_metrics.json", metrics)
    save_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        all_labels,
        y_pred,
        labels=list(range(datamodule.bundle.num_known_classes)) + [unknown_label],
    )
    if bool(config["eval"].get("save_predictions", True)):
        save_prediction_csv(
            output_dir / "open_set_predictions.csv",
            all_labels,
            y_pred,
            q_u,
            q_om,
            q_pd,
            d_min,
        )

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

    notes = [f"图表目录：{figure_dir}", "根目录结果汇总已改为按数据集分别保存，不会相互覆盖。"]
    split_file = config.get("prep", {}).get("split_file")
    if split_file:
        notes.append(f"本次实验使用的 split 文件：{split_file}")
    if fusion_cfg.get("threshold_mode") == "classwise_quantile":
        notes.append(f"拒识阈值模式：按类别分位数阈值（quantile={fusion_cfg.get('threshold_quantile')}）")
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
            path=dataset_summary_path(ROOT, str(dataset_name), str(output_dir)),
            metrics=metrics,
            config_path=str(config["_config_path"]),
            output_dir=str(output_dir),
            dataset_name=str(dataset_name),
            extra_notes=notes,
        )
        write_summary_index(
            path=ROOT / "RESULT_SUMMARY.md",
            entries=[
                {"label": "WiSig 数据集汇总", "path": "RESULT_SUMMARY_WISIG.md"},
                {"label": "Oracle 数据集汇总", "path": "RESULT_SUMMARY_ORACLE.md"},
            ],
            latest_dataset=str(dataset_name),
            latest_output_dir=str(output_dir),
            latest_config_path=str(config["_config_path"]),
        )


if __name__ == "__main__":
    main()
