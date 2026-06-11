from __future__ import annotations

import csv
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import genpareto
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_io import IQDataset, LoadedProtocol, make_iq_dataset
from functions.methods.openmax_wrapper import OpenMaxCalibrator
from metrics import evaluate_clustering, evaluate_open_set
from openrfi_grouping import openrfi_world_prototype_grouping, openrfi_world_prototype_grouping_scores


warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _save_open_set_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["y_true", "y_pred", "unknown_score", "is_unknown", "unknown_label"],
        )
        writer.writeheader()
        for true_label, pred_label, score in zip(y_true, y_pred, unknown_score):
            writer.writerow(
                {
                    "y_true": int(true_label),
                    "y_pred": int(pred_label),
                    "unknown_score": float(score),
                    "is_unknown": int(int(true_label) == int(unknown_label)),
                    "unknown_label": int(unknown_label),
                }
            )


def loader(dataset: IQDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def train_classifier(
    model: torch.nn.Module,
    train_set: IQDataset,
    val_set: IQDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_margin_labels: bool,
    checkpoint_path: Path,
    reuse_checkpoint: bool,
    confusing_sample_generator: torch.nn.Module | None = None,
    confusing_sample_discriminator: torch.nn.Module | None = None,
    confusing_noise_dim: int = 32,
    confusing_beta: float = 0.1,
    confusing_gan_lr: float = 2e-4,
) -> Dict[str, object]:
    model.to(device)
    if confusing_sample_generator is not None:
        confusing_sample_generator.to(device)
    if confusing_sample_discriminator is not None:
        confusing_sample_discriminator.to(device)
    train_loader = loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = loader(val_set, batch_size=batch_size, shuffle=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    training_status = "trained"
    if reuse_checkpoint and checkpoint_path.exists():
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except RuntimeError as exc:
            training_status = "retrained_incompatible_checkpoint"
            print(f"[{checkpoint_path.stem}] checkpoint incompatible with current model; retraining ({exc})")
        else:
            val_acc = closed_set_accuracy(model, val_loader, device)
            return {
                "best_val_acc": float(val_acc),
                "best_epoch": 0,
                "training_status": "reused_checkpoint",
            }

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_confusing_samples = (
        confusing_sample_generator is not None
        and confusing_sample_discriminator is not None
        and hasattr(model, "fake_loss")
    )
    if use_confusing_samples:
        opt_d = torch.optim.Adam(confusing_sample_discriminator.parameters(), lr=confusing_gan_lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(confusing_sample_generator.parameters(), lr=confusing_gan_lr, betas=(0.5, 0.999))
    else:
        opt_d = None
        opt_g = None
    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if confusing_sample_generator is not None:
            confusing_sample_generator.train()
        if confusing_sample_discriminator is not None:
            confusing_sample_discriminator.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            if use_confusing_samples:
                batch_size_now = x.size(0)

                opt_d.zero_grad()
                real_logits = confusing_sample_discriminator(x)
                real_target = torch.ones_like(real_logits)
                loss_d_real = F.binary_cross_entropy_with_logits(real_logits, real_target)
                noise = torch.randn(batch_size_now, confusing_noise_dim, device=device)
                fake_x = confusing_sample_generator(noise)
                fake_logits = confusing_sample_discriminator(fake_x.detach())
                fake_target = torch.zeros_like(fake_logits)
                loss_d_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_target)
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                opt_g.zero_grad()
                noise = torch.randn(batch_size_now, confusing_noise_dim, device=device)
                fake_x = confusing_sample_generator(noise)
                fake_logits = confusing_sample_discriminator(fake_x)
                loss_g_adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                loss_g = loss_g_adv + confusing_beta * model.fake_loss(fake_x)
                loss_g.backward()
                opt_g.step()

            opt.zero_grad()
            if hasattr(model, "training_loss"):
                loss = model.training_loss(x, y)
            else:
                logits = model(x, y if use_margin_labels else None)
                loss = F.cross_entropy(logits, y)
            if use_confusing_samples:
                noise = torch.randn(x.size(0), confusing_noise_dim, device=device)
                fake_x = confusing_sample_generator(noise)
                loss = loss + confusing_beta * model.fake_loss(fake_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        if use_confusing_samples:
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                if hasattr(model, "training_loss"):
                    loss = model.training_loss(x, y)
                else:
                    logits = model(x, y if use_margin_labels else None)
                    loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.item()))

        val_acc = closed_set_accuracy(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        print(f"[{checkpoint_path.stem}] epoch={epoch:03d} loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {
        "best_val_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "training_status": training_status,
    }


@torch.no_grad()
def closed_set_accuracy(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    pred_all = []
    y_all = []
    for x, y in val_loader:
        logits = model(x.to(device), None)
        pred_all.append(logits.argmax(dim=1).cpu().numpy())
        y_all.append(y.numpy())
    pred = np.concatenate(pred_all)
    y = np.concatenate(y_all)
    return float((pred == y).mean())


@torch.no_grad()
def extract_outputs(
    model: torch.nn.Module,
    dataset: IQDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    feats = []
    logits = []
    labels = []
    for x, y in loader(dataset, batch_size=batch_size, shuffle=False):
        x = x.to(device)
        feats.append(model.embed(x).cpu().numpy())
        logits.append(model(x, None).cpu().numpy())
        labels.append(y.numpy())
    return {
        "features": np.concatenate(feats, axis=0),
        "logits": np.concatenate(logits, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }


def _as_list(value, default: list[str]) -> list:
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _encode_label_names(names: np.ndarray) -> np.ndarray:
    unique_names = sorted(np.unique(names.astype(str)).tolist())
    mapping = {name: idx for idx, name in enumerate(unique_names)}
    return np.asarray([mapping[name] for name in names.astype(str)], dtype=np.int64)


def _balanced_indices(y: np.ndarray, max_per_class: int | None, seed: int) -> np.ndarray:
    if max_per_class is None:
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for label in sorted(np.unique(y).tolist()):
        label_idx = np.where(y == label)[0]
        take = min(int(max_per_class), len(label_idx))
        indices.append(rng.choice(label_idx, size=take, replace=False))
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _protocol_metadata(protocol: LoadedProtocol, output_dir: Path) -> Dict[str, object]:
    summary = protocol.metadata.get("dataset_summary", {})
    split = protocol.metadata.get("split_manifest", {})
    summary = summary if isinstance(summary, dict) else {}
    split = split if isinstance(split, dict) else {}

    known_tx = _as_list(summary.get("known_classes") or split.get("known_classes"), [])
    unknown_tx = _as_list(summary.get("unknown_classes") or split.get("unknown_classes"), [])
    rx_used = _as_list(summary.get("rx_used") or split.get("include_rx"), ["N/A"])
    if not rx_used:
        rx_used = ["N/A"]

    number_of_rx_used = summary.get("num_rx_used")
    if number_of_rx_used is None:
        number_of_rx_used = len(rx_used) if rx_used != ["N/A"] else "N/A"
    rx_mode = "not_applicable"
    if rx_used != ["N/A"]:
        try:
            rx_mode = "fixed" if int(number_of_rx_used) == 1 else "mixed"
        except (TypeError, ValueError):
            rx_mode = "mixed"

    number_of_tx = summary.get("num_tx") or summary.get("num_total_labels")
    if number_of_tx is None:
        number_of_tx = len(known_tx) + len(unknown_tx)

    return {
        "number_of_tx": number_of_tx,
        "number_of_rx_used": number_of_rx_used,
        "rx_mode": rx_mode,
        "rx_used": rx_used,
        "known_tx_list": known_tx,
        "unknown_tx_list": unknown_tx,
        "known_classes": int(protocol.num_known_classes),
        "unknown_classes": int(protocol.num_unknown_classes),
        "train_sample_count": int(len(protocol.train_y)),
        "val_sample_count": int(len(protocol.val_y)),
        "test_known_sample_count": int(len(protocol.test_known_y)),
        "test_unknown_sample_count": int(len(protocol.test_unknown_y)),
        "split_file": summary.get("split_file") or split.get("split_file") or split.get("split_name") or "",
        "output_dir": str(output_dir),
    }


def _softmax_open_set_prediction(
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    known_quantile: float,
    unknown_label: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    val_conf = F.softmax(torch.from_numpy(val_logits), dim=1).numpy().max(axis=1)
    threshold = float(np.quantile(val_conf, known_quantile))
    prob = F.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1).astype(np.int64)
    pred[conf < threshold] = unknown_label
    unknown_score = 1.0 - conf
    return pred, unknown_score, threshold


def _openmax_probabilities(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    query_logits: np.ndarray,
    *,
    alpha_rank: int = 3,
    tail_size: int = 25,
    backend: str = "repo_openmax",
    distance_type: str = "eucos",
) -> tuple[np.ndarray, np.ndarray]:
    calibrator = OpenMaxCalibrator(
        alpha_rank=alpha_rank,
        tail_size=tail_size,
        backend=backend,
        distance_type=distance_type,
    )
    calibrator.fit(
        np.asarray(train_logits, dtype=np.float32),
        np.asarray(train_labels, dtype=np.int64),
        np.asarray(train_predictions, dtype=np.int64),
    )
    result = calibrator.predict(np.asarray(query_logits, dtype=np.float32))
    return (
        np.asarray(result["known_probs"], dtype=np.float64),
        np.asarray(result["unknown_prob"], dtype=np.float64),
    )


def _openmax_open_set_prediction(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    known_quantile: float,
    unknown_label: int,
    *,
    backend: str = "native",
    distance_type: str = "eucl",
) -> tuple[np.ndarray, np.ndarray, float]:
    train_predictions = np.argmax(train_logits, axis=1)
    _, val_unknown = _openmax_probabilities(
        train_logits,
        train_labels,
        train_predictions,
        val_logits,
        backend=backend,
        distance_type=distance_type,
    )
    threshold = float(np.quantile(val_unknown, 1.0 - known_quantile))
    known_probs, unknown_score = _openmax_probabilities(
        train_logits,
        train_labels,
        train_predictions,
        test_logits,
        backend=backend,
        distance_type=distance_type,
    )
    pred = known_probs.argmax(axis=1).astype(np.int64)
    pred[unknown_score >= threshold] = unknown_label
    return pred, unknown_score, threshold


def _arpl_open_set_prediction(
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    known_quantile: float,
    unknown_label: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    val_conf = np.max(val_logits, axis=1)
    threshold = float(np.quantile(val_conf, known_quantile))
    test_conf = np.max(test_logits, axis=1)
    pred = np.argmax(test_logits, axis=1).astype(np.int64)
    pred[test_conf < threshold] = unknown_label
    unknown_score = -test_conf
    return pred, unknown_score, threshold


def _center_open_set_prediction(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    known_quantile: float,
    unknown_label: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    centers = []
    for label in range(unknown_label):
        centers.append(train_features[train_labels == label].mean(axis=0))
    centers = np.stack(centers, axis=0)
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

    val_score = val_features @ centers.T
    threshold = float(np.quantile(val_score.max(axis=1), known_quantile))
    test_score = test_features @ centers.T
    conf = test_score.max(axis=1)
    pred = test_score.argmax(axis=1).astype(np.int64)
    pred[conf < threshold] = unknown_label
    unknown_score = 1.0 - ((conf + 1.0) / 2.0)
    return pred, unknown_score, threshold


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + eps)


def _hyperrsi_gpd_open_set_prediction(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    unknown_label: int,
    *,
    tail_quantile: float = 0.99,
    probability_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_features = _normalize_rows(train_features)
    test_features = _normalize_rows(test_features)
    centers = []
    for label in range(unknown_label):
        cls_features = train_features[train_labels == label]
        if len(cls_features) == 0:
            centers.append(np.zeros(train_features.shape[1], dtype=np.float32))
        else:
            centers.append(cls_features.mean(axis=0))
    centers = _normalize_rows(np.stack(centers, axis=0))

    train_cosine = train_features @ centers.T
    intra_distance = 1.0 - train_cosine[np.arange(len(train_labels)), train_labels]
    tail_start = float(np.quantile(intra_distance, tail_quantile))
    excess_tail = intra_distance[intra_distance >= tail_start] - tail_start
    if len(excess_tail) >= 3 and float(excess_tail.max()) > 1e-8:
        shape, _, scale = genpareto.fit(excess_tail, floc=0.0)
        scale = max(float(scale), 1e-6)
    else:
        shape = 0.0
        scale = max(float(np.std(intra_distance)), float(excess_tail.max(initial=0.0)), 1e-3)

    test_cosine = test_features @ centers.T
    pred = test_cosine.argmax(axis=1).astype(np.int64)
    max_distance = 1.0 - test_cosine[np.arange(len(test_features)), pred]
    excess = np.maximum(max_distance - tail_start, 0.0)
    confidence = np.clip(genpareto.sf(excess, shape, loc=0.0, scale=scale), 0.0, 1.0)
    pred[confidence < probability_threshold] = unknown_label
    unknown_score = 1.0 - confidence
    return pred, unknown_score.astype(np.float32), float(probability_threshold)


def _subdivide_rejected_cache(
    *,
    method_name: str,
    protocol: LoadedProtocol,
    output_dir: Path,
    seed: int,
    test_features: np.ndarray,
    y_true_rejection: np.ndarray,
    open_set_pred: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    true_unknown_cluster_labels: np.ndarray,
    closed_set_pred: np.ndarray,
    backend: str,
    n_prototypes: int,
    train_summary: Dict[str, object],
) -> Dict[str, object]:
    unknown_label = protocol.num_known_classes
    selected_mask = open_set_pred == unknown_label
    true_unknown_mask = y_true_rejection == unknown_label
    selected_features = test_features[selected_mask]
    selected_true_unknown = true_unknown_mask[selected_mask]
    selected_true_labels = true_unknown_cluster_labels[selected_mask]
    n_clusters = int(protocol.num_unknown_classes)

    if len(selected_features) == 0:
        pred = np.zeros((0,), dtype=np.int64)
    elif backend == "openrfi_prototype_grouping":
        pred = _prototype_grouping(selected_features, n_clusters=n_clusters, n_prototypes=n_prototypes, seed=seed)
    elif backend == "closed_set_prediction":
        pred = closed_set_pred[selected_mask].astype(np.int64)
    else:
        raise ValueError(f"Unsupported subdivision backend: {backend}")

    eval_mask = selected_true_unknown
    if int(eval_mask.sum()) > 0 and len(np.unique(selected_true_labels[eval_mask])) > 1:
        metrics = evaluate_clustering(selected_true_labels[eval_mask], pred[eval_mask])
    else:
        metrics = {"nmi": 0.0, "ari": 0.0, "purity": 0.0, "hungarian_accuracy": 0.0}

    if len(pred):
        valid_counts = np.unique(pred, return_counts=True)[1]
        cluster_size_min = int(valid_counts.min())
        cluster_size_max = int(valid_counts.max())
        cluster_size_mean = float(valid_counts.mean())
        num_predicted_clusters = int(len(np.unique(pred)))
    else:
        cluster_size_min = 0
        cluster_size_max = 0
        cluster_size_mean = 0.0
        num_predicted_clusters = 0

    known_centers = []
    for label in sorted(np.unique(train_labels).tolist()):
        known_centers.append(train_features[train_labels == label].mean(axis=0))
    if known_centers and len(selected_features):
        known_centers_arr = np.stack(known_centers, axis=0)
        nearest_known = cdist(selected_features, known_centers_arr).min(axis=1)
        nearest_mean = float(nearest_known.mean())
        nearest_min = float(nearest_known.min())
    else:
        nearest_mean = 0.0
        nearest_min = 0.0

    selected_count = int(selected_mask.sum())
    selected_true_unknown_count = int((selected_mask & true_unknown_mask).sum())
    total_unknown_count = int(true_unknown_mask.sum())
    unknown_cache_precision = float(selected_true_unknown_count / max(selected_count, 1))
    unknown_cache_recall = float(selected_true_unknown_count / max(total_unknown_count, 1))

    result = {
        "dataset": protocol.name,
        "method": method_name,
        "task": "unknown_subdivision",
        "seed": seed,
        "method_detail": backend,
        "clustering_backend": backend,
        "feature_mode": "method_native_output" if backend == "closed_set_prediction" else "method_embedding",
        "resolved_num_clusters": num_predicted_clusters,
        "num_evaluated_unknown": selected_true_unknown_count,
        "num_true_unknown_classes": int(protocol.num_unknown_classes),
        "num_predicted_clusters": num_predicted_clusters,
        "selected_unknown_cache_size": selected_count,
        "uncertain_size": int(total_unknown_count - selected_true_unknown_count),
        "uncertain_ratio": float(1.0 - unknown_cache_recall),
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "cluster_size_mean": cluster_size_mean,
        "nearest_known_proto_distance_mean": nearest_mean,
        "nearest_known_proto_distance_min": nearest_min,
        "unknown_cache_precision": unknown_cache_precision,
        "unknown_cache_recall": unknown_cache_recall,
        "coverage_of_selected_true_unknown": 1.0 if selected_true_unknown_count > 0 else 0.0,
        "coverage_of_total_test_unknown": unknown_cache_recall,
        "suspected_known_noise_size": int(selected_count - selected_true_unknown_count),
        "n_prototypes": n_prototypes if backend == "openrfi_prototype_grouping" else "",
        **_protocol_metadata(protocol, output_dir),
        **train_summary,
        **metrics,
    }
    safe_name = method_name.lower().replace(" ", "_").replace("/", "_")
    (output_dir / f"{safe_name}_{protocol.name}_seed{seed}_subdivision_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _subdivide_world_kplusm(
    *,
    method_name: str,
    protocol: LoadedProtocol,
    output_dir: Path,
    seed: int,
    test_features: np.ndarray,
    y_true_rejection: np.ndarray,
    true_unknown_cluster_labels: np.ndarray,
    n_prototypes: int,
    graph_lambda: float,
    graph_neighbors: int,
    confidence_threshold: float | None,
    train_summary: Dict[str, object],
) -> Dict[str, object]:
    unknown_label = protocol.num_known_classes
    pred, confidence = openrfi_world_prototype_grouping_scores(
        test_features,
        total_num_clusters=protocol.num_known_classes + protocol.num_unknown_classes,
        n_prototypes=n_prototypes,
        seed=seed,
        n_neighbors=graph_neighbors,
        graph_lambda=graph_lambda,
    )
    selected_mask = np.ones(len(test_features), dtype=bool)
    if confidence_threshold is not None:
        selected_mask = confidence >= float(confidence_threshold)

    true_unknown_mask = y_true_rejection == unknown_label
    selected_true_unknown_mask = selected_mask & true_unknown_mask
    selected_true_known_mask = selected_mask & ~true_unknown_mask
    selected_true_labels = true_unknown_cluster_labels

    eval_mask = selected_true_unknown_mask
    if int(eval_mask.sum()) > 0 and len(np.unique(selected_true_labels[eval_mask])) > 1:
        metrics = evaluate_clustering(selected_true_labels[eval_mask], pred[eval_mask])
    else:
        metrics = {"nmi": 0.0, "ari": 0.0, "purity": 0.0, "hungarian_accuracy": 0.0}

    if len(pred):
        valid_counts = np.unique(pred, return_counts=True)[1]
        cluster_size_min = int(valid_counts.min())
        cluster_size_max = int(valid_counts.max())
        cluster_size_mean = float(valid_counts.mean())
        num_predicted_clusters = int(len(np.unique(pred)))
    else:
        cluster_size_min = 0
        cluster_size_max = 0
        cluster_size_mean = 0.0
        num_predicted_clusters = 0

    total_count = int(len(test_features))
    selected_count = int(selected_mask.sum())
    selected_true_unknown_count = int(selected_true_unknown_mask.sum())
    total_unknown_count = int(true_unknown_mask.sum())
    unknown_cache_precision = float(selected_true_unknown_count / max(selected_count, 1))
    unknown_cache_recall = float(selected_true_unknown_count / max(total_unknown_count, 1))

    result = {
        "dataset": protocol.name,
        "method": method_name,
        "task": "unknown_subdivision",
        "seed": seed,
        "method_detail": "openrfi_world_kplusm_grouping",
        "clustering_backend": "openrfi_world_kplusm_grouping",
        "feature_mode": "full_test_world",
        "subdivision_scope": "full_test_world",
        "cluster_budget_mode": "K_plus_M",
        "resolved_num_clusters": int(protocol.num_known_classes + protocol.num_unknown_classes),
        "num_evaluated_unknown": selected_true_unknown_count,
        "num_true_unknown_classes": int(protocol.num_unknown_classes),
        "num_predicted_clusters": num_predicted_clusters,
        "selected_unknown_cache_size": selected_count,
        "uncertain_size": int(total_unknown_count - selected_true_unknown_count),
        "uncertain_ratio": float(1.0 - unknown_cache_recall),
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "cluster_size_mean": cluster_size_mean,
        "nearest_known_proto_distance_mean": 0.0,
        "nearest_known_proto_distance_min": 0.0,
        "unknown_cache_precision": unknown_cache_precision,
        "unknown_cache_recall": unknown_cache_recall,
        "coverage_of_selected_true_unknown": 1.0 if selected_true_unknown_count > 0 else 0.0,
        "coverage_of_total_test_unknown": unknown_cache_recall,
        "suspected_known_noise_size": int(selected_true_known_mask.sum()),
        "n_prototypes": n_prototypes,
        "graph_lambda": float(graph_lambda),
        "graph_neighbors": int(graph_neighbors),
        "confidence_threshold": float(confidence_threshold) if confidence_threshold is not None else "",
        **_protocol_metadata(protocol, output_dir),
        **train_summary,
        **metrics,
    }
    safe_name = method_name.lower().replace(" ", "_").replace("/", "_")
    (output_dir / f"{safe_name}_{protocol.name}_seed{seed}_subdivision_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_open_set_baseline(
    *,
    method_name: str,
    model: torch.nn.Module,
    protocol: LoadedProtocol,
    normalize: str,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    max_train_per_class: int | None,
    max_eval_per_class: int | None,
    score_mode: str,
    known_quantile: float,
    use_margin_labels: bool,
    reuse_checkpoint: bool,
    subdivision_backend: str | None = None,
    n_prototypes: int = 50,
    subdivision_n_prototypes: int | None = None,
    subdivision_graph_neighbors: int = 3,
    subdivision_graph_lambda: float = 1.0,
    subdivision_confidence_threshold: float | None = None,
    confusing_sample_generator: torch.nn.Module | None = None,
    confusing_sample_discriminator: torch.nn.Module | None = None,
    confusing_noise_dim: int = 32,
    confusing_beta: float = 0.1,
    confusing_gan_lr: float = 2e-4,
    openmax_backend: str = "native",
    openmax_distance_type: str = "eucl",
) -> list[Dict[str, object]]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = make_iq_dataset(protocol.train_x, protocol.train_y, normalize, max_train_per_class, seed)
    val_set = make_iq_dataset(protocol.val_x, protocol.val_y, normalize, max_eval_per_class, seed)
    test_known = make_iq_dataset(protocol.test_known_x, protocol.test_known_y, normalize, max_eval_per_class, seed)
    test_unknown_y_reject = np.full(len(protocol.test_unknown_y), protocol.num_known_classes, dtype=np.int64)
    test_unknown = make_iq_dataset(protocol.test_unknown_x, test_unknown_y_reject, normalize, max_eval_per_class, seed)
    known_eval_indices = _balanced_indices(protocol.test_known_y, max_eval_per_class, seed)
    unknown_eval_indices = _balanced_indices(test_unknown_y_reject, max_eval_per_class, seed)
    unknown_cluster_labels = _encode_label_names(protocol.test_unknown_names)
    true_unknown_cluster_labels = np.concatenate(
        [
            np.full(len(known_eval_indices), -1, dtype=np.int64),
            unknown_cluster_labels[unknown_eval_indices],
        ],
        axis=0,
    )

    ckpt = output_dir / f"{method_name}_{protocol.name}_seed{seed}.pt"
    train_summary = train_classifier(
        model,
        train_set,
        val_set,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        use_margin_labels=use_margin_labels,
        checkpoint_path=ckpt,
        reuse_checkpoint=reuse_checkpoint,
        confusing_sample_generator=confusing_sample_generator,
        confusing_sample_discriminator=confusing_sample_discriminator,
        confusing_noise_dim=confusing_noise_dim,
        confusing_beta=confusing_beta,
        confusing_gan_lr=confusing_gan_lr,
    )

    train_out = extract_outputs(model, train_set, batch_size=batch_size, device=device)
    val_out = extract_outputs(model, val_set, batch_size=batch_size, device=device)
    known_out = extract_outputs(model, test_known, batch_size=batch_size, device=device)
    unknown_out = extract_outputs(model, test_unknown, batch_size=batch_size, device=device)

    test_logits = np.concatenate([known_out["logits"], unknown_out["logits"]], axis=0)
    test_features = np.concatenate([known_out["features"], unknown_out["features"]], axis=0)
    y_true = np.concatenate([known_out["labels"], unknown_out["labels"]], axis=0)
    unknown_label = protocol.num_known_classes

    if score_mode == "center_cosine":
        pred, unknown_score, threshold = _center_open_set_prediction(
            train_out["features"],
            train_out["labels"],
            val_out["features"],
            test_features,
            known_quantile,
            unknown_label,
        )
    elif score_mode == "softmax":
        pred, unknown_score, threshold = _softmax_open_set_prediction(
            val_out["logits"],
            test_logits,
            known_quantile,
            unknown_label,
        )
    elif score_mode == "hyperrsi_evt":
        pred, unknown_score, threshold = _hyperrsi_gpd_open_set_prediction(
            train_out["features"],
            train_out["labels"],
            test_features,
            unknown_label,
        )
    elif score_mode == "openmax":
        pred, unknown_score, threshold = _openmax_open_set_prediction(
            train_out["logits"],
            train_out["labels"],
            val_out["logits"],
            test_logits,
            known_quantile,
            unknown_label,
            backend=openmax_backend,
            distance_type=openmax_distance_type,
        )
    elif score_mode == "arpl":
        pred, unknown_score, threshold = _arpl_open_set_prediction(
            val_out["logits"],
            test_logits,
            known_quantile,
            unknown_label,
        )
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    metrics = evaluate_open_set(y_true, pred, unknown_score, unknown_label)
    _save_open_set_predictions(
        output_dir / f"{method_name}_{protocol.name}_seed{seed}_predictions.csv",
        y_true=y_true,
        y_pred=pred,
        unknown_score=unknown_score,
        unknown_label=unknown_label,
    )
    result = {
        "dataset": protocol.name,
        "method": method_name,
        "task": "open_set_rejection",
        "seed": seed,
        "threshold_strategy_used": f"{score_mode}_val_known_quantile",
        "threshold_mode": f"{score_mode}_val_known_quantile",
        "threshold": threshold,
        "threshold_quantile": known_quantile,
        **_protocol_metadata(protocol, output_dir),
        **train_summary,
        **metrics,
    }
    (output_dir / f"{method_name}_{protocol.name}_seed{seed}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [result]
    if subdivision_backend is not None:
        if subdivision_backend == "openrfi_world_kplusm":
            rows.append(
                _subdivide_world_kplusm(
                    method_name=method_name,
                    protocol=protocol,
                    output_dir=output_dir,
                    seed=seed,
                    test_features=test_features,
                    y_true_rejection=y_true,
                    true_unknown_cluster_labels=true_unknown_cluster_labels,
                    n_prototypes=subdivision_n_prototypes if subdivision_n_prototypes is not None else n_prototypes,
                    graph_lambda=subdivision_graph_lambda,
                    graph_neighbors=subdivision_graph_neighbors,
                    confidence_threshold=subdivision_confidence_threshold,
                    train_summary=train_summary,
                )
            )
            return rows
        rows.append(
            _subdivide_rejected_cache(
                method_name=method_name,
                protocol=protocol,
                output_dir=output_dir,
                seed=seed,
                test_features=test_features,
                y_true_rejection=y_true,
                open_set_pred=pred,
                train_features=train_out["features"],
                train_labels=train_out["labels"],
                true_unknown_cluster_labels=true_unknown_cluster_labels,
                closed_set_pred=np.argmax(test_logits, axis=1).astype(np.int64),
                backend=subdivision_backend,
                n_prototypes=n_prototypes,
                train_summary=train_summary,
            )
        )
    return rows


def _prototype_grouping(
    features: np.ndarray,
    n_clusters: int,
    n_prototypes: int,
    seed: int,
) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    proto_count = min(max(n_clusters, n_prototypes), max(n_clusters, len(scaled) // 2))
    if proto_count <= n_clusters:
        return KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(scaled)

    proto_kmeans = KMeans(n_clusters=proto_count, n_init=20, random_state=seed).fit(scaled)
    proto_centers = proto_kmeans.cluster_centers_
    try:
        proto_labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(10, max(1, proto_count - 1)),
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(proto_centers)
    except Exception:
        proto_labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(proto_centers)

    nearest_proto = cdist(scaled, proto_centers).argmin(axis=1)
    return proto_labels[nearest_proto].astype(np.int64)


def run_openrfi_subdivision(
    *,
    model: torch.nn.Module,
    protocol: LoadedProtocol,
    normalize: str,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    max_train_per_class: int | None,
    max_unknown_per_class: int | None,
    n_prototypes: int,
    reuse_checkpoint: bool,
) -> Dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = make_iq_dataset(protocol.train_x, protocol.train_y, normalize, max_train_per_class, seed)
    val_set = make_iq_dataset(protocol.val_x, protocol.val_y, normalize, max_train_per_class, seed)
    true_unknown_y = _encode_label_names(protocol.test_unknown_names)
    unknown_set = make_iq_dataset(protocol.test_unknown_x, true_unknown_y, normalize, max_unknown_per_class, seed)

    ckpt = output_dir / f"openrfi_{protocol.name}_seed{seed}.pt"
    train_summary = train_classifier(
        model,
        train_set,
        val_set,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        use_margin_labels=False,
        checkpoint_path=ckpt,
        reuse_checkpoint=reuse_checkpoint,
    )
    train_out = extract_outputs(model, train_set, batch_size=batch_size, device=device)
    unknown_out = extract_outputs(model, unknown_set, batch_size=batch_size, device=device)
    true_unknown = unknown_out["labels"]
    n_clusters = int(len(np.unique(true_unknown)))
    pred = _prototype_grouping(unknown_out["features"], n_clusters=n_clusters, n_prototypes=n_prototypes, seed=seed)
    metrics = evaluate_clustering(true_unknown, pred)
    cluster_sizes = np.unique(pred, return_counts=True)[1]

    known_centers = []
    for label in sorted(np.unique(train_out["labels"]).tolist()):
        known_centers.append(train_out["features"][train_out["labels"] == label].mean(axis=0))
    known_centers = np.stack(known_centers, axis=0)
    nearest_known = cdist(unknown_out["features"], known_centers).min(axis=1)
    coverage = float(len(true_unknown) / max(len(protocol.test_unknown_y), 1))

    result = {
        "dataset": protocol.name,
        "method": "OpenRFI-style Prototype Grouping",
        "task": "unknown_subdivision",
        "seed": seed,
        "method_detail": "openrfi_style_prototype_grouping",
        "clustering_backend": "prototype_spectral",
        "feature_mode": "embedding",
        "resolved_num_clusters": n_clusters,
        "num_evaluated_unknown": int(len(true_unknown)),
        "num_true_unknown_classes": n_clusters,
        "num_predicted_clusters": int(len(np.unique(pred))),
        "selected_unknown_cache_size": int(len(true_unknown)),
        "uncertain_size": 0,
        "uncertain_ratio": 0.0,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_mean": float(cluster_sizes.mean()),
        "nearest_known_proto_distance_mean": float(nearest_known.mean()),
        "nearest_known_proto_distance_min": float(nearest_known.min()),
        "unknown_cache_precision": 1.0,
        "unknown_cache_recall": coverage,
        "coverage_of_selected_true_unknown": 1.0,
        "coverage_of_total_test_unknown": coverage,
        "suspected_known_noise_size": 0,
        "n_prototypes": n_prototypes,
        **_protocol_metadata(protocol, output_dir),
        **train_summary,
        **metrics,
    }
    (output_dir / f"openrfi_{protocol.name}_seed{seed}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
