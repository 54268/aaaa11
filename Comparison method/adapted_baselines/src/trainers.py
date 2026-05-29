from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_io import IQDataset, LoadedProtocol, make_iq_dataset
from metrics import evaluate_clustering, evaluate_open_set


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
) -> Dict[str, object]:
    model.to(device)
    train_loader = loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = loader(val_set, batch_size=batch_size, shuffle=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_checkpoint and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        val_acc = closed_set_accuracy(model, val_loader, device)
        return {
            "best_val_acc": float(val_acc),
            "best_epoch": 0,
            "training_status": "reused_checkpoint",
        }

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, y if use_margin_labels else None)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
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
        "training_status": "trained",
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
) -> Dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = make_iq_dataset(protocol.train_x, protocol.train_y, normalize, max_train_per_class, seed)
    val_set = make_iq_dataset(protocol.val_x, protocol.val_y, normalize, max_eval_per_class, seed)
    test_known = make_iq_dataset(protocol.test_known_x, protocol.test_known_y, normalize, max_eval_per_class, seed)
    test_unknown_y_reject = np.full(len(protocol.test_unknown_y), protocol.num_known_classes, dtype=np.int64)
    test_unknown = make_iq_dataset(protocol.test_unknown_x, test_unknown_y_reject, normalize, max_eval_per_class, seed)

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
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    metrics = evaluate_open_set(y_true, pred, unknown_score, unknown_label)
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
    return result


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
