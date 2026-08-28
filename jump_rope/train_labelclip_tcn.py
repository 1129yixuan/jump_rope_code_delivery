"""Train a TCN from per-jump intervals and evaluate peak-count accuracy.

The model learns a frame-level jump heatmap from intervals, then converts the
predicted heatmap into a count by selecting local probability peaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependency. Please install dependencies first:\n"
        "  python3 -m pip install numpy torch"
    ) from exc

from train_tcn import (
    Sample,
    TemporalBlock,
    choose_device,
    is_countable_labelclip_item,
    load_manifest,
    make_group_folds,
    metadata_matrix,
    metric_summary,
    normalize_track_ids,
    pose_json_to_features,
    resample_sequence,
)


@dataclass(frozen=True)
class LabelclipTarget:
    heatmap: np.ndarray
    count: float
    intervals: tuple[tuple[int, int], ...]


def load_labelclip_intervals(path: Path, sample: Sample, source: str | None) -> tuple[tuple[int, int], ...]:
    if not path.is_file():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    intervals = []
    if isinstance(data.get("zones"), dict):
        for item in data["zones"].get(str(sample.zone), []):
            if not is_countable_labelclip_item(item):
                continue
            start = int(item["start"])
            end = int(item["end"])
            intervals.append((min(start, end), max(start, end)))
        return tuple(sorted(intervals))

    for item in data.get("segments", data.get("intervals", [])):
        if not is_countable_labelclip_item(item):
            continue
        if sample.track_id is not None:
            track_ids = normalize_track_ids(item.get("track_id", item.get("trackId")))
            if int(sample.track_id) not in track_ids:
                continue
        elif item.get("zone") is not None and int(item.get("zone", 0)) != int(sample.zone):
            continue
        if source and str(item.get("source", "")) != source:
            continue
        interval = item.get("interval")
        if not isinstance(interval, list) or len(interval) < 2:
            continue
        start = int(interval[0])
        end = int(interval[1])
        intervals.append((min(start, end), max(start, end)))
    return tuple(sorted(intervals))


def labelclip_path_for(sample: Sample, output_root: Path) -> Path:
    pose_path = Path(sample.pose_json)
    jump = output_root / f"{pose_path.stem}.jump.json"
    if jump.is_file():
        return jump
    direct = output_root / f"{pose_path.stem}.labelclip"
    if direct.is_file():
        return direct
    nested_jump = output_root / sample.area / f"{pose_path.stem}.jump.json"
    if nested_jump.is_file():
        return nested_jump
    return output_root / sample.area / f"{pose_path.stem}.labelclip"


def build_heatmap(
    intervals: tuple[tuple[int, int], ...],
    *,
    original_length: int,
    output_length: int,
    sigma_bins: float,
) -> np.ndarray:
    heatmap = np.zeros((output_length,), dtype=np.float32)
    if original_length <= 1:
        return heatmap
    grid = np.arange(output_length, dtype=np.float32)
    scale = (output_length - 1) / max(1, original_length - 1)
    for start, end in intervals:
        center = ((start + end) / 2.0) * scale
        values = np.exp(-0.5 * ((grid - center) / max(sigma_bins, 1e-6)) ** 2)
        heatmap = np.maximum(heatmap, values.astype(np.float32))
    return np.clip(heatmap, 0.0, 1.0)


def load_sequences_and_targets(
    samples: list[Sample],
    *,
    length: int,
    labelclip_root: Path,
    label_source: str | None,
    sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[tuple[int, int], ...]]]:
    sequences = []
    heatmaps = []
    counts = []
    intervals_by_sample = []
    pose_cache: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if sample.sequence_npz is not None and sample.sequence_npz.is_file():
            with np.load(sample.sequence_npz) as data:
                features = data["features"].astype(np.float32)
        else:
            pose_path = str(Path(sample.pose_json))
            if pose_path not in pose_cache:
                pose_cache[pose_path] = json.loads(Path(pose_path).read_text(encoding="utf-8"))
            features = pose_json_to_features(pose_cache[pose_path], sample.track_id, sample.zone)
        intervals = load_labelclip_intervals(
            labelclip_path_for(sample, labelclip_root),
            sample,
            label_source,
        )
        sequences.append(resample_sequence(features, length))
        heatmaps.append(
            build_heatmap(
                intervals,
                original_length=features.shape[0],
                output_length=length,
                sigma_bins=sigma_bins,
            )
        )
        counts.append(float(len(intervals)))
        intervals_by_sample.append(intervals)
    return (
        np.stack(sequences, axis=0).astype(np.float32),
        np.stack(heatmaps, axis=0).astype(np.float32),
        np.asarray(counts, dtype=np.float32),
        intervals_by_sample,
    )


class HeatmapDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        metadata: np.ndarray,
        heatmaps: np.ndarray,
        counts: np.ndarray,
        indices: np.ndarray,
    ):
        self.sequences = torch.from_numpy(sequences[indices]).float()
        self.metadata = torch.from_numpy(metadata[indices]).float()
        self.heatmaps = torch.from_numpy(heatmaps[indices]).float()
        self.counts = torch.from_numpy(counts[indices]).float()

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    def __getitem__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[self._index], self.metadata[self._index], self.heatmaps[self._index], self.counts[self._index]

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [self[index] for index in indices]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[no-redef]
        return self.sequences[index], self.metadata[index], self.heatmaps[index], self.counts[index]


class SequenceHeatmapTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        meta_dim: int,
        hidden_channels: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        use_metadata: bool,
    ):
        super().__init__()
        self.use_metadata = use_metadata
        self.input_projection = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** layer,
                    dropout=dropout,
                )
                for layer in range(layers)
            ]
        )
        self.meta_projection = nn.Linear(meta_dim, hidden_channels) if use_metadata else None
        self.output = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def encode(self, sequence: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        x = self.temporal(self.input_projection(x))
        if self.meta_projection is not None:
            meta = self.meta_projection(metadata).unsqueeze(-1)
            x = x + meta
        return x

    def forward(self, sequence: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x = self.encode(sequence, metadata)
        return self.output(x).squeeze(1)


def standardize_by_train(
    sequences: np.ndarray,
    metadata: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seq_mean = sequences[train_indices].mean(axis=(0, 1), keepdims=True)
    seq_std = sequences[train_indices].std(axis=(0, 1), keepdims=True)
    seq_std[seq_std < 1e-6] = 1.0
    meta_mean = metadata[train_indices].mean(axis=0, keepdims=True)
    meta_std = metadata[train_indices].std(axis=0, keepdims=True)
    meta_std[meta_std < 1e-6] = 1.0
    return (
        ((sequences - seq_mean) / seq_std).astype(np.float32),
        ((metadata - meta_mean) / meta_std).astype(np.float32),
    )


def peak_count_one(probabilities: np.ndarray, threshold: float, min_distance: int) -> int:
    candidates = []
    for index in range(1, len(probabilities) - 1):
        value = probabilities[index]
        if value < threshold:
            continue
        if value >= probabilities[index - 1] and value > probabilities[index + 1]:
            candidates.append((float(value), index))
    selected: list[int] = []
    for _value, index in sorted(candidates, reverse=True):
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return len(selected)


def peak_counts(probabilities: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    return np.asarray(
        [peak_count_one(row, threshold, min_distance) for row in probabilities],
        dtype=np.float32,
    )


def tune_peak_params(
    probabilities: np.ndarray,
    target_counts: np.ndarray,
    thresholds: list[float],
    min_distances: list[int],
) -> dict[str, float | int]:
    best = None
    for threshold in thresholds:
        for min_distance in min_distances:
            counts = peak_counts(probabilities, threshold, min_distance)
            mae = float(np.mean(np.abs(counts - target_counts)))
            candidate = {
                "threshold": float(threshold),
                "min_distance": int(min_distance),
                "train_labelclip_mae": mae,
            }
            if best is None or mae < best["train_labelclip_mae"]:
                best = candidate
    assert best is not None
    return best


def predict_probabilities(
    model: nn.Module,
    dataset: HeatmapDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_meta, _batch_heatmap, _batch_count in loader:
            logits = model(batch_x.to(device), batch_meta.to(device))
            rows.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def make_model(
    *,
    feature_dim: int,
    meta_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> SequenceHeatmapTCN:
    return SequenceHeatmapTCN(
        feature_dim=feature_dim,
        meta_dim=meta_dim,
        hidden_channels=args.hidden_channels,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_metadata=args.use_metadata,
    ).to(device)


def train_fold(
    *,
    fold_index: int,
    sequences: np.ndarray,
    metadata: np.ndarray,
    heatmaps: np.ndarray,
    label_counts: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    x, meta = standardize_by_train(sequences, metadata, train_indices)
    train_ds = HeatmapDataset(x, meta, heatmaps, label_counts, train_indices)
    val_ds = HeatmapDataset(x, meta, heatmaps, label_counts, validation_indices)
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(
        feature_dim=sequences.shape[2],
        meta_dim=metadata.shape[1],
        args=args,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positive_fraction = float(max(heatmaps[train_indices].mean(), 1e-4))
    pos_weight = min(args.max_pos_weight, max(1.0, (1.0 - positive_fraction) / positive_fraction))
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    count_loss = nn.SmoothL1Loss(beta=args.count_huber_beta)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_meta, batch_heatmap, batch_count in train_loader:
            batch_x = batch_x.to(device)
            batch_meta = batch_meta.to(device)
            batch_heatmap = batch_heatmap.to(device)
            batch_count = batch_count.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_meta)
            heatmap_loss = bce_loss(logits, batch_heatmap)
            probs = torch.sigmoid(logits)
            raw_count = probs.sum(dim=1) / (math.sqrt(2.0 * math.pi) * args.sigma_bins)
            scaled_pred = raw_count / args.count_scale
            scaled_target = batch_count / args.count_scale
            loss = heatmap_loss + args.count_loss_weight * count_loss(scaled_pred, scaled_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_meta, batch_heatmap, batch_count in val_loader:
                batch_x = batch_x.to(device)
                batch_meta = batch_meta.to(device)
                batch_heatmap = batch_heatmap.to(device)
                batch_count = batch_count.to(device)
                logits = model(batch_x, batch_meta)
                heatmap_loss = bce_loss(logits, batch_heatmap)
                probs = torch.sigmoid(logits)
                raw_count = probs.sum(dim=1) / (math.sqrt(2.0 * math.pi) * args.sigma_bins)
                scaled_pred = raw_count / args.count_scale
                scaled_target = batch_count / args.count_scale
                loss = heatmap_loss + args.count_loss_weight * count_loss(scaled_pred, scaled_target)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else math.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss + args.min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_probs = predict_probabilities(model, train_ds, args.batch_size, device)
    val_probs = predict_probabilities(model, val_ds, args.batch_size, device)

    peak_params = tune_peak_params(
        train_probs,
        label_counts[train_indices],
        thresholds=[float(value) for value in np.linspace(args.peak_threshold_min, args.peak_threshold_max, args.peak_threshold_steps)],
        min_distances=[int(value) for value in args.peak_min_distances.split(",")],
    )
    val_peak = peak_counts(
        val_probs,
        float(peak_params["threshold"]),
        int(peak_params["min_distance"]),
    )

    return val_peak, {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "pos_weight": pos_weight,
        "peak_params": peak_params,
        "train_peak_labelclip_mae": float(peak_params["train_labelclip_mae"]),
        "history": history,
    }


def train_final_model(
    *,
    sequences: np.ndarray,
    metadata: np.ndarray,
    heatmaps: np.ndarray,
    label_counts: np.ndarray,
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    all_indices = np.arange(len(samples), dtype=np.int64)
    seq_mean = sequences.mean(axis=(0, 1), keepdims=True)
    seq_std = sequences.std(axis=(0, 1), keepdims=True)
    seq_std[seq_std < 1e-6] = 1.0
    meta_mean = metadata.mean(axis=0, keepdims=True)
    meta_std = metadata.std(axis=0, keepdims=True)
    meta_std[meta_std < 1e-6] = 1.0
    x = ((sequences - seq_mean) / seq_std).astype(np.float32)
    meta = ((metadata - meta_mean) / meta_std).astype(np.float32)

    dataset = HeatmapDataset(x, meta, heatmaps, label_counts, all_indices)
    generator = torch.Generator()
    generator.manual_seed(args.seed + 999)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = make_model(
        feature_dim=sequences.shape[2],
        meta_dim=metadata.shape[1],
        args=args,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positive_fraction = float(max(heatmaps.mean(), 1e-4))
    pos_weight = min(args.max_pos_weight, max(1.0, (1.0 - positive_fraction) / positive_fraction))
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    count_loss = nn.SmoothL1Loss(beta=args.count_huber_beta)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_meta, batch_heatmap, batch_count in loader:
            batch_x = batch_x.to(device)
            batch_meta = batch_meta.to(device)
            batch_heatmap = batch_heatmap.to(device)
            batch_count = batch_count.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_meta)
            heatmap_loss = bce_loss(logits, batch_heatmap)
            probs = torch.sigmoid(logits)
            raw_count = probs.sum(dim=1) / (math.sqrt(2.0 * math.pi) * args.sigma_bins)
            scaled_pred = raw_count / args.count_scale
            scaled_target = batch_count / args.count_scale
            loss = heatmap_loss + args.count_loss_weight * count_loss(scaled_pred, scaled_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan})

    probabilities = predict_probabilities(model, dataset, args.batch_size, device)
    peak_params = tune_peak_params(
        probabilities,
        label_counts,
        thresholds=[float(value) for value in np.linspace(args.peak_threshold_min, args.peak_threshold_max, args.peak_threshold_steps)],
        min_distances=[int(value) for value in args.peak_min_distances.split(",")],
    )
    peak_predictions = peak_counts(
        probabilities,
        float(peak_params["threshold"]),
        int(peak_params["min_distance"]),
    )
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "feature_dim": int(sequences.shape[2]),
        "meta_dim": int(metadata.shape[1]),
        "sequence_length": int(args.length),
        "sigma_bins": float(args.sigma_bins),
        "model_config": {
            "hidden_channels": int(args.hidden_channels),
            "layers": int(args.layers),
            "kernel_size": int(args.kernel_size),
            "dropout": float(args.dropout),
            "use_metadata": bool(args.use_metadata),
        },
        "normalization": {
            "seq_mean": seq_mean.astype(np.float32),
            "seq_std": seq_std.astype(np.float32),
            "meta_mean": meta_mean.astype(np.float32),
            "meta_std": meta_std.astype(np.float32),
        },
        "peak_params": peak_params,
        "metadata": {
            "sample_count": len(samples),
            "epochs": int(epochs),
            "history": history,
            "train_peak_metrics": metric_summary(label_counts, peak_predictions),
        },
    }
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.save_model)
    return {
        "path": str(args.save_model),
        "epochs": int(epochs),
        "peak_params": peak_params,
        "train_peak_metrics": checkpoint["metadata"]["train_peak_metrics"],
    }


def write_predictions(
    path: Path,
    samples: list[Sample],
    label_counts: np.ndarray,
    peak_predictions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    fold_by_index: dict[int, int] = {}
    for fold_index, (_train, validation) in enumerate(folds, start=1):
        for index in validation:
            fold_by_index[int(index)] = fold_index
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "fold",
                "area",
                "zone",
                "pose_json",
                "actual_count",
                "baseline_routed",
                "labelclip_count",
                "peak_count",
                "peak_actual_error",
                "peak_labelclip_error",
                "detection_coverage",
            ],
        )
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "fold": fold_by_index[index],
                    "area": sample.area,
                    "zone": sample.zone,
                    "pose_json": sample.pose_json,
                    "actual_count": sample.actual,
                    "baseline_routed": sample.baseline_routed,
                    "labelclip_count": float(label_counts[index]),
                    "peak_count": float(peak_predictions[index]),
                    "peak_actual_error": float(peak_predictions[index] - sample.actual),
                    "peak_labelclip_error": float(peak_predictions[index] - label_counts[index]),
                    "detection_coverage": sample.detection_coverage,
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest, samples = load_manifest(
        args.manifest,
        args.min_coverage,
        args.pose_json_root,
        args.labelclip_root,
    )
    coverage_sample_count = len(samples)
    sequences, heatmaps, label_counts, intervals = load_sequences_and_targets(
        samples,
        length=args.length,
        labelclip_root=args.labelclip_root,
        label_source=args.label_source,
        sigma_bins=args.sigma_bins,
    )

    keep = label_counts >= args.min_labelclip_count
    samples = [sample for sample, keep_item in zip(samples, keep) if keep_item]
    sequences = sequences[keep]
    heatmaps = heatmaps[keep]
    label_counts = label_counts[keep]
    intervals = [item for item, keep_item in zip(intervals, keep) if keep_item]
    metadata = metadata_matrix(samples)
    actual = np.asarray([sample.actual for sample in samples], dtype=np.float32)
    baseline_mask = np.asarray([sample.baseline_routed is not None for sample in samples], dtype=np.bool_)
    baseline = np.asarray([sample.baseline_routed or 0.0 for sample in samples], dtype=np.float32)

    folds = make_group_folds(samples, args.folds)
    device = choose_device(args.device)
    peak_predictions = np.empty((len(samples),), dtype=np.float32)
    fold_reports = []

    for fold_index, (train_indices, validation_indices) in enumerate(folds, start=1):
        peak, report = train_fold(
            fold_index=fold_index,
            sequences=sequences,
            metadata=metadata,
            heatmaps=heatmaps,
            label_counts=label_counts,
            train_indices=train_indices,
            validation_indices=validation_indices,
            args=args,
            device=device,
        )
        peak_predictions[validation_indices] = peak
        report["validation_sample_count"] = int(validation_indices.size)
        fold_reports.append(report)
        if args.verbose:
            print(
                f"fold={fold_index} best_epoch={report['best_epoch']} "
                f"peak_actual_mae={metric_summary(actual[validation_indices], peak)['mae']:.3f}"
            )

    metrics = {
        "labelclip_vs_actual": metric_summary(actual, label_counts),
        "peak_vs_actual": metric_summary(actual, peak_predictions),
        "peak_vs_labelclip": metric_summary(label_counts, peak_predictions),
        "baseline_routed": metric_summary(actual[baseline_mask], baseline[baseline_mask]),
    }
    output = {
        "experiment": "labelclip_heatmap_tcn_v1",
        "manifest": str(args.manifest) if args.manifest else None,
        "pose_json_root": [str(path) for path in args.pose_json_root],
        "labelclip_root": str(args.labelclip_root),
        "label_source": args.label_source,
        "sample_count": len(samples),
        "manifest_sample_count": manifest.get("sample_count"),
        "excluded_by_min_coverage": int(manifest.get("sample_count", coverage_sample_count) - coverage_sample_count),
        "excluded_by_min_labelclip_count": int(np.sum(~keep)),
        "sequence_shape": list(sequences.shape),
        "resample_length": args.length,
        "sigma_bins": args.sigma_bins,
        "fold_count": args.folds,
        "split": "grouped_by_pose_json",
        "device": str(device),
        "hyperparameters": {
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "count_loss_weight": args.count_loss_weight,
            "use_metadata": args.use_metadata,
        },
        "metrics": metrics,
        "fold_reports": fold_reports,
        "output_dir": str(args.out),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_model is not None:
        best_epochs = [int(report["best_epoch"]) for report in fold_reports if int(report["best_epoch"]) > 0]
        final_epochs = args.final_epochs or (int(round(float(np.median(best_epochs)))) if best_epochs else args.epochs)
        final_model_report = train_final_model(
            sequences=sequences,
            metadata=metadata,
            heatmaps=heatmaps,
            label_counts=label_counts,
            samples=samples,
            args=args,
            device=device,
            epochs=final_epochs,
        )
        output["final_model"] = final_model_report
    (args.out / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(
        args.out / "predictions.csv",
        samples,
        label_counts,
        peak_predictions,
        folds,
    )
    return output


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional old-format manifest. If omitted, samples are built from --pose-json-root.",
    )
    parser.add_argument("--pose-json-root", type=Path, nargs="+", default=[here / "json_output"])
    parser.add_argument(
        "--labelclip-root",
        type=Path,
        default=here / "labelclip_output",
    )
    parser.add_argument("--out", type=Path, default=here / "outputs" / "labelclip_tcn_v1")
    parser.add_argument("--label-source", default=None)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--sigma-bins", type=float, default=1.6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-labelclip-count", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.15)
    parser.add_argument("--count-huber-beta", type=float, default=0.2)
    parser.add_argument("--count-scale", type=float, default=100.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--peak-threshold-min", type=float, default=0.15)
    parser.add_argument("--peak-threshold-max", type=float, default=0.75)
    parser.add_argument("--peak-threshold-steps", type=int, default=13)
    parser.add_argument("--peak-min-distances", default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    metrics = result["metrics"]
    print(f"samples={result['sample_count']}")
    print(f"device={result['device']}")
    print(
        "labelclip_vs_actual: "
        f"mae={metrics['labelclip_vs_actual']['mae']:.3f} "
        f"acc10={metrics['labelclip_vs_actual']['acc10']:.3f}"
    )
    print(
        "peak_vs_actual: "
        f"mae={metrics['peak_vs_actual']['mae']:.3f} "
        f"acc10={metrics['peak_vs_actual']['acc10']:.3f}"
    )
    print(
        "baseline_routed: "
        f"mae={metrics['baseline_routed']['mae']:.3f} "
        f"acc10={metrics['baseline_routed']['acc10']:.3f}"
    )
    if "final_model" in result:
        final_model = result["final_model"]
        print(f"saved_model={Path(final_model['path']).resolve()}")
        print(
            "final_train_peak: "
            f"mae={final_model['train_peak_metrics']['mae']:.3f} "
            f"acc10={final_model['train_peak_metrics']['acc10']:.3f}"
        )
    print(f"output={Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
