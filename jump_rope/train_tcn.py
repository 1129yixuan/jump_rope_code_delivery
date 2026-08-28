"""Train a real PyTorch TCN/1D-CNN residual model on pose time series.

The model predicts actual_count - baseline_routed, then adds it back to the
routed counter output.

Evaluation uses grouped cross-validation by pose JSON, so zones from the same
video stay together in either train or validation.
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


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COUNTABLE_OPTIONS = {0, 1, 8}
COMPETITION_AREA_01_ALIASES = {"competition_area_01", "\u7ade\u8d5b\u533a01"}
COMPETITION_AREA_02_ALIASES = {"competition_area_02", "\u7ade\u8d5b\u533a02"}


@dataclass(frozen=True)
class Sample:
    index: int
    sample_id: str
    pose_json: str
    sequence_npz: Path | None
    actual: float
    baseline_routed: float | None
    area: str
    zone: int
    detection_coverage: float
    duration_sec: float | None
    track_id: int | None = None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def labelclip_path_for_pose(pose_json: Path, labelclip_root: Path) -> Path:
    jump_path = labelclip_root / f"{pose_json.stem}.jump.json"
    if jump_path.is_file():
        return jump_path
    direct = labelclip_root / f"{pose_json.stem}.labelclip"
    if direct.is_file():
        return direct
    nested_jump = labelclip_root / pose_json.parent.name / f"{pose_json.stem}.jump.json"
    if nested_jump.is_file():
        return nested_jump
    return labelclip_root / pose_json.parent.name / f"{pose_json.stem}.labelclip"


def load_labelclip_counts(path: Path) -> dict[int, float]:
    counts: dict[int, float] = defaultdict(float)
    if not path.is_file():
        return counts
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("zones"), dict):
        for zone, items in data["zones"].items():
            try:
                zone_id = int(zone)
            except (TypeError, ValueError):
                continue
            for item in items:
                if is_countable_labelclip_item(item):
                    counts[zone_id] += 1.0
        return counts
    for item in data.get("segments", data.get("intervals", [])):
        if not is_countable_labelclip_item(item):
            continue
        track_ids = normalize_track_ids(item.get("track_id", item.get("trackId")))
        if not track_ids:
            continue
        for track_id in track_ids:
            counts[track_id] += 1.0
    return counts


def is_countable_labelclip_item(item: dict[str, Any]) -> bool:
    if item.get("usable") is False:
        return False
    try:
        option = int(item.get("option", item.get("label")))
    except (TypeError, ValueError):
        return False
    return option in COUNTABLE_OPTIONS


def normalize_track_ids(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    track_ids = []
    for item in values:
        try:
            track_ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return track_ids


def samples_from_pose_jsons(
    pose_json_root: Path | list[Path],
    labelclip_root: Path,
    min_coverage: float,
) -> tuple[dict[str, Any], list[Sample]]:
    pose_roots = pose_json_root if isinstance(pose_json_root, list) else [pose_json_root]
    pose_paths = sorted(path for root in pose_roots for path in root.glob("*.json"))
    if not pose_paths:
        raise FileNotFoundError(f"No pose JSON files found in {pose_roots}")

    samples: list[Sample] = []
    candidate_count = 0
    for pose_path in pose_paths:
        data = json.loads(pose_path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        track_counts = load_labelclip_counts(labelclip_path_for_pose(pose_path, labelclip_root))
        duration_sec = data.get("video", {}).get("duration_sec")
        area = pose_path.parent.name

        track_array = config.get("trackArray", [])
        for item in track_array:
            position = int(item.get("position", 0))
            for track_id_raw in item.get("trackIds", []):
                candidate_count += 1
                track_id = int(track_id_raw)
                coverage = detection_coverage_for_track(data, track_id)
                if coverage < min_coverage:
                    continue
                sample_id = f"{pose_path.stem}_track{track_id}"
                samples.append(
                    Sample(
                        index=len(samples),
                        sample_id=sample_id,
                        pose_json=str(pose_path),
                        sequence_npz=None,
                        actual=float(track_counts.get(track_id, 0.0)),
                        baseline_routed=None,
                        area=area,
                        zone=position,
                        detection_coverage=coverage,
                        duration_sec=float(duration_sec) if is_number(duration_sec) else None,
                        track_id=track_id,
                    )
                )

        if track_array:
            continue

        for item in data.get("zones", []):
            try:
                zone = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            candidate_count += 1
            coverage = detection_coverage_for_zone(data, zone)
            if coverage < min_coverage:
                continue
            sample_id = f"{pose_path.stem}_zone{zone}"
            samples.append(
                Sample(
                    index=len(samples),
                    sample_id=sample_id,
                    pose_json=str(pose_path),
                    sequence_npz=None,
                    actual=float(track_counts.get(zone, 0.0)),
                    baseline_routed=None,
                    area=area,
                    zone=zone,
                    detection_coverage=coverage,
                    duration_sec=float(duration_sec) if is_number(duration_sec) else None,
                    track_id=None,
                )
            )

    manifest = {
        "version": "auto_from_current_pose_json_v2",
        "sample_count": candidate_count,
        "pose_json_root": [str(root) for root in pose_roots],
        "labelclip_root": str(labelclip_root),
        "target_source": "labelclip_or_jump_count",
    }
    return manifest, samples


def load_manifest(
    path: Path | None,
    min_coverage: float,
    pose_json_root: Path | list[Path],
    labelclip_root: Path,
) -> tuple[dict[str, Any], list[Sample]]:
    if path is None or not path.is_file():
        return samples_from_pose_jsons(pose_json_root, labelclip_root, min_coverage)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    project_root = path.resolve().parents[3] if len(path.resolve().parents) >= 4 else path.resolve().parent
    samples: list[Sample] = []
    for index, row in enumerate(manifest["samples"]):
        coverage = float(row.get("detection_coverage", 0.0))
        if coverage < min_coverage:
            continue
        baseline = row.get("baseline_routed")
        if not is_number(baseline):
            baseline = None
        sequence_npz = Path(row["sequence_npz"]) if row.get("sequence_npz") else None
        if sequence_npz is not None and not sequence_npz.is_absolute():
            sequence_npz = project_root / sequence_npz
        samples.append(
            Sample(
                index=index,
                sample_id=row["sample_id"],
                pose_json=row["pose_json"],
                sequence_npz=sequence_npz,
                actual=float(row["actual_count"]),
                baseline_routed=float(baseline) if baseline is not None else None,
                area=row["area"],
                zone=int(row["zone"]),
                detection_coverage=coverage,
                duration_sec=(
                    float(row["duration_sec"])
                    if is_number(row.get("duration_sec"))
                    else None
                ),
                track_id=int(row["track_id"]) if row.get("track_id") is not None else None,
            )
        )
    return manifest, samples


def detection_coverage_for_track(data: dict[str, Any], track_id: int) -> float:
    frames = data.get("frames", [])
    if not frames:
        return 0.0
    detected = 0
    for frame in frames:
        if person_for_track(frame, track_id) is not None:
            detected += 1
    return detected / len(frames)


def person_for_track(frame: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    for person in frame.get("persons", []):
        if int(person.get("track_id", person.get("person_id", -1))) == track_id:
            return person
    return None


def zone_rect(data: dict[str, Any], zone: int) -> tuple[float, float, float, float] | None:
    for item in data.get("zones", []):
        try:
            if int(item.get("id")) != int(zone):
                continue
            return (
                float(item["x1"]),
                float(item["y1"]),
                float(item["x2"]),
                float(item["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def bbox_area(person: dict[str, Any]) -> float:
    bbox = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_zone_overlap(person: dict[str, Any], rect: tuple[float, float, float, float]) -> float:
    bbox = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    zx1, zy1, zx2, zy2 = rect
    return max(0.0, min(x2, zx2) - max(x1, zx1)) * max(0.0, min(y2, zy2) - max(y1, zy1))


def point_in_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = rect
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def person_for_zone(data: dict[str, Any], frame: dict[str, Any], zone: int) -> dict[str, Any] | None:
    rect = zone_rect(data, zone)
    persons = frame.get("persons", [])
    if rect is None or not persons:
        return None

    bottom_center_matches = []
    center_matches = []
    overlap_matches = []
    for person in persons:
        bbox = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bottom_y = y2
        area = bbox_area(person)
        if point_in_rect(center_x, bottom_y, rect):
            bottom_center_matches.append((area, person))
        elif point_in_rect(center_x, center_y, rect):
            center_matches.append((area, person))
        else:
            overlap = bbox_zone_overlap(person, rect)
            if overlap > 0.0:
                overlap_matches.append((overlap, person))

    if bottom_center_matches:
        return max(bottom_center_matches, key=lambda item: item[0])[1]
    if center_matches:
        return max(center_matches, key=lambda item: item[0])[1]
    if overlap_matches:
        return max(overlap_matches, key=lambda item: item[0])[1]
    return None


def detection_coverage_for_zone(data: dict[str, Any], zone: int) -> float:
    frames = data.get("frames", [])
    if not frames:
        return 0.0
    detected = sum(1 for frame in frames if person_for_zone(data, frame, zone) is not None)
    return detected / len(frames)


def pose_json_to_features(
    data: dict[str, Any],
    track_id: int | None,
    zone: int | None = None,
) -> np.ndarray:
    frames = data.get("frames", [])
    width = float(data.get("video", {}).get("width") or 1.0)
    height = float(data.get("video", {}).get("height") or 1.0)
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    feature_dim = 6 + len(KEYPOINT_NAMES) * 3
    rows = np.zeros((len(frames), feature_dim), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        person = None
        if track_id is not None:
            person = person_for_track(frame, track_id)
        elif zone is not None:
            person = person_for_zone(data, frame, zone)
        elif frame.get("persons"):
            person = frame["persons"][0]
        if person is None:
            continue

        bbox = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        rows[frame_index, :6] = [
            ((x1 + x2) / 2.0) / width,
            ((y1 + y2) / 2.0) / height,
            max(0.0, x2 - x1) / width,
            max(0.0, y2 - y1) / height,
            float(person.get("score") or 0.0),
            1.0,
        ]
        keypoints_by_name = {item.get("name"): item for item in person.get("keypoints", [])}
        offset = 6
        for name in KEYPOINT_NAMES:
            point = keypoints_by_name.get(name)
            if point is not None:
                rows[frame_index, offset : offset + 3] = [
                    float(point.get("x") or 0.0) / width,
                    float(point.get("y") or 0.0) / height,
                    float(point.get("score") or 0.0),
                ]
            offset += 3
    deltas = np.diff(rows, axis=0, prepend=rows[:1])
    return np.concatenate([rows, deltas], axis=1).astype(np.float32)


def resample_sequence(features: np.ndarray, length: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected T x F feature matrix, got {features.shape}")
    if features.shape[0] == length:
        return features.astype(np.float32, copy=False)
    if features.shape[0] == 1:
        return np.repeat(features, length, axis=0).astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=features.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    output = np.empty((length, features.shape[1]), dtype=np.float32)
    for column in range(features.shape[1]):
        output[:, column] = np.interp(new_x, old_x, features[:, column])
    return output


def load_sequences(samples: list[Sample], length: int) -> np.ndarray:
    rows = []
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
        rows.append(resample_sequence(features, length))
    return np.stack(rows, axis=0).astype(np.float32)


def metadata_matrix(samples: list[Sample]) -> np.ndarray:
    rows = []
    for sample in samples:
        rows.append(
            [
                sample.baseline_routed or 0.0,
                sample.detection_coverage,
                sample.duration_sec or 0.0,
                sample.zone / 5.0,
                1.0 if sample.area in COMPETITION_AREA_01_ALIASES else 0.0,
                1.0 if sample.area in COMPETITION_AREA_02_ALIASES else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def make_group_folds(samples: list[Sample], fold_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.pose_json].append(index)
    group_items = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -float(np.mean([samples[index].actual for index in item[1]])),
            item[0],
        ),
    )
    fold_targets = [0.0 for _ in range(fold_count)]
    fold_indices: list[list[int]] = [[] for _ in range(fold_count)]
    for _, indices in group_items:
        target_sum = sum(samples[index].actual for index in indices)
        fold_index = min(range(fold_count), key=lambda index: (fold_targets[index], len(fold_indices[index])))
        fold_indices[fold_index].extend(indices)
        fold_targets[fold_index] += target_sum
    all_indices = set(range(len(samples)))
    return [
        (
            np.asarray(sorted(all_indices - set(validation)), dtype=np.int64),
            np.asarray(sorted(validation), dtype=np.int64),
        )
        for validation in fold_indices
    ]


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    abs_error = np.abs(error)
    return {
        "n": int(actual.size),
        "mae": float(abs_error.mean()) if actual.size else math.nan,
        "rmse": float(np.sqrt(np.mean(error**2))) if actual.size else math.nan,
        "bias": float(error.mean()) if actual.size else math.nan,
        "median_abs_error": float(np.median(abs_error)) if actual.size else math.nan,
        "max_abs_error": float(abs_error.max()) if actual.size else math.nan,
        "acc10": float(np.mean(abs_error <= 10.0)) if actual.size else math.nan,
        "acc5": float(np.mean(abs_error <= 5.0)) if actual.size else math.nan,
        "over10": int(np.sum(error > 10.0)),
        "under10": int(np.sum(error < -10.0)),
    }


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        metadata: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
    ):
        self.sequences = torch.from_numpy(sequences[indices]).float()
        self.metadata = torch.from_numpy(metadata[indices]).float()
        self.targets = torch.from_numpy(targets[indices]).float()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.metadata[index], self.targets[index]


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNRegressor(nn.Module):
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
        blocks = []
        for layer in range(layers):
            blocks.append(
                TemporalBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** layer,
                    dropout=dropout,
                )
            )
        self.temporal = nn.Sequential(*blocks)
        head_input = hidden_channels * 2 + (meta_dim if use_metadata else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, sequence: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # sequence: batch x time x feature
        x = sequence.transpose(1, 2)
        x = self.temporal(self.input_projection(x))
        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        if self.use_metadata:
            pooled = torch.cat([pooled, metadata], dim=1)
        return self.head(pooled).squeeze(1)


def standardize_by_train(
    sequences: np.ndarray,
    metadata: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    seq_mean = sequences[train_indices].mean(axis=(0, 1), keepdims=True)
    seq_std = sequences[train_indices].std(axis=(0, 1), keepdims=True)
    seq_std[seq_std < 1e-6] = 1.0
    meta_mean = metadata[train_indices].mean(axis=0, keepdims=True)
    meta_std = metadata[train_indices].std(axis=0, keepdims=True)
    meta_std[meta_std < 1e-6] = 1.0
    target_mean = float(targets[train_indices].mean())
    target_std = float(targets[train_indices].std())
    if target_std < 1e-6:
        target_std = 1.0
    return (
        ((sequences - seq_mean) / seq_std).astype(np.float32),
        ((metadata - meta_mean) / meta_std).astype(np.float32),
        ((targets - target_mean) / target_std).astype(np.float32),
        {
            "target_mean": target_mean,
            "target_std": target_std,
        },
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_fold(
    *,
    fold_index: int,
    sequences: np.ndarray,
    metadata: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    x, meta, y, scaler = standardize_by_train(sequences, metadata, targets, train_indices)
    train_ds = PoseSequenceDataset(x, meta, y, train_indices)
    val_ds = PoseSequenceDataset(x, meta, y, validation_indices)
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold_index)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = TCNRegressor(
        feature_dim=sequences.shape[2],
        meta_dim=metadata.shape[1],
        hidden_channels=args.hidden_channels,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_metadata=args.use_metadata,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_meta, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_meta = batch_meta.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x, batch_meta), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_meta, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_meta = batch_meta.to(device)
                batch_y = batch_y.to(device)
                val_losses.append(float(loss_fn(model(batch_x, batch_meta), batch_y).detach().cpu()))
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

    predictions = np.empty((len(validation_indices),), dtype=np.float32)
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch_x, batch_meta, _batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_meta = batch_meta.to(device)
            pred = model(batch_x, batch_meta).detach().cpu().numpy()
            size = pred.shape[0]
            predictions[offset : offset + size] = pred
            offset += size
    predictions = predictions * scaler["target_std"] + scaler["target_mean"]
    return predictions, {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "history": history,
    }


def write_predictions(
    path: Path,
    samples: list[Sample],
    predictions: np.ndarray,
    targets: np.ndarray,
    final_counts: np.ndarray,
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
                "target_value",
                "predicted_target",
                "final_count",
                "final_error",
                "detection_coverage",
                "duration_sec",
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
                    "target_value": float(targets[index]),
                    "predicted_target": float(predictions[index]),
                    "final_count": float(final_counts[index]),
                    "final_error": float(final_counts[index] - sample.actual),
                    "detection_coverage": sample.detection_coverage,
                    "duration_sec": sample.duration_sec,
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
    samples = [sample for sample in samples if sample.baseline_routed is not None]
    if len(samples) < args.folds:
        raise ValueError(f"Need at least {args.folds} samples, got {len(samples)}")

    sequences = load_sequences(samples, args.length)
    metadata = metadata_matrix(samples)
    actual = np.asarray([sample.actual for sample in samples], dtype=np.float32)
    baseline = np.asarray([sample.baseline_routed or 0.0 for sample in samples], dtype=np.float32)
    targets = actual - baseline
    folds = make_group_folds(samples, args.folds)
    device = choose_device(args.device)

    predictions = np.empty((len(samples),), dtype=np.float32)
    fold_reports = []
    for fold_index, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_predictions, fold_report = train_fold(
            fold_index=fold_index,
            sequences=sequences,
            metadata=metadata,
            targets=targets,
            train_indices=train_indices,
            validation_indices=validation_indices,
            args=args,
            device=device,
        )
        predictions[validation_indices] = fold_predictions
        fold_report["validation_sample_count"] = int(validation_indices.size)
        fold_report["validation_pose_json"] = sorted({samples[index].pose_json for index in validation_indices})
        fold_reports.append(fold_report)
        if args.verbose:
            final = baseline[validation_indices] + fold_predictions
            metrics = metric_summary(actual[validation_indices], final)
            print(
                f"fold={fold_index} best_epoch={fold_report['best_epoch']} "
                f"mae={metrics['mae']:.3f} acc10={metrics['acc10']:.3f}"
            )

    final_counts = baseline + predictions
    baseline_mask = np.asarray([sample.baseline_routed is not None for sample in samples], dtype=np.bool_)
    metrics = {
        "tcn": metric_summary(actual, final_counts),
        "baseline_routed": metric_summary(actual[baseline_mask], baseline[baseline_mask]),
        "by_fold": [
            {
                "fold": fold_index,
                "tcn": metric_summary(
                    actual[validation],
                    final_counts[validation],
                ),
                "baseline_routed": metric_summary(
                    actual[validation][baseline_mask[validation]],
                    baseline[validation][baseline_mask[validation]],
                ),
            }
            for fold_index, (_train, validation) in enumerate(folds, start=1)
        ],
    }

    output = {
        "experiment": "tcn_pose_sequence_v1",
        "manifest": str(args.manifest) if args.manifest else None,
        "pose_json_root": [str(path) for path in args.pose_json_root],
        "labelclip_root": str(args.labelclip_root),
        "dataset_version": manifest.get("version"),
        "sample_count": len(samples),
        "excluded_by_min_coverage": int(manifest.get("sample_count", coverage_sample_count) - coverage_sample_count),
        "target": "residual",
        "resample_length": args.length,
        "sequence_shape": list(sequences.shape),
        "metadata_dim": int(metadata.shape[1]),
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
            "use_metadata": args.use_metadata,
            "huber_beta": args.huber_beta,
        },
        "metrics": metrics,
        "fold_reports": fold_reports,
        "output_dir": str(args.out),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions(args.out / "predictions.csv", samples, predictions, targets, final_counts, folds)
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
    parser.add_argument("--labelclip-root", type=Path, default=here / "labelclip_output")
    parser.add_argument("--out", type=Path, default=here / "outputs" / "tcn_pose_sequence_v1")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--use-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    metrics = result["metrics"]
    print(f"samples={result['sample_count']}")
    print(f"target={result['target']}")
    print(f"device={result['device']}")
    print(
        "tcn: "
        f"mae={metrics['tcn']['mae']:.3f} "
        f"acc10={metrics['tcn']['acc10']:.3f} "
        f"acc5={metrics['tcn']['acc5']:.3f} "
        f"bias={metrics['tcn']['bias']:.3f}"
    )
    print(
        "baseline_routed: "
        f"mae={metrics['baseline_routed']['mae']:.3f} "
        f"acc10={metrics['baseline_routed']['acc10']:.3f} "
        f"acc5={metrics['baseline_routed']['acc5']:.3f} "
        f"bias={metrics['baseline_routed']['bias']:.3f}"
    )
    print(f"output={Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
