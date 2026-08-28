from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing prediction dependency. Please install dependencies first:\n"
        "  python -m pip install numpy torch"
    ) from exc

import train_tcn
from train_labelclip_tcn import SequenceHeatmapTCN
from train_tcn import Sample, choose_device, metadata_matrix, pose_json_to_features, resample_sequence


BASE_KEYPOINT_NAMES = tuple(train_tcn.KEYPOINT_NAMES[:17])


# ===== Preprocessing helpers =====


def configure_keypoints_for_checkpoint(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    """Select the feature schema expected by the loaded 17kp or 19kp model."""
    feature_dim = int(checkpoint["feature_dim"])
    if feature_dim == 114:
        keypoint_names = BASE_KEYPOINT_NAMES
    elif feature_dim == 126:
        keypoint_names = BASE_KEYPOINT_NAMES + ("extra_17", "extra_18")
    else:
        raise ValueError(
            f"Unsupported checkpoint feature_dim={feature_dim}; "
            "expected 114 (17 keypoints) or 126 (19 keypoints)."
        )

    expected_feature_dim = 2 * (6 + len(keypoint_names) * 3)
    if expected_feature_dim != feature_dim:
        raise RuntimeError(
            f"Feature schema mismatch: {len(keypoint_names)} keypoints produce "
            f"{expected_feature_dim} features, but checkpoint expects {feature_dim}."
        )

    # pose_json_to_features() resolves KEYPOINT_NAMES from the train_tcn module
    # at call time, so updating the list also affects the function imported above.
    train_tcn.KEYPOINT_NAMES[:] = keypoint_names
    return keypoint_names


@dataclass(frozen=True)
class PredictionSample:
    sample: Sample
    pose_data: dict[str, Any]
    track_ids: tuple[int, ...]
    track_label: str


def track_label(track_ids: tuple[int, ...]) -> str:
    return "+".join(str(track_id) for track_id in track_ids)


def normalize_track_id_group(value: Any) -> tuple[int, ...]:
    return tuple(train_tcn.normalize_track_ids(value))


def person_for_any_track(frame: dict[str, Any], track_ids: tuple[int, ...]) -> dict[str, Any] | None:
    for track_id in track_ids:
        for person in frame.get("persons", []):
            try:
                person_track_id = int(person.get("track_id", person.get("person_id", -1)))
            except (TypeError, ValueError):
                continue
            if person_track_id == track_id:
                return person
    return None


def pose_data_for_track_group(data: dict[str, Any], track_ids: tuple[int, ...]) -> dict[str, Any]:
    if len(track_ids) <= 1:
        return data

    canonical_track_id = track_ids[0]
    selected = dict(data)
    selected_frames = []
    for frame in data.get("frames", []):
        output_frame = dict(frame)
        merged_person = person_for_any_track(frame, track_ids)
        persons = []
        if merged_person is not None:
            merged_person = dict(merged_person)
            merged_person["person_id"] = canonical_track_id
            merged_person["track_id"] = canonical_track_id
            merged_person["merged_track_ids"] = list(track_ids)
            persons.append(merged_person)
        for person in frame.get("persons", []):
            try:
                person_track_id = int(person.get("track_id", person.get("person_id", -1)))
            except (TypeError, ValueError):
                person_track_id = None
            if person_track_id not in track_ids:
                persons.append(person)
        output_frame["persons"] = persons
        selected_frames.append(output_frame)
    selected["frames"] = selected_frames
    selected["config"] = {
        **data.get("config", {}),
        "trackArray": [
            {
                "position": 1,
                "trackIds": [canonical_track_id],
                "mergedTrackIds": list(track_ids),
            }
        ],
    }
    return selected


def track_ids_from_frames(data: dict[str, Any]) -> list[int]:
    seen = set()
    for frame in data.get("frames", []):
        for person in frame.get("persons", []):
            track_id = person.get("track_id", person.get("person_id"))
            if track_id is None:
                continue
            try:
                seen.add(int(track_id))
            except (TypeError, ValueError):
                continue
    return sorted(seen)


def bbox_area(person: dict[str, Any]) -> float:
    bbox = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def largest_person_pose_data(data: dict[str, Any]) -> dict[str, Any]:
    selected = dict(data)
    selected_frames = []
    for frame in data.get("frames", []):
        output_frame = dict(frame)
        persons = frame.get("persons", [])
        if persons:
            largest = max(persons, key=bbox_area)
            largest = dict(largest)
            largest["person_id"] = 0
            largest["track_id"] = 0
            output_frame["persons"] = [largest]
        else:
            output_frame["persons"] = []
        selected_frames.append(output_frame)
    selected["frames"] = selected_frames
    selected["config"] = {
        "trackArray": [
            {
                "position": 1,
                "trackIds": [0],
            }
        ]
    }
    return selected


def samples_from_json(path: Path, single_largest_person: bool) -> list[PredictionSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if single_largest_person:
        data = largest_person_pose_data(data)
    samples = []
    configured_tracks = []
    for item in data.get("config", {}).get("trackArray", []):
        position = int(item.get("position", 0))
        track_ids = normalize_track_id_group(item.get("trackIds", []))
        if not track_ids:
            continue
        configured_tracks.extend(track_ids)
        primary_track_id = track_ids[0]
        label = track_label(track_ids)
        samples.append(
            PredictionSample(
                sample=Sample(
                    index=len(samples),
                    sample_id=f"{path.stem}_track{label}",
                    pose_json=str(path),
                    sequence_npz=None,
                    actual=0.0,
                    baseline_routed=None,
                    area=path.parent.name,
                    zone=position,
                    detection_coverage=0.0,
                    duration_sec=float(data.get("video", {}).get("duration_sec") or 0.0),
                    track_id=primary_track_id,
                ),
                pose_data=pose_data_for_track_group(data, track_ids),
                track_ids=track_ids,
                track_label=label,
            )
        )
    if samples:
        return samples

    for item in data.get("zones", []):
        try:
            zone = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        label = str(zone)
        samples.append(
            PredictionSample(
                sample=Sample(
                    index=len(samples),
                    sample_id=f"{path.stem}_zone{zone}",
                    pose_json=str(path),
                    sequence_npz=None,
                    actual=0.0,
                    baseline_routed=None,
                    area=path.parent.name,
                    zone=zone,
                    detection_coverage=0.0,
                    duration_sec=float(data.get("video", {}).get("duration_sec") or 0.0),
                    track_id=None,
                ),
                pose_data=data,
                track_ids=(),
                track_label=label,
            )
        )
    if samples:
        return samples

    for track_id in track_ids_from_frames(data):
        track_ids = (track_id,)
        samples.append(
            PredictionSample(
                sample=Sample(
                    index=len(samples),
                    sample_id=f"{path.stem}_track{track_id}",
                    pose_json=str(path),
                    sequence_npz=None,
                    actual=0.0,
                    baseline_routed=None,
                    area=path.parent.name,
                    zone=0,
                    detection_coverage=0.0,
                    duration_sec=float(data.get("video", {}).get("duration_sec") or 0.0),
                    track_id=track_id,
                ),
                pose_data=data,
                track_ids=track_ids,
                track_label=track_label(track_ids),
            )
        )
    return samples


# ===== Prediction result helpers =====


def peak_indices(probabilities: np.ndarray, threshold: float, min_distance: int) -> list[int]:
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
    return sorted(selected)


# ===== Labelclip export and evaluation helpers =====


def labelclip_path_for_json(json_path: Path, labelclip_root: Path) -> Path:
    jump = labelclip_root / f"{json_path.stem}.jump.json"
    if jump.is_file():
        return jump
    direct = labelclip_root / f"{json_path.stem}.labelclip"
    if direct.is_file():
        return direct
    nested_jump = labelclip_root / json_path.parent.name / f"{json_path.stem}.jump.json"
    if nested_jump.is_file():
        return nested_jump
    nested = labelclip_root / json_path.parent.name / f"{json_path.stem}.labelclip"
    if nested.is_file():
        return nested
    return direct


def labelclip_segments_from_predictions(
    predictions: list[dict[str, Any]],
    half_window: int,
) -> list[dict[str, Any]]:
    segments = []
    for row in predictions:
        frame_count = int(row["frame_count"])
        peak_scores = row.get("peak_scores", [])
        peak_sequence_indices = row.get("peak_sequence_indices", [])
        row_track_ids = row.get("track_ids", [])
        for peak_index, center in enumerate(row["peak_frame_indices"]):
            start = max(0, int(center) - half_window)
            end = min(max(0, frame_count - 1), int(center) + half_window)
            score = float(peak_scores[peak_index]) if peak_index < len(peak_scores) else None
            sequence_index = (
                int(peak_sequence_indices[peak_index])
                if peak_index < len(peak_sequence_indices)
                else None
            )
            segments.append(
                {
                    "interval": [start, end],
                    "usable": True,
                    "option": 0,
                    "distance": None,
                    "note": "model_generated",
                    "source": "model_prediction",
                    "track_id": str(row["track_id"]),
                    "track_ids": row_track_ids,
                    "position": int(row["position"]),
                    "score": score,
                    "peak_frame": int(center),
                    "peak_sequence_index": sequence_index,
                }
            )
    return segments


def load_countable_labelclip_segments(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    if isinstance(data.get("zones"), dict):
        for zone, items in data["zones"].items():
            for item_index, item in enumerate(items):
                if not train_tcn.is_countable_labelclip_item(item):
                    continue
                try:
                    start = int(item["start"])
                    end = int(item["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                segment = dict(item)
                segment["interval"] = [min(start, end), max(start, end)]
                segment["option"] = int(item.get("label", item.get("option", 0)))
                segment["track_id"] = str(zone)
                segment["track_ids"] = []
                segment["position"] = int(zone)
                segment["_item_index"] = item_index
                segments.append(segment)
        return segments

    for item_index, item in enumerate(data.get("segments", data.get("intervals", []))):
        if not train_tcn.is_countable_labelclip_item(item):
            continue
        interval = item.get("interval")
        if not isinstance(interval, list) or len(interval) < 2:
            continue
        try:
            start = int(interval[0])
            end = int(interval[1])
        except (TypeError, ValueError):
            continue
        track_ids = normalize_track_id_group(item.get("track_id", item.get("trackId")))
        if not track_ids:
            continue
        segment = dict(item)
        segment["interval"] = [min(start, end), max(start, end)]
        segment["track_id"] = track_label(track_ids)
        segment["track_ids"] = list(track_ids)
        segment["_item_index"] = item_index
        segments.append(segment)
    return segments


def interval_center(segment: dict[str, Any]) -> float:
    start, end = segment["interval"][:2]
    return (float(start) + float(end)) / 2.0


def compare_labelclip_segments(
    *,
    video: str,
    gt_path: Path,
    predicted_segments: list[dict[str, Any]],
    match_tolerance_frames: int,
) -> list[dict[str, Any]]:
    gt_segments = load_countable_labelclip_segments(gt_path)
    gt_by_track: dict[str, list[dict[str, Any]]] = {}
    pred_by_track: dict[str, list[dict[str, Any]]] = {}
    for segment in gt_segments:
        gt_by_track.setdefault(str(segment["track_id"]), []).append(segment)
    for segment in predicted_segments:
        pred_by_track.setdefault(str(segment["track_id"]), []).append(segment)

    rows = []
    track_ids = sorted(
        set(gt_by_track) | set(pred_by_track),
        key=lambda value: (not value.lstrip("-").isdigit(), int(value) if value.lstrip("-").isdigit() else value),
    )
    for track_id in track_ids:
        gt_items = sorted(gt_by_track.get(track_id, []), key=interval_center)
        pred_items = sorted(pred_by_track.get(track_id, []), key=interval_center)
        unmatched_pred = set(range(len(pred_items)))
        for gt_index, gt_segment in enumerate(gt_items):
            gt_center = interval_center(gt_segment)
            best_pred_index = None
            best_delta = None
            for pred_index in unmatched_pred:
                delta = interval_center(pred_items[pred_index]) - gt_center
                if abs(delta) > match_tolerance_frames:
                    continue
                if best_delta is None or abs(delta) < abs(best_delta):
                    best_pred_index = pred_index
                    best_delta = delta
            if best_pred_index is None:
                rows.append(
                    {
                        "video": video,
                        "track_id": track_id,
                        "status": "missed",
                        "gt_index": gt_index,
                        "gt_start": int(gt_segment["interval"][0]),
                        "gt_end": int(gt_segment["interval"][1]),
                        "gt_option": gt_segment.get("option"),
                        "pred_index": "",
                        "pred_start": "",
                        "pred_end": "",
                        "pred_score": "",
                        "frame_delta": "",
                        "gt_labelclip": str(gt_path),
                    }
                )
                continue

            unmatched_pred.remove(best_pred_index)
            pred_segment = pred_items[best_pred_index]
            rows.append(
                {
                    "video": video,
                    "track_id": track_id,
                    "status": "matched",
                    "gt_index": gt_index,
                    "gt_start": int(gt_segment["interval"][0]),
                    "gt_end": int(gt_segment["interval"][1]),
                    "gt_option": gt_segment.get("option"),
                    "pred_index": best_pred_index,
                    "pred_start": int(pred_segment["interval"][0]),
                    "pred_end": int(pred_segment["interval"][1]),
                    "pred_score": pred_segment.get("score", ""),
                    "frame_delta": float(best_delta),
                    "gt_labelclip": str(gt_path),
                }
            )
        for pred_index in sorted(unmatched_pred):
            pred_segment = pred_items[pred_index]
            rows.append(
                {
                    "video": video,
                    "track_id": track_id,
                    "status": "extra",
                    "gt_index": "",
                    "gt_start": "",
                    "gt_end": "",
                    "gt_option": "",
                    "pred_index": pred_index,
                    "pred_start": int(pred_segment["interval"][0]),
                    "pred_end": int(pred_segment["interval"][1]),
                    "pred_score": pred_segment.get("score", ""),
                    "frame_delta": "",
                    "gt_labelclip": str(gt_path),
                }
            )
    return rows


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def clipped_count_accuracy(predicted_count: int, actual_count: int) -> float | None:
    if actual_count == 0:
        return 1.0 if predicted_count == 0 else 0.0
    return max(0.0, 1.0 - abs(predicted_count - actual_count) / actual_count)


def count_accuracy_summary(comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, dict[str, int]] = {}
    for row in comparison_rows:
        video_counts = by_video.setdefault(
            row["video"],
            {"matched": 0, "missed": 0, "extra": 0},
        )
        video_counts[row["status"]] += 1

    per_video = {}
    total_matched = 0
    total_missed = 0
    total_extra = 0
    for video, counts in sorted(by_video.items()):
        matched = int(counts["matched"])
        missed = int(counts["missed"])
        extra = int(counts["extra"])
        actual_count = matched + missed
        predicted_count = matched + extra
        error = predicted_count - actual_count
        precision = safe_ratio(matched, predicted_count)
        recall = safe_ratio(matched, actual_count)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_video[video] = {
            "actual_count": actual_count,
            "predicted_count": predicted_count,
            "error": error,
            "abs_error": abs(error),
            "count_accuracy": clipped_count_accuracy(predicted_count, actual_count),
            "matched": matched,
            "missed": missed,
            "extra": extra,
            "interval_precision": precision,
            "interval_recall": recall,
            "interval_f1": f1,
        }
        total_matched += matched
        total_missed += missed
        total_extra += extra

    total_actual = total_matched + total_missed
    total_predicted = total_matched + total_extra
    total_error = total_predicted - total_actual
    video_accuracies = [
        item["count_accuracy"]
        for item in per_video.values()
        if item["count_accuracy"] is not None
    ]
    total_precision = safe_ratio(total_matched, total_predicted)
    total_recall = safe_ratio(total_matched, total_actual)
    total_f1 = (
        None
        if total_precision is None or total_recall is None or total_precision + total_recall == 0
        else 2 * total_precision * total_recall / (total_precision + total_recall)
    )
    return {
        "definition": "count_accuracy = max(0, 1 - abs(predicted_count - actual_count) / actual_count)",
        "overall": {
            "actual_count": total_actual,
            "predicted_count": total_predicted,
            "error": total_error,
            "abs_error": abs(total_error),
            "count_accuracy": clipped_count_accuracy(total_predicted, total_actual),
            "macro_count_accuracy": (
                float(np.mean(video_accuracies)) if video_accuracies else None
            ),
            "matched": total_matched,
            "missed": total_missed,
            "extra": total_extra,
            "interval_precision": total_precision,
            "interval_recall": total_recall,
            "interval_f1": total_f1,
        },
        "by_video": per_video,
    }


# ===== Model inference helpers =====


def build_model(checkpoint: dict[str, Any], device: torch.device) -> SequenceHeatmapTCN:
    config = checkpoint["model_config"]
    model = SequenceHeatmapTCN(
        feature_dim=int(checkpoint["feature_dim"]),
        meta_dim=int(checkpoint["meta_dim"]),
        hidden_channels=int(config["hidden_channels"]),
        layers=int(config["layers"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        use_metadata=bool(config["use_metadata"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_one_batch(
    model: SequenceHeatmapTCN,
    sequences: np.ndarray,
    metadata: np.ndarray,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = start + batch_size
            batch_x = torch.from_numpy(sequences[start:end]).float().to(device)
            batch_meta = torch.from_numpy(metadata[start:end]).float().to(device)
            rows.append(torch.sigmoid(model(batch_x, batch_meta)).detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


# ===== Labelclip writers =====


def export_labelclip(path: Path, segments: list[dict[str, Any]]) -> None:
    output = {
        "version": "model_generated_v1",
        "product": "MP",
        "sport": "skip_rope",
        "segments": segments,
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_jump_id(video: str, zone: int, index: int, start: int, end: int) -> str:
    raw = f"{video}|{zone}|{index}|{start}|{end}|model_prediction"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def export_jump_json(
    path: Path,
    *,
    video: str,
    frame_count: int,
    duration_sec: float,
    predictions: list[dict[str, Any]],
    half_window: int,
) -> None:
    zones: dict[str, list[dict[str, int | str]]] = {}
    item_index = 0
    for row in predictions:
        zone = int(row["position"])
        zone_key = str(zone)
        for center in row.get("peak_frame_indices", []):
            start = max(0, int(center) - half_window)
            end = min(max(0, frame_count - 1), int(center) + half_window)
            zones.setdefault(zone_key, []).append(
                {
                    "id": stable_jump_id(video, zone, item_index, start, end),
                    "start": start,
                    "end": end,
                    "label": 0,
                }
            )
            item_index += 1

    output = {
        "video": video,
        "total_frames": int(frame_count),
        "duration_sec": float(duration_sec),
        "zones": {
            zone: sorted(items, key=lambda item: (int(item["start"]), int(item["end"])))
            for zone, items in sorted(
                zones.items(),
                key=lambda item: int(item[0]) if item[0].lstrip("-").isdigit() else item[0],
            )
        },
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ===== CLI orchestration =====


@dataclass(frozen=True)
class PredictorContext:
    device: torch.device
    checkpoint: dict[str, Any]
    model: SequenceHeatmapTCN
    sequence_length: int
    normalization: dict[str, Any]
    threshold: float
    min_distance: int


@dataclass(frozen=True)
class PipelineOutputs:
    input_paths: list[Path]
    export_labelclip_dir: Path | None


@dataclass(frozen=True)
class JsonPipelineResult:
    rows: list[dict[str, Any]]
    comparison_rows: list[dict[str, Any]]


def load_predictor_context(args: argparse.Namespace) -> PredictorContext:
    device = choose_device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    keypoint_names = configure_keypoints_for_checkpoint(checkpoint)
    print(
        f"feature_schema={len(keypoint_names)}kp/"
        f"{int(checkpoint['feature_dim'])}d"
    )
    model = build_model(checkpoint, device)
    sequence_length = int(checkpoint["sequence_length"])
    normalization = checkpoint["normalization"]
    peak_params = checkpoint["peak_params"]
    threshold = float(args.peak_threshold if args.peak_threshold is not None else peak_params["threshold"])
    min_distance = int(args.peak_min_distance if args.peak_min_distance is not None else peak_params["min_distance"])
    return PredictorContext(
        device=device,
        checkpoint=checkpoint,
        model=model,
        sequence_length=sequence_length,
        normalization=normalization,
        threshold=threshold,
        min_distance=min_distance,
    )


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    input_paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            input_paths.extend(sorted(path.glob("*.json")))
        else:
            input_paths.append(path)
    if not input_paths:
        raise FileNotFoundError("No input JSON files found")
    return input_paths


def prepare_pipeline_outputs(args: argparse.Namespace, input_paths: list[Path]) -> PipelineOutputs:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_labelclip_dir = args.export_labelclip_dir
    if export_labelclip_dir is None and not args.no_export_labelclips:
        export_labelclip_dir = args.out_dir / "predicted_labelclips"
    if export_labelclip_dir is not None:
        export_labelclip_dir.mkdir(parents=True, exist_ok=True)
    if args.gt_labelclip_dir is not None and not args.gt_labelclip_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth labelclip directory not found: {args.gt_labelclip_dir}")
    return PipelineOutputs(input_paths=input_paths, export_labelclip_dir=export_labelclip_dir)


def filter_samples_by_ground_truth(
    prediction_samples: list[PredictionSample],
    gt_path: Path,
) -> list[PredictionSample]:
    gt_track_ids = {
        str(segment["track_id"])
        for segment in load_countable_labelclip_segments(gt_path)
    }
    if not gt_track_ids:
        return prediction_samples
    return [
        item
        for item in prediction_samples
        if item.track_label in gt_track_ids
    ]


def preprocess_samples_for_model(
    *,
    json_path: Path,
    prediction_samples: list[PredictionSample],
    context: PredictorContext,
) -> tuple[list[Sample], np.ndarray, np.ndarray]:
    samples = [item.sample for item in prediction_samples]
    sequences = []
    for item in prediction_samples:
        features = pose_json_to_features(item.pose_data, item.sample.track_id, item.sample.zone)
        sequences.append(resample_sequence(features, context.sequence_length))
    sequences_np = np.stack(sequences, axis=0).astype(np.float32)
    expected_feature_dim = int(context.checkpoint["feature_dim"])
    if sequences_np.shape[2] != expected_feature_dim:
        raise RuntimeError(
            f"Generated {sequences_np.shape[2]} features from {json_path}, "
            f"but checkpoint expects {expected_feature_dim}."
        )
    metadata_np = metadata_matrix(samples)
    normalization = context.normalization
    sequences_np = ((sequences_np - normalization["seq_mean"]) / normalization["seq_std"]).astype(np.float32)
    metadata_np = ((metadata_np - normalization["meta_mean"]) / normalization["meta_std"]).astype(np.float32)
    return samples, sequences_np, metadata_np


def infer_prediction_rows(
    *,
    json_path: Path,
    prediction_samples: list[PredictionSample],
    samples: list[Sample],
    sequences_np: np.ndarray,
    metadata_np: np.ndarray,
    context: PredictorContext,
    batch_size: int,
) -> list[dict[str, Any]]:
    probabilities = predict_one_batch(
        context.model,
        sequences_np,
        metadata_np,
        context.checkpoint,
        context.device,
        batch_size,
    )

    rows_for_json = []
    for prediction_sample, sample, probs in zip(
        prediction_samples,
        samples,
        probabilities,
    ):
        data = prediction_samples[0].pose_data
        frame_count = len(data.get("frames", []))
        indexes = peak_indices(probs, context.threshold, context.min_distance)
        peak_frame_indices = [
            int(round(index * max(1, frame_count - 1) / max(1, context.sequence_length - 1)))
            for index in indexes
        ]
        peak_scores = [float(probs[index]) for index in indexes]
        rows_for_json.append(
            {
                "json": str(json_path),
                "video": json_path.stem,
                "position": sample.zone,
                "track_id": prediction_sample.track_label,
                "track_ids": list(prediction_sample.track_ids),
                "peak_count": len(indexes),
                "threshold": context.threshold,
                "min_distance": context.min_distance,
                "frame_count": frame_count,
                "peak_frame_indices": peak_frame_indices,
                "peak_sequence_indices": indexes,
                "peak_scores": peak_scores,
            }
        )
    return rows_for_json


def export_empty_prediction_if_needed(
    *,
    json_path: Path,
    export_labelclip_dir: Path | None,
    interval_half_window: int,
) -> None:
    if export_labelclip_dir is None:
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frame_count = len(data.get("frames", []))
    export_jump_json(
        export_labelclip_dir / f"{json_path.stem}.jump.json",
        video=json_path.stem,
        frame_count=frame_count,
        duration_sec=float(data.get("video", {}).get("duration_sec") or 0.0),
        predictions=[],
        half_window=interval_half_window,
    )


def export_prediction_if_needed(
    *,
    json_path: Path,
    prediction_samples: list[PredictionSample],
    rows_for_json: list[dict[str, Any]],
    export_labelclip_dir: Path | None,
    interval_half_window: int,
) -> None:
    if export_labelclip_dir is None:
        return
    data = prediction_samples[0].pose_data
    export_jump_json(
        export_labelclip_dir / f"{json_path.stem}.jump.json",
        video=json_path.stem,
        frame_count=len(data.get("frames", [])),
        duration_sec=float(data.get("video", {}).get("duration_sec") or 0.0),
        predictions=rows_for_json,
        half_window=interval_half_window,
    )


def compare_predictions_if_needed(
    *,
    json_path: Path,
    gt_labelclip_dir: Path | None,
    gt_path: Path | None,
    predicted_segments: list[dict[str, Any]],
    match_tolerance_frames: int,
) -> list[dict[str, Any]]:
    if gt_labelclip_dir is None:
        return []
    return compare_labelclip_segments(
        video=json_path.stem,
        gt_path=gt_path if gt_path is not None else labelclip_path_for_json(json_path, gt_labelclip_dir),
        predicted_segments=predicted_segments,
        match_tolerance_frames=match_tolerance_frames,
    )


def run_json_pipeline(
    *,
    json_path: Path,
    args: argparse.Namespace,
    context: PredictorContext,
    export_labelclip_dir: Path | None,
) -> JsonPipelineResult:
    prediction_samples = samples_from_json(json_path, args.single_largest_person)
    gt_path = (
        labelclip_path_for_json(json_path, args.gt_labelclip_dir)
        if args.gt_labelclip_dir is not None
        else None
    )
    if gt_path is not None and not args.include_non_gt_tracks:
        prediction_samples = filter_samples_by_ground_truth(prediction_samples, gt_path)

    if not prediction_samples:
        predicted_segments: list[dict[str, Any]] = []
        export_empty_prediction_if_needed(
            json_path=json_path,
            export_labelclip_dir=export_labelclip_dir,
            interval_half_window=args.interval_half_window,
        )
        comparison_rows = compare_predictions_if_needed(
            json_path=json_path,
            gt_labelclip_dir=args.gt_labelclip_dir,
            gt_path=gt_path,
            predicted_segments=predicted_segments,
            match_tolerance_frames=args.match_tolerance_frames,
        )
        return JsonPipelineResult(rows=[], comparison_rows=comparison_rows)

    samples, sequences_np, metadata_np = preprocess_samples_for_model(
        json_path=json_path,
        prediction_samples=prediction_samples,
        context=context,
    )
    rows_for_json = infer_prediction_rows(
        json_path=json_path,
        prediction_samples=prediction_samples,
        samples=samples,
        sequences_np=sequences_np,
        metadata_np=metadata_np,
        context=context,
        batch_size=args.batch_size,
    )
    predicted_segments = labelclip_segments_from_predictions(rows_for_json, args.interval_half_window)
    export_prediction_if_needed(
        json_path=json_path,
        prediction_samples=prediction_samples,
        rows_for_json=rows_for_json,
        export_labelclip_dir=export_labelclip_dir,
        interval_half_window=args.interval_half_window,
    )
    comparison_rows = compare_predictions_if_needed(
        json_path=json_path,
        gt_labelclip_dir=args.gt_labelclip_dir,
        gt_path=gt_path,
        predicted_segments=predicted_segments,
        match_tolerance_frames=args.match_tolerance_frames,
    )
    return JsonPipelineResult(rows=rows_for_json, comparison_rows=comparison_rows)


def write_predictions_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "json",
                "video",
                "position",
                "track_id",
                "peak_count",
                "threshold",
                "min_distance",
                "frame_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_comparison_outputs(
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> Path:
    comparison_csv_path = args.out_dir / "interval_comparison.csv"
    with comparison_csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "video",
            "track_id",
            "status",
            "gt_index",
            "gt_start",
            "gt_end",
            "gt_option",
            "pred_index",
            "pred_start",
            "pred_end",
            "pred_score",
            "frame_delta",
            "gt_labelclip",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    comparison_json_path = args.out_dir / "interval_comparison_summary.json"
    comparison_summary: dict[str, Any] = {
        "matched": sum(1 for row in comparison_rows if row["status"] == "matched"),
        "missed": sum(1 for row in comparison_rows if row["status"] == "missed"),
        "extra": sum(1 for row in comparison_rows if row["status"] == "extra"),
        "by_video": {},
    }
    for row in comparison_rows:
        video_summary = comparison_summary["by_video"].setdefault(
            row["video"],
            {"matched": 0, "missed": 0, "extra": 0},
        )
        video_summary[row["status"]] += 1
    accuracy_summary = count_accuracy_summary(comparison_rows)
    comparison_summary["accuracy"] = accuracy_summary
    comparison_json_path.write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["interval_comparison_csv"] = str(comparison_csv_path)
    summary["interval_comparison_summary"] = str(comparison_json_path)
    summary["accuracy"] = accuracy_summary
    return comparison_csv_path


def write_pipeline_outputs(
    *,
    args: argparse.Namespace,
    context: PredictorContext,
    pipeline_outputs: PipelineOutputs,
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    csv_path = args.out_dir / "predictions.csv"
    write_predictions_csv(csv_path, rows)

    summary = {
        "model": str(args.model),
        "inputs": [str(path) for path in pipeline_outputs.input_paths],
        "prediction_count": len(rows),
        "single_largest_person": bool(args.single_largest_person),
        "threshold": context.threshold,
        "min_distance": context.min_distance,
        "output_csv": str(csv_path),
        "export_labelclip_dir": (
            str(pipeline_outputs.export_labelclip_dir)
            if pipeline_outputs.export_labelclip_dir
            else None
        ),
        "gt_labelclip_dir": str(args.gt_labelclip_dir) if args.gt_labelclip_dir else None,
        "include_non_gt_tracks": bool(args.include_non_gt_tracks),
        "match_tolerance_frames": args.match_tolerance_frames,
    }
    comparison_csv_path = None
    if args.gt_labelclip_dir is not None:
        comparison_csv_path = write_comparison_outputs(
            args=args,
            summary=summary,
            comparison_rows=comparison_rows,
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"predictions={len(rows)}")
    print(f"output={csv_path.resolve()}")
    if pipeline_outputs.export_labelclip_dir is not None:
        print(f"labelclip_output={pipeline_outputs.export_labelclip_dir.resolve()}")
    if comparison_csv_path is not None:
        print(f"interval_comparison={comparison_csv_path.resolve()}")


def run(args: argparse.Namespace) -> None:
    context = load_predictor_context(args)
    input_paths = resolve_input_paths(args.inputs)
    pipeline_outputs = prepare_pipeline_outputs(args, input_paths)

    all_rows = []
    all_comparison_rows = []
    for json_path in pipeline_outputs.input_paths:
        result = run_json_pipeline(
            json_path=json_path,
            args=args,
            context=context,
            export_labelclip_dir=pipeline_outputs.export_labelclip_dir,
        )
        all_rows.extend(result.rows)
        all_comparison_rows.extend(result.comparison_rows)

    write_pipeline_outputs(
        args=args,
        context=context,
        pipeline_outputs=pipeline_outputs,
        rows=all_rows,
        comparison_rows=all_comparison_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict jump counts or labelclip files from pose JSON.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON files or directories containing JSON files.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "predicted_labelclip")
    parser.add_argument(
        "--export-labelclip-dir",
        type=Path,
        default=None,
        help="Directory for model-generated .labelclip files. Defaults to OUT_DIR/predicted_labelclips.",
    )
    parser.add_argument(
        "--no-export-labelclips",
        action="store_true",
        help="Disable the default model-generated .labelclip export.",
    )
    parser.add_argument(
        "--gt-labelclip-dir",
        type=Path,
        default=None,
        help="Optional ground-truth .labelclip directory. Enables interval_comparison.csv/json.",
    )
    parser.add_argument(
        "--include-non-gt-tracks",
        action="store_true",
        help="When --gt-labelclip-dir is set, also predict tracks that do not appear in ground truth.",
    )
    parser.add_argument("--interval-half-window", type=int, default=3)
    parser.add_argument(
        "--match-tolerance-frames",
        type=int,
        default=6,
        help="Max center-frame distance for matching a predicted interval to a ground-truth interval.",
    )
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--peak-min-distance", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--single-largest-person",
        action="store_true",
        help="For each JSON, predict only one virtual person built from the largest bbox in each frame.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
