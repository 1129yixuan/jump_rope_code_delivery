"""Routing counter entry point: read pose JSON and output jump counts by zone."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from posecount import RopeJumpConfig, RopeJumpCounter
from posecount.counters.base import CounterResult
from posecount.pose.base import Keypoint, PersonPose, PoseResult

LOGGER = logging.getLogger(__name__)
KEYPOINT_MIN_SCORE = 0.1

# Ankle filtering in distant zones is more sensitive to perspective distortion,
# so these zones retain individual threshold overrides.
ZONE_COUNTER_OVERRIDES: dict[int, dict[str, float | int]] = {
    3: {
        "min_ankle_reject_samples": 4,
        "ankle_reject_diff_ratio": 0.030,
    },
    4: {
        "min_ankle_reject_samples": 2,
        "ankle_reject_diff_ratio": 0.006,
    },
    5: {
        "min_ankle_reject_samples": 4,
        "ankle_reject_diff_ratio": 0.030,
    },
}

# Routing thresholds: estimate a low, middle, or high level first, then apply
# feature-based guards and rescue rules.
LOW_SCORE_MAX = 100
HIGH_SCORE_MIN = 130
LOW_RISK_MIN = 100
LOW_RISK_MAX = 115
EXPERT_SHOULDER_HEIGHT_RATIO = 0.10
LOW_RISK_SHOULDER_HEIGHT_RATIO = 0.11
ANKLE_SUPPORT_MIN_RHYTHM = 0.35
ANKLE_SUPPORT_COUNT_GAP = 20
EXPERT_RISK_ZONES = {3, 5}
LOW_SECOND_STAGE_SUPPRESSED_MAX = 45
LOW_SECOND_STAGE_RESCUE_MIN = 70
LOW_SECOND_STAGE_RESCUE_MAX = 110
LOW_SECOND_STAGE_RESCUE_GAP = 40
LOW_SECOND_STAGE_HIGH_RESCUE_MIN = 120
LOW_SECOND_STAGE_HIGH_RESCUE_MAX = 145
LOW_SECOND_STAGE_HIGH_RESCUE_GAP = 35
LOW_SECOND_STAGE_MIDDLE_MAX = 115
NEAR_ZONE_IDS = {1, 2}
NEAR_ZONE_FILTERED_RESCUE_MIN = 120
NEAR_ZONE_FILTERED_RESCUE_GAP = 15
NEAR_ZONE_STRONG_EXPERT_SHOULDER_MAX = 0.12
NEAR_ZONE_STRONG_EXPERT_HIGH_GAP = 20
NEAR_ZONE_SUBTLE_EXPERT_HIGH_MIN = 145
NEAR_ZONE_SUBTLE_EXPERT_MIDDLE_MIN = 135
NEAR_ZONE_SUBTLE_EXPERT_HIGH_GAP = 8
NEAR_ZONE_SUBTLE_EXPERT_SHOULDER_MAX = 0.125
NEAR_ZONE_SUBTLE_EXPERT_ANKLE_RHYTHM_MAX = 0.40
NEAR_ZONE_LOW_SUPPRESS_MIN = 80
NEAR_ZONE_LOW_SUPPRESS_MAX = 115
NEAR_ZONE_LOW_SUPPRESS_SHOULDER_MIN = 0.145
NEAR_ZONE_LOW_SUPPRESS_SEGMENTS_MIN = 5
NEAR_ZONE_LOW_SUPPRESS_FACTOR = 0.90
NEAR_ZONE_VERY_LOW_SUPPRESS_MAX = 85
NEAR_ZONE_VERY_LOW_SUPPRESS_SHOULDER_MIN = 0.145
NEAR_ZONE_VERY_LOW_SUPPRESS_ALGORITHM_RANGE_MAX = 25
NEAR_ZONE_VERY_LOW_SUPPRESS_SEGMENTS_MIN = 6
NEAR_ZONE_VERY_LOW_SUPPRESS_FACTOR = 0.80
FAR_ZONE_IDS = {3, 4, 5}
FAR_ZONE_STABLE_MIDDLE_MIN = 100
FAR_ZONE_STABLE_MIDDLE_MAX = 130
FAR_ZONE_STABLE_ALGORITHM_RANGE_MAX = 20
FAR_ZONE_STABLE_SEGMENTS_MAX = 5
FAR_ZONE_LOW_ABNORMAL_MIDDLE_MAX = 115
FAR_ZONE_LOW_ABNORMAL_SEGMENTS_MIN = 5
FAR_ZONE_LOW_ABNORMAL_SHOULDER_MIN = 0.13
FAR_ZONE_LOW_ABNORMAL_HAND_Y_MIN = 0.14
FAR_ZONE_LOW_SUPPRESS_FACTOR = 0.90
FAR_ZONE_LOW_SUPPRESS_STRONG_FACTOR = 0.85
FAR_ZONE_LOW_SUPPRESS_STRONG_HAND_Y_MIN = 0.16
FAR_ZONE_LOW_SUPPRESS_STRONG_SEGMENTS_MIN = 8
FEET_TOGETHER_ANKLE_STRENGTH_MAX = 0.023
FEET_TOGETHER_ANKLE_SHOULDER_RATIO_MIN = 0.85

# Expert path: use a more sensitive shoulder counter for high-score samples
# whose shoulders barely move.
HIGH_COUNTER_KWARGS: dict[str, Any] = {
    "count_mode": "shoulder",
    "extrema_window_size": 7,
    "min_frame_interval": 5,
    "min_symmetry_ratio": 0.10,
    "min_jump_height_ratio": 0.04,
    "recommended_jump_height_ratio": 0.10,
    "max_jump_height_ratio": 0.60,
    "min_jump_duration": 4,
    "max_jump_duration": 30,
    "enable_smoothing": False,
    "smoothing_window_size": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count rope jumps per zone from an existing pose JSON."
    )
    parser.add_argument(
        "json",
        type=Path,
        help="Existing pose JSON path.",
    )
    parser.add_argument("--zones", type=Path, required=True, help="Zone config JSON.")
    parser.add_argument("--zone", type=int, default=None, help="Only print one zone id.")
    parser.add_argument(
        "--count-mode",
        choices=("shoulder", "shoulder_ankle"),
        default="shoulder_ankle",
        help="Use shoulder_ankle for the new routing rule, or shoulder for raw shoulder count.",
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=None,
        help="Optional path for saving the count summary JSON.",
    )
    return parser.parse_args()

def load_frames(json_path: Path) -> list[dict[str, Any]]:
    try:
        with json_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not read inference JSON as UTF-8: {json_path}. "
            "The file is likely corrupted. Re-run with --force-infer to regenerate it."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse inference JSON: {json_path}. "
            "The file is likely incomplete or corrupted. Re-run with --force-infer to regenerate it."
        ) from exc
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"No frames list in JSON: {json_path}")
    return frames


def load_zones(config_path: Path | None) -> list[dict[str, Any]]:
    if config_path is None:
        return []
    with config_path.open(encoding="utf-8") as file:
        data = json.load(file)
    zones = data.get("zones", [])
    if not isinstance(zones, list):
        raise ValueError(f"No zones list in config: {config_path}")
    return zones


def parse_pose_result(frame: dict[str, Any]) -> PoseResult:
    return PoseResult(
        frame_index=frame.get("frame_index"),
        timestamp_sec=frame.get("timestamp_sec"),
        persons=[
            PersonPose(
                person_id=int(person["person_id"]),
                bbox=tuple(person["bbox"]),
                score=float(person["score"]),
                keypoints=[
                    Keypoint(
                        name=str(kp["name"]),
                        x=float(kp["x"]),
                        y=float(kp["y"]),
                        score=float(kp["score"]),
                    )
                    for kp in person.get("keypoints", [])
                ],
            )
            for person in frame.get("persons", [])
        ],
    )


def bbox_area(person: PersonPose) -> float:
    x1, y1, x2, y2 = person.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def foot_in_zone(person: PersonPose, zone: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
    for kp in person.keypoints:
        if kp.name in ("left_ankle", "right_ankle") and kp.score >= KEYPOINT_MIN_SCORE:
            if x1 <= kp.x <= x2 and y1 <= kp.y <= y2:
                return True
    return False


def build_zone_counter_config(zone_id: int, count_mode: str = "shoulder") -> RopeJumpConfig:
    kwargs: dict[str, Any] = {"count_mode": count_mode}
    kwargs.update(ZONE_COUNTER_OVERRIDES.get(zone_id, {}))
    return RopeJumpConfig(**kwargs)


def build_high_counter_config() -> RopeJumpConfig:
    return RopeJumpConfig(**HIGH_COUNTER_KWARGS)


def counter_count(counter: RopeJumpCounter) -> int:
    get_count = getattr(counter, "get_count", None)
    if callable(get_count):
        return int(get_count())
    return int(getattr(counter, "_jump_count", 0))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def ratio(value: float | None, denominator: float) -> float | None:
    if value is None or denominator <= 0:
        return None
    return value / denominator


def candidate_height(candidate: Any) -> float:
    return (
        (candidate.peak.value - candidate.start_valley.value)
        + (candidate.peak.value - candidate.end_valley.value)
    ) / 2.0


def counter_body_height(counter: RopeJumpCounter) -> float:
    return float(
        getattr(counter, "_body_height_ref", 0.0)
        or getattr(counter, "body_height_ref", 0.0)
        or 0.0
    )


def counter_history(counter: RopeJumpCounter) -> list[Any]:
    return list(getattr(counter, "_history", getattr(counter, "history_list", [])))


def counter_candidates(counter: RopeJumpCounter) -> list[Any]:
    return list(getattr(counter, "_candidates", getattr(counter, "candidate_intervals", [])))


def counter_ankle_extrema(counter: RopeJumpCounter) -> list[Any]:
    return list(getattr(counter, "_ankle_extrema", getattr(counter, "_ankle_extrema_list", [])))


def summarize_shoulder_features(counter: RopeJumpCounter) -> dict[str, Any]:
    # Shoulder features identify stable experts, low-score anomalies, and breaks.
    body_height = counter_body_height(counter)
    history = counter_history(counter)
    candidates = counter_candidates(counter)
    shoulder_values = [
        float(getattr(sample, "mid_shoulder_y", getattr(sample, "signal_value", 0.0)))
        for sample in history
    ]
    p10 = percentile(shoulder_values, 0.10)
    p90 = percentile(shoulder_values, 0.90)
    signal_range = p90 - p10 if p10 is not None and p90 is not None else None

    accepted = [
        candidate
        for candidate in candidates
        if getattr(getattr(candidate, "judge_result", None), "name", "") == "ACCEPTED"
    ]
    accepted_heights = [candidate_height(candidate) for candidate in accepted]
    reject_counts = Counter(
        getattr(getattr(candidate, "judge_result", None), "name", "UNKNOWN")
        for candidate in candidates
    )
    return {
        "shoulder_candidate_count": len(candidates),
        "shoulder_accepted_candidate_count": len(accepted),
        "shoulder_segment_count": estimate_accepted_segment_count(accepted),
        "shoulder_signal_p10_p90_ratio": ratio(signal_range, body_height),
        "shoulder_accepted_height_median_ratio": ratio(
            median_or_none(accepted_heights),
            body_height,
        ),
        "shoulder_reject_counts": dict(sorted(reject_counts.items())),
    }


def estimate_accepted_segment_count(candidates: list[Any]) -> int:
    # Estimate the number of continuous-jump segments from large gaps between accepted peaks.
    if not candidates:
        return 0
    frames = sorted(int(candidate.peak.frame_index) for candidate in candidates)
    gaps = [
        current - previous
        for previous, current in zip(frames, frames[1:])
        if current > previous
    ]
    median_gap = median_or_none([float(gap) for gap in gaps])
    split_gap = max(24, int(round(float(median_gap or 24) * 2.2)))
    return 1 + sum(1 for gap in gaps if gap > split_gap)


def summarize_ankle_features(counter: RopeJumpCounter) -> dict[str, Any]:
    # Ankle features measure rhythmic extrema and direction changes in left-right height differences.
    body_height = counter_body_height(counter)
    extrema = counter_ankle_extrema(counter)
    valid_transitions = 0
    intervals: list[int] = []
    config = counter.config
    for previous, current in zip(extrema, extrema[1:]):
        gap = current.frame_index - previous.frame_index
        intervals.append(gap)
        if (
            config.min_ankle_event_interval <= gap <= config.max_ankle_event_interval
            and (current.type != previous.type or current.value * previous.value < 0)
        ):
            valid_transitions += 1

    transition_count = max(0, len(extrema) - 1)
    rhythm_score = valid_transitions / transition_count if transition_count else None
    history = counter_history(counter)
    ankle_values = [
        float(sample.ankle_diff)
        for sample in history
        if getattr(sample, "ankle_diff", None) is not None
    ]
    if ankle_values:
        baseline = statistics.median(ankle_values)
        abs_values = [abs(value - baseline) for value in ankle_values]
    else:
        abs_values = []
    return {
        "ankle_extrema_count": len(extrema),
        "ankle_valid_transition_count": valid_transitions,
        "ankle_rhythm_score": rhythm_score,
        "ankle_strength_median_ratio": ratio(median_or_none(abs_values), body_height),
        "ankle_event_interval_median": median_or_none([float(v) for v in intervals]),
    }


def keypoint_by_name(person: PersonPose, name: str) -> Keypoint | None:
    for keypoint in person.keypoints:
        if keypoint.name == name:
            return keypoint
    return None


def valid_keypoint(keypoint: Keypoint | None, min_score: float = KEYPOINT_MIN_SCORE) -> bool:
    return (
        keypoint is not None
        and math.isfinite(keypoint.x)
        and math.isfinite(keypoint.y)
        and math.isfinite(keypoint.score)
        and keypoint.score >= min_score
    )


def body_height_for_hand_features(person: PersonPose) -> float | None:
    left_shoulder = keypoint_by_name(person, "left_shoulder")
    right_shoulder = keypoint_by_name(person, "right_shoulder")
    left_hip = keypoint_by_name(person, "left_hip")
    right_hip = keypoint_by_name(person, "right_hip")
    if not all(valid_keypoint(point) for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return None

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0
    torso_height = abs(hip_y - shoulder_y)
    left_ankle = keypoint_by_name(person, "left_ankle")
    right_ankle = keypoint_by_name(person, "right_ankle")
    if valid_keypoint(left_ankle) and valid_keypoint(right_ankle):
        ankle_y = (left_ankle.y + right_ankle.y) / 2.0
        return max(abs(ankle_y - shoulder_y), torso_height * 2.2, 1.0)

    x1, y1, x2, y2 = person.bbox
    return max(y2 - y1, torso_height * 2.2, 1.0)


def summarize_hand_features(poses: list[PersonPose]) -> dict[str, Any]:
    # Vertical hand range mainly identifies low-score anomalies and erratic motion.
    wrist_mid_y_values: list[float] = []
    both_wrist_frames = 0
    for person in poses:
        left_shoulder = keypoint_by_name(person, "left_shoulder")
        right_shoulder = keypoint_by_name(person, "right_shoulder")
        left_wrist = keypoint_by_name(person, "left_wrist")
        right_wrist = keypoint_by_name(person, "right_wrist")
        body_height = body_height_for_hand_features(person)
        if (
            body_height is None
            or not valid_keypoint(left_shoulder)
            or not valid_keypoint(right_shoulder)
            or not valid_keypoint(left_wrist)
            or not valid_keypoint(right_wrist)
        ):
            continue
        both_wrist_frames += 1
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        wrist_mid_y = ((left_wrist.y + right_wrist.y) / 2.0 - shoulder_y) / body_height
        wrist_mid_y_values.append(wrist_mid_y)

    p10 = percentile(wrist_mid_y_values, 0.10)
    p90 = percentile(wrist_mid_y_values, 0.90)
    hand_y_range = p90 - p10 if p10 is not None and p90 is not None else None
    return {
        "wrist_both_visibility": both_wrist_frames / len(poses) if poses else None,
        "hand_y_range_ratio": hand_y_range,
    }


def base_score_level(zone_aware_count: int) -> str:
    if zone_aware_count >= HIGH_SCORE_MIN:
        return "high"
    if zone_aware_count >= LOW_SCORE_MAX:
        return "normal"
    return "low"


def build_level_routing(
    *,
    zone_id: int,
    zone_aware_count: int,
    ankle_only_count: int,
    shoulder_features: dict[str, Any],
    ankle_features: dict[str, Any],
) -> dict[str, Any]:
    # Stage one assigns a coarse route; it does not directly determine the final count.
    shoulder_height = shoulder_features.get("shoulder_accepted_height_median_ratio")
    ankle_rhythm = ankle_features.get("ankle_rhythm_score")
    initial_level = base_score_level(zone_aware_count)
    reasons: list[str] = [f"zone_aware_count={zone_aware_count}", f"initial={initial_level}"]

    high_candidate = (
        zone_aware_count < HIGH_SCORE_MIN
        and zone_id in EXPERT_RISK_ZONES
        and isinstance(shoulder_height, (int, float))
        and shoulder_height < EXPERT_SHOULDER_HEIGHT_RATIO
    )
    low_candidate = (
        LOW_RISK_MIN <= zone_aware_count <= LOW_RISK_MAX
        and isinstance(shoulder_height, (int, float))
        and shoulder_height >= LOW_RISK_SHOULDER_HEIGHT_RATIO
        and (ankle_rhythm is None or ankle_rhythm < ANKLE_SUPPORT_MIN_RHYTHM)
    )
    ankle_support = (
        ankle_only_count >= zone_aware_count + ANKLE_SUPPORT_COUNT_GAP
        and isinstance(ankle_rhythm, (int, float))
        and ankle_rhythm >= ANKLE_SUPPORT_MIN_RHYTHM
    )

    if zone_aware_count >= HIGH_SCORE_MIN:
        return {
            "initial_level": initial_level,
            "routing_level": "high",
            "recommended_algorithm": "zone_aware",
            "routing_reasons": reasons + ["zone_aware>=130"],
        }
    if high_candidate:
        return {
            "initial_level": initial_level,
            "routing_level": "high_candidate",
            "recommended_algorithm": "expert_placeholder",
            "routing_reasons": reasons
            + [
                f"zone={zone_id} is expert-risk",
                f"shoulder_height<{EXPERT_SHOULDER_HEIGHT_RATIO}",
            ],
        }
    if zone_aware_count < LOW_SCORE_MAX:
        if ankle_support:
            return {
                "initial_level": initial_level,
                "routing_level": "low_candidate",
                "recommended_algorithm": "low_guard_placeholder",
                "routing_reasons": reasons
                + [
                    "zone_aware<100",
                    f"ankle_count>=zone_aware+{ANKLE_SUPPORT_COUNT_GAP}",
                    f"ankle_rhythm>={ANKLE_SUPPORT_MIN_RHYTHM}",
                    "ankle_support alone is not enough for expert routing",
                ],
            }
        return {
            "initial_level": initial_level,
            "routing_level": "low",
            "recommended_algorithm": "low_guard_placeholder",
            "routing_reasons": reasons + ["zone_aware<100"],
        }
    if low_candidate:
        return {
            "initial_level": initial_level,
            "routing_level": "low_candidate",
            "recommended_algorithm": "low_guard_placeholder",
            "routing_reasons": reasons
            + [
                f"{LOW_RISK_MIN}<=zone_aware<={LOW_RISK_MAX}",
                f"shoulder_height>={LOW_RISK_SHOULDER_HEIGHT_RATIO}",
                f"ankle_rhythm<{ANKLE_SUPPORT_MIN_RHYTHM}",
            ],
        }
    return {
        "initial_level": initial_level,
        "routing_level": "normal",
        "recommended_algorithm": "zone_aware",
        "routing_reasons": reasons + ["default_normal_path"],
    }


def apply_low_candidate_second_stage(
    routing: dict[str, Any],
    *,
    low_count: int,
    middle_count: int,
    high_count: int,
) -> dict[str, Any]:
    """Second-pass low-score check for samples clearly suppressed too far."""
    updated = dict(routing)
    updated.setdefault("selected_algorithm_path", _path_from_routing_level(updated["routing_level"]))
    if updated.get("routing_level") != "low_candidate":
        return updated

    upper_path = "high" if high_count >= middle_count else "middle"
    upper_count = max(middle_count, high_count)
    reasons = list(updated.get("routing_reasons", []))

    if (
        low_count <= LOW_SECOND_STAGE_SUPPRESSED_MAX
        and LOW_SECOND_STAGE_RESCUE_MIN <= upper_count <= LOW_SECOND_STAGE_RESCUE_MAX
        and upper_count - low_count >= LOW_SECOND_STAGE_RESCUE_GAP
    ):
        updated.update(
            {
                "routing_level": "low_candidate_second_stage",
                "recommended_algorithm": (
                    "expert_placeholder" if upper_path == "high" else "zone_aware"
                ),
                "selected_algorithm_path": upper_path,
                "routing_reasons": reasons
                + [
                    "low_candidate second-stage rescue",
                    f"low_count<={LOW_SECOND_STAGE_SUPPRESSED_MAX}",
                    f"{LOW_SECOND_STAGE_RESCUE_MIN}<=max(middle,high)<={LOW_SECOND_STAGE_RESCUE_MAX}",
                    f"max(middle,high)-low>={LOW_SECOND_STAGE_RESCUE_GAP}",
                ],
            }
        )
        return updated

    if (
        LOW_SECOND_STAGE_HIGH_RESCUE_MIN <= high_count <= LOW_SECOND_STAGE_HIGH_RESCUE_MAX
        and high_count - low_count >= LOW_SECOND_STAGE_HIGH_RESCUE_GAP
        and middle_count <= LOW_SECOND_STAGE_MIDDLE_MAX
    ):
        updated.update(
            {
                "routing_level": "high_rescue_candidate",
                "recommended_algorithm": "expert_placeholder",
                "selected_algorithm_path": "high",
                "routing_reasons": reasons
                + [
                    "low_candidate high rescue",
                    f"{LOW_SECOND_STAGE_HIGH_RESCUE_MIN}<=high_count<={LOW_SECOND_STAGE_HIGH_RESCUE_MAX}",
                    f"high_count-low_count>={LOW_SECOND_STAGE_HIGH_RESCUE_GAP}",
                    f"middle_count<={LOW_SECOND_STAGE_MIDDLE_MAX}",
                ],
            }
        )
        return updated

    return updated


def select_count_with_second_stage(
    routing: dict[str, Any],
    *,
    low_count: int,
    middle_count: int,
    high_count: int,
) -> dict[str, Any]:
    updated = apply_low_candidate_second_stage(
        routing,
        low_count=low_count,
        middle_count=middle_count,
        high_count=high_count,
    )
    path = updated.get("selected_algorithm_path") or _path_from_routing_level(
        updated["routing_level"]
    )
    counts = {
        "low": low_count,
        "middle": middle_count,
        "high": high_count,
    }
    return {
        "count": counts[path],
        "selected_algorithm_path": path,
        "routing": updated,
    }


def apply_far_zone_low_middle_guards(
    selection: dict[str, Any],
    *,
    low_count: int,
    middle_count: int,
    high_count: int,
    shoulder_features: dict[str, Any],
    hand_features: dict[str, Any],
) -> dict[str, Any]:
    # Zones 3/4/5 suppress low-score overcounts while protecting stable middle scores.
    selected_count = int(selection["count"])
    selected_path = str(selection["selected_algorithm_path"])
    routing = dict(selection.get("routing", {}))
    reasons = list(routing.get("routing_reasons", []))
    shoulder_segment_count = shoulder_features.get("shoulder_segment_count")
    shoulder_height = shoulder_features.get("shoulder_accepted_height_median_ratio")
    hand_y_range = hand_features.get("hand_y_range_ratio")
    algorithm_range = max(low_count, middle_count, high_count) - min(
        low_count,
        middle_count,
        high_count,
    )

    stable_middle = (
        FAR_ZONE_STABLE_MIDDLE_MIN <= middle_count < FAR_ZONE_STABLE_MIDDLE_MAX
        and algorithm_range <= FAR_ZONE_STABLE_ALGORITHM_RANGE_MAX
        and isinstance(shoulder_segment_count, (int, float))
        and shoulder_segment_count <= FAR_ZONE_STABLE_SEGMENTS_MAX
    )
    if stable_middle:
        routing.update(
            {
                "routing_level": "far_zone_stable_middle_guard",
                "recommended_algorithm": "zone_aware",
                "selected_algorithm_path": "middle",
                "selected_raw_count": middle_count,
                "routing_reasons": reasons
                + [
                    "zone 3/4/5 stable-middle guard",
                    f"{FAR_ZONE_STABLE_MIDDLE_MIN}<=middle<{FAR_ZONE_STABLE_MIDDLE_MAX}",
                    f"algorithm_range<={FAR_ZONE_STABLE_ALGORITHM_RANGE_MAX}",
                    f"shoulder_segments<={FAR_ZONE_STABLE_SEGMENTS_MAX}",
                ],
            }
        )
        return {
            "count": middle_count,
            "selected_algorithm_path": "middle",
            "routing": routing,
        }

    shoulder_low_risk = (
        isinstance(shoulder_height, (int, float))
        and shoulder_height >= FAR_ZONE_LOW_ABNORMAL_SHOULDER_MIN
    )
    hand_low_risk = (
        isinstance(hand_y_range, (int, float))
        and hand_y_range >= FAR_ZONE_LOW_ABNORMAL_HAND_Y_MIN
    )
    low_abnormal = (
        middle_count < FAR_ZONE_LOW_ABNORMAL_MIDDLE_MAX
        and isinstance(shoulder_segment_count, (int, float))
        and shoulder_segment_count >= FAR_ZONE_LOW_ABNORMAL_SEGMENTS_MIN
        and (shoulder_low_risk or hand_low_risk)
    )
    if low_abnormal:
        raw_count = min(low_count, middle_count, high_count, selected_count)
        strong_suppression = (
            (
                isinstance(hand_y_range, (int, float))
                and hand_y_range >= FAR_ZONE_LOW_SUPPRESS_STRONG_HAND_Y_MIN
            )
            or shoulder_segment_count >= FAR_ZONE_LOW_SUPPRESS_STRONG_SEGMENTS_MIN
        )
        factor = (
            FAR_ZONE_LOW_SUPPRESS_STRONG_FACTOR
            if strong_suppression
            else FAR_ZONE_LOW_SUPPRESS_FACTOR
        )
        suppressed_count = int(round(raw_count * factor))
        routing.update(
            {
                "routing_level": "far_zone_low_abnormal_suppressed",
                "recommended_algorithm": "low_guard_placeholder",
                "selected_algorithm_path": selected_path,
                "selected_raw_count": selected_count,
                "suppressed_from_count": raw_count,
                "suppression_factor": factor,
                "routing_reasons": reasons
                + [
                    "zone 3/4/5 low-abnormal suppression",
                    f"middle<{FAR_ZONE_LOW_ABNORMAL_MIDDLE_MAX}",
                    f"shoulder_segments>={FAR_ZONE_LOW_ABNORMAL_SEGMENTS_MIN}",
                    (
                        f"shoulder_height>={FAR_ZONE_LOW_ABNORMAL_SHOULDER_MIN}"
                        if shoulder_low_risk
                        else f"hand_y_range>={FAR_ZONE_LOW_ABNORMAL_HAND_Y_MIN}"
                    ),
                ],
            }
        )
        return {
            "count": suppressed_count,
            "selected_algorithm_path": selected_path,
            "routing": routing,
        }

    return selection


def select_count_with_zone_overrides(
    routing: dict[str, Any],
    *,
    zone_id: int,
    low_count: int,
    middle_count: int,
    high_count: int,
    shoulder_features: dict[str, Any],
    ankle_features: dict[str, Any],
    hand_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Zones 1/2 are clearer and favor the middle path; distant zones use extra guards.
    hand_features = hand_features or {}
    if zone_id not in NEAR_ZONE_IDS:
        selection = select_count_with_second_stage(
            routing,
            low_count=low_count,
            middle_count=middle_count,
            high_count=high_count,
        )
        if zone_id in FAR_ZONE_IDS:
            return apply_far_zone_low_middle_guards(
                selection,
                low_count=low_count,
                middle_count=middle_count,
                high_count=high_count,
                shoulder_features=shoulder_features,
                hand_features=hand_features,
            )
        return selection

    shoulder_height = shoulder_features.get("shoulder_accepted_height_median_ratio")
    ankle_rhythm = ankle_features.get("ankle_rhythm_score")
    selected_path = "middle"
    routing_level = "near_zone_tight_middle"
    reasons = list(routing.get("routing_reasons", [])) + [
        "zone 1/2 tightened: default to middle",
    ]

    if (
        low_count >= NEAR_ZONE_FILTERED_RESCUE_MIN
        and high_count >= NEAR_ZONE_FILTERED_RESCUE_MIN
        and min(low_count, high_count) - middle_count >= NEAR_ZONE_FILTERED_RESCUE_GAP
    ):
        selected_path = "low" if low_count <= high_count else "high"
        routing_level = "near_zone_filtered_rescue"
        reasons += [
            f"low_count/high_count>={NEAR_ZONE_FILTERED_RESCUE_MIN}",
            f"min(low,high)-middle>={NEAR_ZONE_FILTERED_RESCUE_GAP}",
        ]

    if (
        high_count >= NEAR_ZONE_FILTERED_RESCUE_MIN
        and high_count - middle_count >= NEAR_ZONE_STRONG_EXPERT_HIGH_GAP
        and isinstance(shoulder_height, (int, float))
        and shoulder_height < NEAR_ZONE_STRONG_EXPERT_SHOULDER_MAX
    ):
        selected_path = "high"
        routing_level = "near_zone_strong_expert_rescue"
        reasons += [
            f"high-middle>={NEAR_ZONE_STRONG_EXPERT_HIGH_GAP}",
            f"shoulder_height<{NEAR_ZONE_STRONG_EXPERT_SHOULDER_MAX}",
        ]

    if (
        high_count >= NEAR_ZONE_SUBTLE_EXPERT_HIGH_MIN
        and middle_count >= NEAR_ZONE_SUBTLE_EXPERT_MIDDLE_MIN
        and high_count - middle_count >= NEAR_ZONE_SUBTLE_EXPERT_HIGH_GAP
        and isinstance(shoulder_height, (int, float))
        and shoulder_height < NEAR_ZONE_SUBTLE_EXPERT_SHOULDER_MAX
        and isinstance(ankle_rhythm, (int, float))
        and ankle_rhythm < NEAR_ZONE_SUBTLE_EXPERT_ANKLE_RHYTHM_MAX
    ):
        selected_path = "high"
        routing_level = "near_zone_subtle_expert_rescue"
        reasons += [
            f"high>={NEAR_ZONE_SUBTLE_EXPERT_HIGH_MIN}",
            f"middle>={NEAR_ZONE_SUBTLE_EXPERT_MIDDLE_MIN}",
            f"high-middle>={NEAR_ZONE_SUBTLE_EXPERT_HIGH_GAP}",
            f"shoulder_height<{NEAR_ZONE_SUBTLE_EXPERT_SHOULDER_MAX}",
            f"ankle_rhythm<{NEAR_ZONE_SUBTLE_EXPERT_ANKLE_RHYTHM_MAX}",
        ]

    selected_count = {
        "low": low_count,
        "middle": middle_count,
        "high": high_count,
    }[selected_path]
    shoulder_segment_count = shoulder_features.get("shoulder_segment_count")
    algorithm_range = max(low_count, middle_count, high_count) - min(
        low_count,
        middle_count,
        high_count,
    )
    if selected_path in {"low", "middle"}:
        if (
            selected_count < NEAR_ZONE_VERY_LOW_SUPPRESS_MAX
            and isinstance(shoulder_height, (int, float))
            and shoulder_height >= NEAR_ZONE_VERY_LOW_SUPPRESS_SHOULDER_MIN
            and isinstance(shoulder_segment_count, (int, float))
            and shoulder_segment_count >= NEAR_ZONE_VERY_LOW_SUPPRESS_SEGMENTS_MIN
            and algorithm_range <= NEAR_ZONE_VERY_LOW_SUPPRESS_ALGORITHM_RANGE_MAX
        ):
            selected_count = int(round(selected_count * NEAR_ZONE_VERY_LOW_SUPPRESS_FACTOR))
            routing_level = "near_zone_very_low_suppressed"
            reasons += [
                "zone 1/2 very-low over-count suppression",
                f"count<{NEAR_ZONE_VERY_LOW_SUPPRESS_MAX}",
                f"shoulder_height>={NEAR_ZONE_VERY_LOW_SUPPRESS_SHOULDER_MIN}",
                f"shoulder_segments>={NEAR_ZONE_VERY_LOW_SUPPRESS_SEGMENTS_MIN}",
                f"algorithm_range<={NEAR_ZONE_VERY_LOW_SUPPRESS_ALGORITHM_RANGE_MAX}",
            ]
        elif (
            NEAR_ZONE_LOW_SUPPRESS_MIN <= selected_count <= NEAR_ZONE_LOW_SUPPRESS_MAX
            and isinstance(shoulder_height, (int, float))
            and shoulder_height >= NEAR_ZONE_LOW_SUPPRESS_SHOULDER_MIN
            and isinstance(shoulder_segment_count, (int, float))
            and shoulder_segment_count >= NEAR_ZONE_LOW_SUPPRESS_SEGMENTS_MIN
        ):
            selected_count = int(round(selected_count * NEAR_ZONE_LOW_SUPPRESS_FACTOR))
            routing_level = "near_zone_low_suppressed"
            reasons += [
                "zone 1/2 low/middle over-count suppression",
                f"{NEAR_ZONE_LOW_SUPPRESS_MIN}<=count<={NEAR_ZONE_LOW_SUPPRESS_MAX}",
                f"shoulder_height>={NEAR_ZONE_LOW_SUPPRESS_SHOULDER_MIN}",
                f"shoulder_segments>={NEAR_ZONE_LOW_SUPPRESS_SEGMENTS_MIN}",
            ]

    counts = {
        "low": low_count,
        "middle": middle_count,
        "high": high_count,
    }
    updated = {
        **routing,
        "routing_level": routing_level,
        "recommended_algorithm": selected_path,
        "selected_algorithm_path": selected_path,
        "selected_raw_count": counts[selected_path],
        "routing_reasons": reasons,
    }
    return {
        "count": selected_count,
        "selected_algorithm_path": selected_path,
        "routing": updated,
    }


def apply_feet_together_shoulder_fallback(
    selection: dict[str, Any],
    *,
    shoulder_count: int,
    ankle_only_count: int,
    ankle_features: dict[str, Any],
) -> dict[str, Any]:
    ankle_strength = ankle_features.get("ankle_strength_median_ratio")
    if shoulder_count <= 0 or not isinstance(ankle_strength, (int, float)):
        return selection

    ankle_shoulder_ratio = ankle_only_count / shoulder_count
    if (
        ankle_strength >= FEET_TOGETHER_ANKLE_STRENGTH_MAX
        or ankle_shoulder_ratio < FEET_TOGETHER_ANKLE_SHOULDER_RATIO_MIN
    ):
        return selection

    routing = {
        **selection["routing"],
        "routing_level": "feet_together_shoulder_fallback",
        "recommended_algorithm": "shoulder",
        "selected_algorithm_path": "shoulder",
        "selected_raw_count": shoulder_count,
        "routing_reasons": list(selection["routing"].get("routing_reasons", []))
        + [
            "feet-together fallback to shoulder",
            f"ankle_strength<{FEET_TOGETHER_ANKLE_STRENGTH_MAX}",
            f"ankle_only/shoulder>={FEET_TOGETHER_ANKLE_SHOULDER_RATIO_MIN}",
        ],
    }
    return {
        "count": shoulder_count,
        "selected_algorithm_path": "shoulder",
        "routing": routing,
    }


def _path_from_routing_level(routing_level: str) -> str:
    if routing_level in {"low", "low_candidate"}:
        return "low"
    if routing_level in {"high", "high_candidate", "normal_or_high_candidate"}:
        return "high"
    return "middle"


def count_by_zones(
    frames: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    count_mode: str = "shoulder_ankle",
) -> list[dict[str, Any]]:
    # Run low, middle, and high candidates for every zone, then select one through routing.
    if count_mode == "shoulder_ankle":
        counters = {
            int(zone["id"]): {
                "primary": RopeJumpCounter(
                    build_zone_counter_config(int(zone["id"]), "shoulder_ankle")
                ),
                "low": RopeJumpCounter(RopeJumpConfig(count_mode="shoulder")),
                "high": RopeJumpCounter(build_high_counter_config()),
                "no_zoneaware": RopeJumpCounter(RopeJumpConfig(count_mode="shoulder_ankle")),
                "ankle_only": RopeJumpCounter(RopeJumpConfig(count_mode="ankle_only")),
            }
            for zone in zones
        }
    else:
        counters = {
            int(zone["id"]): {
                "primary": RopeJumpCounter(build_zone_counter_config(int(zone["id"]), count_mode)),
            }
            for zone in zones
        }
    last_results: dict[int, CounterResult] = {}
    detections_seen = {int(zone["id"]): 0 for zone in zones}
    selected_poses = {int(zone["id"]): [] for zone in zones}

    for frame in frames:
        pose_result = parse_pose_result(frame)
        for zone in zones:
            zone_id = int(zone["id"])
            candidates = [p for p in pose_result.persons if foot_in_zone(p, zone)]
            if not candidates:
                continue
            best = dataclasses.replace(max(candidates, key=bbox_area), person_id=zone_id)
            detections_seen[zone_id] += 1
            selected_poses[zone_id].append(best)
            for name, counter in counters[zone_id].items():
                result = counter.process_frame(best)
                if name == "primary":
                    last_results[zone_id] = result

    rows = []
    for zone in sorted(zones, key=lambda z: int(z["id"])):
        zone_id = int(zone["id"])
        result = last_results.get(zone_id)
        row = {
            "id": f"zone_{zone_id}",
            "jump_count": result.jump_count if result else 0,
            "frames_with_person": detections_seen[zone_id],
            "current_streak": result.current_streak if result else 0,
            "interrupt_count": result.interrupt_count if result else 0,
        }
        if count_mode == "shoulder_ankle":
            zone_counters = counters[zone_id]
            shoulder_features = summarize_shoulder_features(zone_counters["low"])
            ankle_features = summarize_ankle_features(zone_counters["ankle_only"])
            hand_features = summarize_hand_features(selected_poses[zone_id])
            zone_aware_count = counter_count(zone_counters["primary"])
            ankle_only_count = counter_count(zone_counters["ankle_only"])
            low_count = counter_count(zone_counters["low"])
            high_count = counter_count(zone_counters["high"])
            no_zoneaware_count = counter_count(zone_counters["no_zoneaware"])
            ankle_shoulder_ratio = (
                ankle_only_count / low_count if low_count > 0 else None
            )
            diagnostics = {
                **shoulder_features,
                **ankle_features,
                **hand_features,
                "ankle_vs_zoneaware_gap": ankle_only_count - zone_aware_count,
                "ankle_only_to_shoulder_ratio": ankle_shoulder_ratio,
            }
            routing = build_level_routing(
                zone_id=zone_id,
                zone_aware_count=zone_aware_count,
                ankle_only_count=ankle_only_count,
                shoulder_features=shoulder_features,
                ankle_features=ankle_features,
            )
            selection = select_count_with_zone_overrides(
                routing,
                zone_id=zone_id,
                low_count=low_count,
                middle_count=zone_aware_count,
                high_count=high_count,
                shoulder_features=shoulder_features,
                ankle_features=ankle_features,
                hand_features=hand_features,
            )
            selection = apply_feet_together_shoulder_fallback(
                selection,
                shoulder_count=low_count,
                ankle_only_count=ankle_only_count,
                ankle_features=ankle_features,
            )
            selected_routing = selection["routing"]
            row["raw_primary_count"] = row["jump_count"]
            row["jump_count"] = selection["count"]
            row.update(
                {
                    "initial_level": selected_routing["initial_level"],
                    "routing_level": selected_routing["routing_level"],
                    "recommended_algorithm": selected_routing["recommended_algorithm"],
                    "routing_reasons": selected_routing["routing_reasons"],
                    "selected_algorithm_path": selection["selected_algorithm_path"],
                    "algorithm_counts": {
                        "low": low_count,
                        "middle": zone_aware_count,
                        "high": high_count,
                        "zone_aware": zone_aware_count,
                        "shoulder": low_count,
                        "shoulder_ankle_no_zoneaware": no_zoneaware_count,
                        "ankle_only": ankle_only_count,
                    },
                    "diagnostics": diagnostics,
                    "rule_routing": {
                        "selected_algorithm_path": selection["selected_algorithm_path"],
                        "routing_level": selected_routing["routing_level"],
                        "routing_reasons": selected_routing["routing_reasons"],
                    },
                }
            )
        rows.append(row)
    return rows

def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(f"{'person/zone':<12} {'jumps':>7} {'frames':>8} {'streak':>8} {'interrupts':>11}")
    print("-" * 50)
    for row in rows:
        print(
            f"{row['id']:<12} "
            f"{row['jump_count']:>7} "
            f"{row['frames_with_person']:>8} "
            f"{row['current_streak']:>8} "
            f"{row['interrupt_count']:>11}"
        )


def filter_zone(rows: list[dict[str, Any]], zone_id: int | None) -> list[dict[str, Any]]:
    if zone_id is None:
        return rows
    return [row for row in rows if row["id"] == f"zone_{zone_id}"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if not args.json.exists():
        raise FileNotFoundError(f"Pose JSON does not exist: {args.json}")

    frames = load_frames(args.json)
    zones = load_zones(args.zones)
    rows = filter_zone(count_by_zones(frames, zones, args.count_mode), args.zone)
    summary = {
        "inference_json": str(args.json),
        "mode": "zones",
        "count_mode": args.count_mode,
        "routing_version": "new",
        "results": rows,
    }

    print_table(rows)

    if args.counts_output is not None:
        args.counts_output.parent.mkdir(parents=True, exist_ok=True)
        with args.counts_output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"\nSaved counts: {args.counts_output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from None
