"""Rope jump counter — detects jumps via shoulder trajectory signal analysis."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from posecount.counters.base import BaseCounter, CounterResult
from posecount.pose.base import Keypoint, PersonPose


# Internal enums

class _ExtremaType(Enum):
    VALLEY = auto()
    PEAK = auto()


class _JudgeResult(Enum):
    ACCEPTED = auto()
    INVALID_KEYPOINTS = auto()
    PATTERN_NOT_MATCH = auto()
    ANKLE_PATTERN_NOT_MATCH = auto()
    HEIGHT_TOO_SMALL = auto()
    HEIGHT_TOO_LARGE = auto()
    DURATION_TOO_SHORT = auto()
    DURATION_TOO_LONG = auto()


# Internal data structures

@dataclass
class _ExtremaPoint:
    frame_index: int
    value: float
    type: _ExtremaType


@dataclass
class _JumpCandidate:
    start_valley: _ExtremaPoint
    peak: _ExtremaPoint
    end_valley: _ExtremaPoint
    judge_result: _JudgeResult = _JudgeResult.ACCEPTED


@dataclass
class _SignalSample:
    frame_index: int
    mid_shoulder_y: float
    left_ankle_y: Optional[float] = None
    right_ankle_y: Optional[float] = None
    ankle_diff: Optional[float] = None


# Public configuration

@dataclass
class RopeJumpConfig:
    count_mode: str = "shoulder"  # "shoulder", "shoulder_ankle", or "ankle_only"
    min_confidence: float = 0.1
    extrema_window_size: int = 7
    min_frame_interval: int = 5
    min_symmetry_ratio: float = 0.12
    min_jump_height_ratio: float = 0.05
    recommended_jump_height_ratio: float = 0.15
    max_jump_height_ratio: float = 0.60
    min_jump_duration: int = 4
    max_jump_duration: int = 30
    enable_smoothing: bool = False
    smoothing_window_size: int = 3
    epsilon: float = 1e-6
    max_frame_index: int = -1  # -1 means unlimited
    ankle_extrema_window_size: int = 7
    min_ankle_diff_ratio: float = 0.005
    min_ankle_event_interval: int = 10
    max_ankle_event_interval: int = 35
    ankle_phase_offsets: Tuple[int, ...] = (2, 4, -4, 0)
    ankle_phase_tolerance_frames: int = 2
    ankle_reject_diff_ratio: float = 0.015
    min_ankle_reject_samples: int = 2


# Counter

class RopeJumpCounter(BaseCounter):
    """Count rope jumps by analyzing the shoulder trajectory signal.

    Usage::

        counter = RopeJumpCounter()
        for frame in ...:
            pose_result = estimator.predict(frame.image, ...)
            result = counter.process_frame(pose_result)
            print(result.jump_count)
    """

    def __init__(self, config: Optional[RopeJumpConfig] = None) -> None:
        self.config = config if config is not None else RopeJumpConfig()
        self.reset()

    # BaseCounter interface

    def process_frame(self, person: PersonPose) -> CounterResult:
        """Process the selected person's pose for one frame.

        The pipeline caller chooses the tracked person and passes the matching
        PersonPose. The frame index still advances when keypoints are missing,
        but the frame does not contribute to the count.
        """
        frame_index = self._next_frame_index

        if self.config.max_frame_index >= 0 and frame_index >= self.config.max_frame_index:
            self.reset()

        self._next_frame_index += 1

        if len(person.keypoints) < 17:
            return self._build_result()

        kps = person.keypoints
        ls, rs, lh, rh = kps[5], kps[6], kps[11], kps[12]  # shoulders & hips
        la, ra = kps[15], kps[16]  # ankles

        if not all(self._valid(kp) for kp in (ls, rs, lh, rh)):
            return self._build_result()

        if not self._init_body_height_ref(ls, rs, lh, rh):
            return self._build_result()

        mid_shoulder_y = -(ls.y + rs.y) / 2.0
        left_ankle_y = -la.y if self._valid(la) else None
        right_ankle_y = -ra.y if self._valid(ra) else None
        return self._process_signal(frame_index, mid_shoulder_y, left_ankle_y, right_ankle_y)

    def reset(self) -> None:
        self._next_frame_index: int = 0
        self._jump_count: int = 0
        self._current_streak: int = 0
        self._interrupt_count: int = 0
        self._body_height_ref: float = 0.0
        self._history: List[_SignalSample] = []
        self._extrema: List[_ExtremaPoint] = []
        self._candidates: List[_JumpCandidate] = []
        self._extrema_scan_start: int = 0
        self._smooth_buf: deque[float] = deque()
        self._ankle_smooth_buf: deque[float] = deque()
        self._ankle_extrema: List[_ExtremaPoint] = []
        self._last_accepted_ankle_extremum: Optional[_ExtremaPoint] = None
        self._last_shoulder_ankle_sign: Optional[int] = None

    # Core processing

    def _process_signal(
        self,
        frame_index: int,
        mid_shoulder_y: float,
        left_ankle_y: Optional[float] = None,
        right_ankle_y: Optional[float] = None,
    ) -> CounterResult:
        self._append_sample(frame_index, mid_shoulder_y, left_ankle_y, right_ankle_y)

        if self.config.count_mode == "ankle_only":
            return self._process_ankle_only()

        if len(self._history) < self.config.extrema_window_size:
            return self._build_result()

        extremum = self._detect_extrema()
        if extremum is None:
            return self._build_result()

        self._merge_or_append(extremum)

        candidate = self._detect_candidate()
        if candidate is None:
            return self._build_result()

        self._validate(candidate)
        if (
            candidate.judge_result == _JudgeResult.ACCEPTED
            and self.config.count_mode == "shoulder_ankle"
        ):
            self._validate_ankle_alternation_for_candidate(candidate)
        self._candidates.append(candidate)

        if candidate.judge_result == _JudgeResult.ACCEPTED:
            self._jump_count += 1
            is_continuous = (
                len(self._candidates) >= 2
                and self._candidates[-2].judge_result == _JudgeResult.ACCEPTED
                and candidate.start_valley.frame_index == self._candidates[-2].end_valley.frame_index
            )
            self._current_streak = self._current_streak + 1 if is_continuous else 1
        elif self._current_streak > 0:
            self._interrupt_count += 1
            self._current_streak = 0

        return self._build_result()

    # Validation and initialization

    def _valid(self, kp: Keypoint) -> bool:
        return (
            kp.score >= self.config.min_confidence
            and math.isfinite(kp.x)
            and math.isfinite(kp.y)
        )

    def _init_body_height_ref(
        self, ls: Keypoint, rs: Keypoint, lh: Keypoint, rh: Keypoint
    ) -> bool:
        if self._body_height_ref > 0.0:
            return True
        body_height = abs((ls.y + rs.y) / 2.0 - (lh.y + rh.y) / 2.0)
        if body_height <= self.config.epsilon:
            return False
        self._body_height_ref = body_height
        return True

    # Signal buffering and smoothing

    def _append_sample(
        self,
        frame_index: int,
        mid_shoulder_y: float,
        left_ankle_y: Optional[float] = None,
        right_ankle_y: Optional[float] = None,
    ) -> None:
        value = mid_shoulder_y
        if self.config.enable_smoothing:
            self._smooth_buf.append(mid_shoulder_y)
            while len(self._smooth_buf) > self.config.smoothing_window_size:
                self._smooth_buf.popleft()
            value = sum(self._smooth_buf) / len(self._smooth_buf)

        ankle_diff = None
        if left_ankle_y is not None and right_ankle_y is not None:
            ankle_diff = left_ankle_y - right_ankle_y
            if self.config.enable_smoothing:
                self._ankle_smooth_buf.append(ankle_diff)
                while len(self._ankle_smooth_buf) > self.config.smoothing_window_size:
                    self._ankle_smooth_buf.popleft()
                ankle_diff = sum(self._ankle_smooth_buf) / len(self._ankle_smooth_buf)

        self._history.append(
            _SignalSample(frame_index, value, left_ankle_y, right_ankle_y, ankle_diff)
        )

    # Extremum detection and merging

    def _detect_extrema(self) -> Optional[_ExtremaPoint]:
        n, ws = len(self._history), self.config.extrema_window_size
        if n < ws:
            return None
        window = self._history[n - ws:]
        center = window[ws // 2]
        values = [s.mid_shoulder_y for s in window]
        min_v, max_v = min(values), max(values)
        if max_v - min_v <= self.config.epsilon:
            return None
        eps = self.config.epsilon
        is_valley = abs(center.mid_shoulder_y - min_v) <= eps
        is_peak = abs(center.mid_shoulder_y - max_v) <= eps
        if is_valley and not is_peak:
            return _ExtremaPoint(center.frame_index, center.mid_shoulder_y, _ExtremaType.VALLEY)
        if is_peak and not is_valley:
            return _ExtremaPoint(center.frame_index, center.mid_shoulder_y, _ExtremaType.PEAK)
        return None

    def _merge_or_append(self, extremum: _ExtremaPoint) -> None:
        if self._extrema:
            last = self._extrema[-1]
            if last.type == extremum.type:
                if extremum.frame_index - last.frame_index < self.config.min_frame_interval:
                    if self._more_significant(extremum, last):
                        self._extrema[-1] = extremum
                    return
        self._extrema.append(extremum)

    # Candidate detection and validation

    def _detect_candidate(self) -> Optional[_JumpCandidate]:
        i = self._extrema_scan_start
        if i > len(self._extrema) - 4:
            return None
        self._extrema_scan_start += 1
        sv, pk, ev = self._extrema[i], self._extrema[i + 1], self._extrema[i + 2]
        if sv.type != _ExtremaType.VALLEY:
            return None
        if pk.type != _ExtremaType.PEAK:
            return None
        if ev.type != _ExtremaType.VALLEY:
            return None
        return _JumpCandidate(sv, pk, ev)

    def _validate(self, c: _JumpCandidate) -> None:
        left_dur = c.peak.frame_index - c.start_valley.frame_index
        right_dur = c.end_valley.frame_index - c.peak.frame_index
        total_dur = left_dur + right_dur

        if total_dur < self.config.min_jump_duration:
            c.judge_result = _JudgeResult.DURATION_TOO_SHORT
            return
        if total_dur > self.config.max_jump_duration:
            c.judge_result = _JudgeResult.DURATION_TOO_LONG
            return

        shorter, longer = min(left_dur, right_dur), max(left_dur, right_dur)
        if longer <= 0 or shorter / longer < self.config.min_symmetry_ratio:
            c.judge_result = _JudgeResult.PATTERN_NOT_MATCH
            return

        mean_height = (
            (c.peak.value - c.start_valley.value) + (c.peak.value - c.end_valley.value)
        ) / 2.0
        min_h = self._body_height_ref * self.config.min_jump_height_ratio
        max_h = self._body_height_ref * self.config.max_jump_height_ratio

        if mean_height < min_h:
            c.judge_result = _JudgeResult.HEIGHT_TOO_SMALL
            return
        if mean_height > max_h:
            c.judge_result = _JudgeResult.HEIGHT_TOO_LARGE
            return

        c.judge_result = _JudgeResult.ACCEPTED

    def _validate_ankle_alternation_for_candidate(self, c: _JumpCandidate) -> None:
        sign = self._candidate_ankle_sign(c)
        if sign is None:
            return
        if sign == 0:
            c.judge_result = _JudgeResult.ANKLE_PATTERN_NOT_MATCH
            return
        self._last_shoulder_ankle_sign = sign

    def _candidate_ankle_sign(self, c: _JumpCandidate) -> Optional[int]:
        baseline = self._ankle_diff_baseline(c.end_valley.frame_index)
        if baseline is None:
            return None

        raw_values = [
            value
            for value in (
                self._ankle_phase_value(c.peak.frame_index + offset, baseline)
                for offset in self.config.ankle_phase_offsets
            )
            if value is not None
        ]
        values = [value for value in raw_values if abs(value) >= self._ankle_diff_threshold()]
        if not values:
            return None

        if self._last_shoulder_ankle_sign is not None:
            alternating_values = [
                value
                for value in values
                if value * self._last_shoulder_ankle_sign < 0
            ]
            if alternating_values:
                value = max(alternating_values, key=abs)
                return 1 if value > 0 else -1
            reject_threshold = self._ankle_reject_threshold()
            same_side_strong = [
                value
                for value in raw_values
                if (
                    value * self._last_shoulder_ankle_sign > 0
                    and abs(value) >= reject_threshold
                )
            ]
            if len(same_side_strong) >= self.config.min_ankle_reject_samples:
                return 0
            return None

        value = max(values, key=abs)
        return 1 if value > 0 else -1

    def _ankle_phase_value(self, target_frame: int, baseline: float) -> Optional[float]:
        samples = [
            sample
            for sample in self._history
            if (
                sample.ankle_diff is not None
                and abs(sample.frame_index - target_frame) <= self.config.ankle_phase_tolerance_frames
            )
        ]
        if not samples:
            return None
        sample = min(samples, key=lambda item: abs(item.frame_index - target_frame))
        return (sample.ankle_diff or 0.0) - baseline

    def _process_ankle_only(self) -> CounterResult:
        if len(self._history) < self.config.ankle_extrema_window_size:
            return self._build_result()

        extremum = self._detect_ankle_extrema()
        if extremum is None:
            return self._build_result()

        accepted = self._merge_or_append_ankle_extremum(extremum)
        if accepted is None:
            return self._build_result()

        if self._is_valid_ankle_event(accepted):
            self._jump_count += 1
            self._current_streak += 1
            self._last_accepted_ankle_extremum = accepted
        else:
            self._restart_ankle_sequence_if_needed(accepted)

        return self._build_result()

    def _detect_ankle_extrema(self) -> Optional[_ExtremaPoint]:
        ws = self.config.ankle_extrema_window_size
        window = self._history[-ws:]
        if any(sample.ankle_diff is None for sample in window):
            return None

        baseline = self._ankle_diff_baseline(window[-1].frame_index)
        if baseline is None:
            return None

        center = window[ws // 2]
        values = [(sample.ankle_diff or 0.0) - baseline for sample in window]
        min_v, max_v = min(values), max(values)
        if max_v - min_v < self._ankle_diff_threshold():
            return None

        value = (center.ankle_diff or 0.0) - baseline
        eps = self.config.epsilon
        is_valley = abs(value - min_v) <= eps
        is_peak = abs(value - max_v) <= eps
        if is_valley and not is_peak:
            return _ExtremaPoint(center.frame_index, value, _ExtremaType.VALLEY)
        if is_peak and not is_valley:
            return _ExtremaPoint(center.frame_index, value, _ExtremaType.PEAK)
        return None

    def _merge_or_append_ankle_extremum(self, extremum: _ExtremaPoint) -> Optional[_ExtremaPoint]:
        if abs(extremum.value) < self._ankle_diff_threshold():
            return None

        if self._ankle_extrema:
            last = self._ankle_extrema[-1]
            if last.type == extremum.type:
                frame_gap = extremum.frame_index - last.frame_index
                if frame_gap < self.config.min_ankle_event_interval:
                    if self._more_significant(extremum, last):
                        self._ankle_extrema[-1] = extremum
                        return extremum
                    return None

        self._ankle_extrema.append(extremum)
        return extremum

    def _is_valid_ankle_event(self, extremum: _ExtremaPoint) -> bool:
        if self._last_accepted_ankle_extremum is None:
            return True

        previous = self._last_accepted_ankle_extremum
        frame_gap = extremum.frame_index - previous.frame_index
        if frame_gap < self.config.min_ankle_event_interval:
            return False
        if frame_gap > self.config.max_ankle_event_interval:
            return False
        return extremum.type != previous.type or extremum.value * previous.value < 0

    def _restart_ankle_sequence_if_needed(self, extremum: _ExtremaPoint) -> None:
        previous = self._last_accepted_ankle_extremum
        if previous is None:
            self._last_accepted_ankle_extremum = extremum
            return

        frame_gap = extremum.frame_index - previous.frame_index
        if frame_gap > self.config.max_ankle_event_interval:
            self._last_accepted_ankle_extremum = extremum
        elif frame_gap >= self.config.min_ankle_event_interval:
            self._last_accepted_ankle_extremum = extremum

        if self._current_streak > 0:
            self._interrupt_count += 1
            self._current_streak = 0

    def _ankle_diff_threshold(self) -> float:
        return max(self.config.epsilon, self._body_height_ref * self.config.min_ankle_diff_ratio)

    def _ankle_reject_threshold(self) -> float:
        return max(self._ankle_diff_threshold(), self._body_height_ref * self.config.ankle_reject_diff_ratio)

    def _ankle_diff_baseline(self, end_frame: int) -> Optional[float]:
        values = [
            sample.ankle_diff
            for sample in self._history
            if sample.frame_index <= end_frame and sample.ankle_diff is not None
        ]
        if not values:
            return None
        return statistics.median(values)

    # Helpers

    def _build_result(self) -> CounterResult:
        return CounterResult(
            jump_count=self._jump_count,
            frame_count=len(self._history),
            current_streak=self._current_streak,
            interrupt_count=self._interrupt_count,
        )

    @staticmethod
    def _more_significant(cur: _ExtremaPoint, prev: _ExtremaPoint) -> bool:
        return cur.value < prev.value if cur.type == _ExtremaType.VALLEY else cur.value > prev.value
