"""Common pose estimation data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


COCO_KEYPOINT_NAMES = [
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


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PersonPose:
    person_id: int
    bbox: tuple[float, float, float, float]
    score: float
    keypoints: list[Keypoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "bbox": list(self.bbox),
            "score": self.score,
            "keypoints": [keypoint.to_dict() for keypoint in self.keypoints],
        }


@dataclass(frozen=True)
class PoseResult:
    frame_index: int | None
    timestamp_sec: float | None
    persons: list[PersonPose]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "persons": [person.to_dict() for person in self.persons],
        }


class PoseEstimator(ABC):
    """Abstract interface for image-to-pose inference."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        *,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
    ) -> PoseResult:
        """Infer human poses from one BGR image."""

    def close(self) -> None:
        """Release resources held by the estimator."""
