"""Base interfaces for activity counters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from posecount.pose.base import PersonPose


@dataclass
class CounterResult:
    jump_count: int
    frame_count: int
    current_streak: int
    interrupt_count: int


class BaseCounter(ABC):
    """Accumulate counts frame by frame for the PersonPose selected by the pipeline."""

    @abstractmethod
    def process_frame(self, person: PersonPose) -> CounterResult:
        """Process the selected person's pose for one frame and return the current count."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the counter's internal state."""
