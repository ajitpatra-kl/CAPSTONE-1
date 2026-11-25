"""Session-scoped observations and the rule verification interface."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleObservation:
    session_id: str
    frame_number: int
    timestamp: float
    track_id: int
    vehicle_class: str
    confidence: float
    bbox: tuple
    bottom_center: tuple
    estimated_speed: Optional[float] = None
    timing_valid: bool = True
    track_age_frames: int = 0
    missing_frames: int = 0
    lane_id: Optional[int] = None
    signal_context: Optional[dict] = None
    stop_line_distance: Optional[float] = None


@dataclass
class VehicleContext:
    track: object
    observations: deque = field(default_factory=lambda: deque(maxlen=60))
    first_frame: Optional[int] = None

    @property
    def missing_frames(self):
        return self.track.miss_count


@dataclass
class ViolationCandidate:
    event_id: str
    session_id: str
    video_id: str
    track_id: int
    violation_type: str
    timestamp: float
    frame_number: int
    vehicle_class: str
    confidence: float
    bbox: tuple
    trajectory: list
    speed: Optional[float] = None
    speed_limit: Optional[float] = None
    lane_information: Optional[dict] = None
    signal_information: Optional[dict] = None
    evidence_frames: list = field(default_factory=list)
    evidence_paths: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def verify_with_rules(candidate, rule):
    """Return the existing rule's decision for a candidate."""
    return rule()
