"""
modules/speed_detection/violation_detector.py

Overspeed violation detection with false-positive suppression.

A violation is confirmed only when:
 1. The vehicle's smoothed speed exceeds the speed limit
 2. This persists for at least `min_violation_duration_seconds`
 3. The speed estimate is based on at least `min_speed_samples` frames

One violation record per vehicle per speed limit exceedance event.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .vehicle_tracker import TrackState

logger = logging.getLogger(__name__)


@dataclass
class ViolationRecord:
    """A confirmed overspeed violation event."""
    track_id: int
    class_id: int
    class_name: str
    speed_kmh: float
    speed_limit_kmh: float
    excess_kmh: float
    timestamp_sec: float
    frame_number: int
    bbox: Optional[Tuple] = None
    evidence_path: Optional[str] = None
    is_calibrated: bool = True
    original_frame_path: Optional[str] = None
    vehicle_number: str = ""
    plate_confidence: Optional[float] = None
    anpr_status: str = ""
    vehicle_crop_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    plate_bbox: Optional[Tuple] = None
    plate_frame_number: Optional[int] = None
    plate_detection_confidence: Optional[float] = None
    plate_raw_text: str = ""
    plate_evidence_path: Optional[str] = None
    plate_candidate: str = ""
    plate_candidate_format: str = ""
    plate_candidate_frames: tuple = ()
    plate_character_agreement: Optional[float] = None
    plate_engine_results: dict = field(default_factory=dict)
    plate_enhanced_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "speed_kmh": round(self.speed_kmh, 2),
            "speed_limit_kmh": self.speed_limit_kmh,
            "excess_kmh": round(self.excess_kmh, 2),
            "timestamp_sec": round(self.timestamp_sec, 3),
            "frame_number": self.frame_number,
            "evidence_path": self.evidence_path or "",
            "is_calibrated": self.is_calibrated,
            "original_frame_path": self.original_frame_path or "",
            "bbox": self.bbox,
            "status": "Confirmed",
            "vehicle_number": self.vehicle_number,
            "plate_confidence": self.plate_confidence,
            "anpr_status": self.anpr_status,
            "vehicle_crop_path": self.vehicle_crop_path or "",
            "plate_crop_path": self.plate_crop_path or "",
            "plate_bbox": self.plate_bbox,
            "plate_frame_number": self.plate_frame_number,
            "plate_detection_confidence": self.plate_detection_confidence,
            "plate_raw_text": self.plate_raw_text,
            "plate_evidence_path": self.plate_evidence_path or "",
            "plate_candidate": self.plate_candidate,
            "plate_candidate_format": self.plate_candidate_format,
            "plate_candidate_frames": self.plate_candidate_frames,
            "plate_character_agreement": self.plate_character_agreement,
            "plate_engine_results": self.plate_engine_results,
            "plate_enhanced_path": self.plate_enhanced_path or "",
        }


class ViolationDetector:
    """
    Detects overspeed violations from tracked vehicle speed estimates.
    
    Implements:
     - Configurable speed limit
     - Minimum exceedance duration requirement
     - Confirmation frame count
     - One-violation-per-event deduplication
     - Speed buffer analysis to suppress transient spikes
    """

    def __init__(
        self,
        speed_limit_kmh: float = 40.0,
        min_violation_duration_sec: float = 1.0,
        confirmation_frames: int = 3,
        speed_buffer_kmh: float = 0.0,
        allow_estimated: bool = False,
    ):
        """
        Args:
            speed_limit_kmh: Speed threshold for violations
            min_violation_duration_sec: Vehicle must exceed limit for this long
            confirmation_frames: Minimum consecutive frames above limit
            speed_buffer_kmh: Additional buffer above limit (reduces false positives)
            allow_estimated: Allow configured approximate speeds to trigger alerts
        """
        self.speed_limit_kmh = speed_limit_kmh
        self.min_violation_duration_sec = min_violation_duration_sec
        self.confirmation_frames = confirmation_frames
        self.speed_buffer_kmh = speed_buffer_kmh
        self.allow_estimated = allow_estimated
        self.effective_limit = speed_limit_kmh + speed_buffer_kmh
        
        # Per-track: timestamp when vehicle first exceeded limit
        self._exceedance_start: Dict[int, float] = {}
        # Per-track: consecutive frames above limit
        self._consecutive_frames: Dict[int, int] = {}
        # Keep a confirmed event latched through temporary observation loss.
        self._violation_issued: Dict[int, bool] = {}

        self.violations: List[ViolationRecord] = []

    def update(
        self,
        state: TrackState,
        current_speed_kmh: Optional[float],
        frame_number: int,
        timestamp_sec: float,
        is_calibrated: bool = True,
        confirm=None,
    ) -> Optional[ViolationRecord]:
        """
        Update violation status for one vehicle track.
        
        Args:
            state: Current TrackState
            current_speed_kmh: Latest speed estimate (None if unknown)
            frame_number: Current video frame number
            timestamp_sec: Current video timestamp
            is_calibrated: Whether the speed estimate uses calibrated homography
            confirm: Optional candidate callback. The legacy rule owns episode state.
        
        Returns:
            ViolationRecord if a NEW violation is confirmed, else None
        """
        tid = state.track_id
        
        if ((not is_calibrated and not self.allow_estimated) or current_speed_kmh is None
                or not math.isfinite(current_speed_kmh) or current_speed_kmh < 0):
            # No speed info → reset exceedance counter
            self.interrupt_track(tid)
            state.frames_above_limit = 0
            return None
        
        is_speeding = current_speed_kmh > self.effective_limit
        
        if is_speeding:
            # Update exceedance tracking
            if tid not in self._exceedance_start:
                self._exceedance_start[tid] = timestamp_sec
                self._consecutive_frames[tid] = 1
            else:
                self._consecutive_frames[tid] = self._consecutive_frames.get(tid, 0) + 1
            
            state.frames_above_limit = self._consecutive_frames.get(tid, 0)
            
            # Check if violation is confirmed
            duration = timestamp_sec - self._exceedance_start.get(tid, timestamp_sec)
            frames_above = self._consecutive_frames.get(tid, 0)
            
            already_issued = self._violation_issued.get(tid, False)
            
            if (
                duration >= self.min_violation_duration_sec
                and frames_above >= self.confirmation_frames
                and not already_issued
            ):
                def rule():
                    return len(state.speed_samples_kmh) >= 3

                record = ViolationRecord(
                    track_id=tid,
                    class_id=state.class_id,
                    class_name=state.class_name,
                    speed_kmh=current_speed_kmh,
                    speed_limit_kmh=self.speed_limit_kmh,
                    excess_kmh=current_speed_kmh - self.speed_limit_kmh,
                    timestamp_sec=timestamp_sec,
                    frame_number=frame_number,
                    bbox=state.latest_bbox,
                    is_calibrated=is_calibrated,
                )
                
                if confirm is not None:
                    confirm(record, rule, {"duration_above_limit": duration, "consecutive_frames": frames_above,
                                           "speed_samples": len(state.speed_samples_kmh)})
                if not rule():
                    return None
                self.violations.append(record)
                state.violation_confirmed = True
                state.violation_recorded = True
                self._violation_issued[tid] = True
                
                logger.info(
                    f"🚨 VIOLATION: Track #{tid} ({state.class_name}) | "
                    f"{current_speed_kmh:.1f} km/h > {self.speed_limit_kmh} km/h | "
                    f"Duration: {duration:.1f}s | Frame: {frame_number}"
                )
                return record
        else:
            # Vehicle is within limit — reset exceedance
            self._reset_exceedance(tid)
            self._violation_issued.pop(tid, None)
            state.frames_above_limit = 0
        
        return None

    def interrupt_track(self, track_id: int):
        """Discard pending duration when timing or tracking continuity is lost."""
        self._reset_exceedance(track_id)

    def _reset_exceedance(self, track_id: int):
        """Reset the exceedance tracking for a track."""
        self._exceedance_start.pop(track_id, None)
        self._consecutive_frames.pop(track_id, None)

    def update_speed_limit(self, new_limit_kmh: float):
        """Update the speed limit and reset all exceedance counters."""
        self.speed_limit_kmh = new_limit_kmh
        self.effective_limit = new_limit_kmh + self.speed_buffer_kmh
        self._exceedance_start.clear()
        self._consecutive_frames.clear()
        self._violation_issued.clear()
        logger.info(f"Speed limit updated to {new_limit_kmh} km/h")

    def get_violations(self) -> List[ViolationRecord]:
        return list(self.violations)

    def get_violations_as_dicts(self) -> List[dict]:
        return [v.to_dict() for v in self.violations]

    def reset(self):
        """Reset all state."""
        self._exceedance_start.clear()
        self._consecutive_frames.clear()
        self._violation_issued.clear()
        self.violations.clear()

    def is_event_confirmed(self, track_id: int) -> bool:
        return self._violation_issued.get(track_id, False)

    def is_currently_speeding(self, track_id: int) -> bool:
        """Check if track is currently in an exceedance window."""
        return track_id in self._exceedance_start
