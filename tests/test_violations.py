"""
tests/test_violations.py

Unit tests for ViolationDetector.
"""

import sys
import os
import pytest
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.speed_detection.violation_detector import ViolationDetector, ViolationRecord
from modules.speed_detection.vehicle_tracker import TrackState


def make_track(track_id=1, class_name="car") -> TrackState:
    state = TrackState(track_id=track_id, class_id=2, class_name=class_name)
    state.bboxes.append((100, 100, 200, 200))
    for _ in range(10):
        state.speed_samples_kmh.append(45.0)
    return state


class TestViolationDetector:

    def test_no_violation_below_limit(self):
        vd = ViolationDetector(speed_limit_kmh=50.0, min_violation_duration_sec=1.0, confirmation_frames=3)
        state = make_track(1)
        
        result = None
        for frame in range(50):
            result = vd.update(state, 45.0, frame, frame * 0.04, is_calibrated=True)
        
        assert result is None
        assert len(vd.violations) == 0

    def test_violation_above_limit_long_enough(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=1.0, confirmation_frames=3)
        state = make_track(1)
        
        result = None
        # 60 frames at 25 FPS = 2.4 seconds above limit
        for frame in range(60):
            ts = frame * 0.04
            result = vd.update(state, 60.0, frame, ts, is_calibrated=True)
        
        assert result is not None or len(vd.violations) > 0, "Should have flagged a violation"

    def test_no_violation_too_short(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=3.0, confirmation_frames=10)
        state = make_track(1)
        
        # Only 1 second above limit
        for frame in range(25):
            vd.update(state, 60.0, frame, frame * 0.04, is_calibrated=True)
        
        assert len(vd.violations) == 0

    def test_violation_record_fields(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=0.5, confirmation_frames=2)
        state = make_track(99)
        
        for frame in range(30):
            ts = frame * 0.04
            v = vd.update(state, 65.0, frame, ts, is_calibrated=True)
            if v is not None:
                assert v.track_id == 99
                assert v.speed_kmh >= 65.0
                assert v.speed_limit_kmh == 40.0
                assert v.excess_kmh > 0
                assert v.timestamp_sec >= 0
                break

    def test_no_duplicate_violations(self):
        """Same track should not get multiple violations in the same exceedance."""
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=0.5, confirmation_frames=2)
        state = make_track(1)
        
        for frame in range(100):
            vd.update(state, 60.0, frame, frame * 0.04, is_calibrated=True)
        
        # One uninterrupted event yields one record.
        assert len(vd.violations) == 1, f"Too many violations: {len(vd.violations)}"

    def test_violation_resets_after_complying(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=0.5, confirmation_frames=2)
        state = make_track(1)
        
        # Speeding, then complying, then speeding again
        for frame in range(30):
            vd.update(state, 65.0, frame, frame * 0.04)
        for frame in range(30, 60):
            vd.update(state, 30.0, frame, frame * 0.04)  # complying
        for frame in range(60, 100):
            vd.update(state, 65.0, frame, frame * 0.04)  # speeding again
        
        # The observed return below the limit starts a new event.
        assert len(vd.violations) == 2

    def test_speed_limit_update(self):
        vd = ViolationDetector(speed_limit_kmh=40.0)
        vd.update_speed_limit(80.0)
        assert vd.speed_limit_kmh == 80.0
        assert vd.effective_limit == 80.0

    def test_is_currently_speeding(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=5.0)
        state = make_track(5)
        vd.update(state, 60.0, 1, 0.04)
        assert vd.is_currently_speeding(5)
        
        vd.update(state, 30.0, 2, 0.08)  # drops below limit
        assert not vd.is_currently_speeding(5)

    def test_to_dict(self):
        v = ViolationRecord(
            track_id=7, class_id=2, class_name="car",
            speed_kmh=65.5, speed_limit_kmh=50.0, excess_kmh=15.5,
            timestamp_sec=12.3, frame_number=307, is_calibrated=True
        )
        d = v.to_dict()
        assert d["track_id"] == 7
        assert d["speed_kmh"] == 65.5
        assert d["excess_kmh"] == 15.5

    def test_reset(self):
        vd = ViolationDetector(speed_limit_kmh=40.0, min_violation_duration_sec=0.5, confirmation_frames=2)
        state = make_track(1)
        for frame in range(50):
            vd.update(state, 60.0, frame, frame * 0.04)
        
        vd.reset()
        assert len(vd.violations) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
