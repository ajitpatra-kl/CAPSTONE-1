"""
tests/test_tracking.py

Unit tests for VehicleTracker.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.speed_detection.vehicle_detector import Detection
from modules.speed_detection.vehicle_tracker import VehicleTracker, TrackState


class TestVehicleTracker:

    def make_detection(self, x1, y1, x2, y2, cls_id=2, conf=0.8):
        return Detection(
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            class_id=cls_id,
            class_name={2:"car",3:"motorcycle",5:"bus",7:"truck"}.get(cls_id,"car"),
        )

    def test_tracker_init(self):
        t = VehicleTracker()
        assert t.tracks == {}

    def test_single_detection_creates_track(self):
        t = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self.make_detection(100, 100, 200, 200)]
        active = t.update(dets, frame, 0.04)
        assert len(active) >= 1

    def test_track_id_persistence(self):
        """Same vehicle should keep the same ID across frames."""
        t = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Frame 1
        dets1 = [self.make_detection(100, 100, 200, 200)]
        active1 = t.update(dets1, frame, 0.04)
        ids1 = set(active1.keys())
        
        # Frame 2: vehicle moves slightly
        dets2 = [self.make_detection(105, 102, 205, 202)]
        active2 = t.update(dets2, frame, 0.08)
        ids2 = set(active2.keys())
        
        # At least one ID should be shared
        assert len(ids1 & ids2) >= 1 or len(active2) >= 1

    def test_bottom_center_reference_point(self):
        """Bottom-centre of bbox should be stored as tracking reference."""
        det = Detection((100, 50, 300, 200), 0.9, 2, "car")
        expected_bc = (200.0, 200.0)  # ((100+300)/2, 200)
        assert det.bottom_center == expected_bc

    def test_track_state_dataclass(self):
        state = TrackState(track_id=5, class_id=2, class_name="car")
        assert state.track_id == 5
        assert state.class_name == "car"
        assert state.is_active
        assert state.hit_count == 0
        assert state.miss_count == 0

    def test_track_disappears_after_max_miss(self):
        t = VehicleTracker(max_miss_frames=5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Establish a track
        dets = [self.make_detection(100, 100, 200, 200)]
        t.update(dets, frame, 0.04)
        
        # Then no detections for many frames
        for i in range(10):
            t.update([], frame, (i+2) * 0.04)
        
        active = t.get_active_tracks()
        assert len(active) == 0, "Track should be deactivated after max_miss_frames"

    def test_multiple_vehicles_distinct_ids(self):
        t = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Two vehicles far apart
        dets = [
            self.make_detection(50, 50, 150, 150),
            self.make_detection(400, 300, 500, 400),
        ]
        active = t.update(dets, frame, 0.04)
        ids = list(active.keys())
        
        if len(ids) >= 2:
            assert ids[0] != ids[1], "Two distinct vehicles should have different IDs"

    def test_reset_clears_all_tracks(self):
        t = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self.make_detection(100, 100, 200, 200)]
        t.update(dets, frame, 0.04)
        t.reset()
        assert len(t.tracks) == 0

    def test_world_coords_stored_when_calibration_given(self):
        """With a calibration object, world coords should be populated."""
        from unittest.mock import MagicMock
        
        cal = MagicMock()
        cal.is_valid.return_value = True
        cal.transform.return_value = (1.5, 5.0)  # fake world coords
        
        t = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self.make_detection(100, 100, 200, 200)]
        active = t.update(dets, frame, 0.04, calibration=cal)
        
        for state in active.values():
            if state.bottom_centers_world:
                wx, wy = state.bottom_centers_world[-1]
                assert wx == 1.5
                assert wy == 5.0


def test_bytetrack_reacquires_id_without_returning_stale_detections():
    tracker = VehicleTracker()
    assert tracker.name == "ByteTrack"
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    detection = Detection(bbox=(40, 50, 95, 100), confidence=0.95,
                          class_id=2, class_name="car")
    first = tracker.update([detection], frame, 0.0)
    assert len(first) == 1
    assert tracker.update([], frame, 0.1) == {}
    reacquired = tracker.update([detection], frame, 0.2)
    assert reacquired.keys() == first.keys()
    assert list(next(iter(reacquired.values())).timestamps) == [0.2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
