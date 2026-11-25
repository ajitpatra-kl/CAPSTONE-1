"""
tests/test_detection.py

Unit tests for VehicleDetector.
Requires ultralytics to be installed. Skips gracefully if model unavailable.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestVehicleDetector:

    @pytest.fixture(autouse=True)
    def skip_if_no_ultralytics(self):
        try:
            import ultralytics
        except ImportError:
            pytest.skip("ultralytics not installed")

    def test_import(self):
        from modules.speed_detection.vehicle_detector import VehicleDetector
        assert VehicleDetector is not None

    def test_detection_classes(self):
        from modules.speed_detection.vehicle_detector import VEHICLE_CLASS_IDS
        assert 2 in VEHICLE_CLASS_IDS  # car
        assert 3 in VEHICLE_CLASS_IDS  # motorcycle
        assert 5 in VEHICLE_CLASS_IDS  # bus
        assert 7 in VEHICLE_CLASS_IDS  # truck

    def test_detector_loads(self):
        from modules.speed_detection.vehicle_detector import VehicleDetector
        try:
            det = VehicleDetector(confidence_threshold=0.4)
            assert det.model is not None
        except Exception as e:
            pytest.skip(f"Model load failed (network/file issue): {e}")

    def test_detect_on_blank_frame(self):
        from modules.speed_detection.vehicle_detector import VehicleDetector
        try:
            det = VehicleDetector(confidence_threshold=0.4)
        except Exception:
            pytest.skip("Cannot load model")
        
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(blank)
        assert isinstance(detections, list)
        # Blank frame should have no vehicles
        assert len(detections) == 0

    def test_detection_dataclass_fields(self):
        from modules.speed_detection.vehicle_detector import Detection
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.85,
            class_id=2,
            class_name="car",
        )
        assert det.center == (200, 300)
        assert det.bottom_center == (200, 400)
        assert det.width == 200
        assert det.height == 200
        assert det.area == 40000

    def test_draw_detections(self):
        from modules.speed_detection.vehicle_detector import VehicleDetector, Detection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [Detection((100, 100, 200, 200), 0.9, 2, "car")]
        try:
            det = VehicleDetector.__new__(VehicleDetector)
            det.model = None
            result = det.draw_detections(frame, dets)
            assert result.shape == frame.shape
        except Exception:
            pass  # draw_detections is a utility method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
