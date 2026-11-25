"""
tests/test_calibration.py

Unit tests for camera calibration module.
Tests homography computation, coordinate transformation, and profile detection.
"""

import sys
import os
import json
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.speed_detection.camera_calibration import (
    HomographyCalibration,
    CalibrationManager,
)


# ─────────────────────────── Fixtures ────────────────────────────────
@pytest.fixture
def simple_calibration():
    """A well-conditioned calibration with known points."""
    # Image rectangle: (100,300), (500,300), (500,100), (100,100)
    image_pts = np.array([
        [100, 300],
        [500, 300],
        [500, 100],
        [100, 100],
    ], dtype=np.float32)
    # World rectangle: 0–4m wide, 0–8m deep
    world_pts = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 8.0],
        [0.0, 8.0],
    ], dtype=np.float32)
    return HomographyCalibration(image_pts, world_pts, profile_name="test")


@pytest.fixture
def dataset_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "dataset_config.json")
    with open(cfg_path) as f:
        return json.load(f)


# ─────────────────────────── Tests ───────────────────────────────────
class TestHomographyCalibration:

    def test_init_success(self, simple_calibration):
        cal = simple_calibration
        assert cal.H is not None
        assert cal.H.shape == (3, 3)

    def test_is_valid(self, simple_calibration):
        assert simple_calibration.is_valid()

    def test_transform_corner_bl(self, simple_calibration):
        """Bottom-left image corner should map to world (0,0)."""
        X, Y = simple_calibration.transform(100, 300)
        assert abs(X - 0.0) < 0.5, f"X={X:.4f}, expected ~0"
        assert abs(Y - 0.0) < 0.5, f"Y={Y:.4f}, expected ~0"

    def test_transform_corner_br(self, simple_calibration):
        """Bottom-right image corner should map to world (4,0)."""
        X, Y = simple_calibration.transform(500, 300)
        assert abs(X - 4.0) < 0.5, f"X={X:.4f}, expected ~4"
        assert abs(Y - 0.0) < 0.5, f"Y={Y:.4f}, expected ~0"

    def test_transform_corner_tl(self, simple_calibration):
        """Top-left image corner should map to world (0,8)."""
        X, Y = simple_calibration.transform(100, 100)
        assert abs(X - 0.0) < 0.5
        assert abs(Y - 8.0) < 0.5

    def test_transform_many(self, simple_calibration):
        pts = np.array([[100, 300], [500, 300], [100, 100]], dtype=np.float32)
        world = simple_calibration.transform_many(pts)
        assert world.shape == (3, 2)
        assert abs(world[0, 0] - 0.0) < 0.5
        assert abs(world[1, 0] - 4.0) < 0.5

    def test_reprojection_error_low(self, simple_calibration):
        """Reprojection error should be very low for exact correspondences."""
        assert simple_calibration.reprojection_error_m < 1.0

    def test_requires_4_points(self):
        with pytest.raises((ValueError, Exception)):
            HomographyCalibration(
                np.array([[0,0],[1,0],[0,1]], dtype=np.float32),
                np.array([[0,0],[1,0],[0,1]], dtype=np.float32),
            )

    def test_summary_contains_profile_name(self, simple_calibration):
        summary = simple_calibration.summary()
        assert "test" in summary


class TestCalibrationManager:

    def test_profile_requires_checksum_and_resolution(self, dataset_config, tmp_path):
        import hashlib

        profile = dataset_config["forespeed"]["videos"]["T1P1_Eufy"]
        video = tmp_path / "renamed.mp4"
        video.write_bytes(b"first camera fixture")
        profile["sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
        manager = CalibrationManager(dataset_config)
        assert manager.detect_profile(str(video), (2304, 1296)) == "T1P1_Eufy"
        assert manager.detect_profile(str(video), (1920, 1080)) is None
        impostor = tmp_path / profile["path"].split("/")[-1]
        impostor.write_bytes(b"second camera fixture")
        assert manager.detect_profile(str(impostor), (2304, 1296)) is None

    def test_legacy_and_unverified_profiles_do_not_calibrate(self, dataset_config):
        manager = CalibrationManager(dataset_config)
        for key in ("forespeed_T1", "forespeed_T2", "T1P1_Eufy", "T1P1_Kasa"):
            assert manager.get_calibration(key) is None

    def test_independent_reference_check(self, dataset_config, simple_calibration):
        profile = dataset_config["forespeed"]["videos"]["T1P1_Eufy"]
        profile["calibration_status"] = "verified"
        profile["calibration"] = {
            "source": "Synthetic independent geometry",
            "image_points": simple_calibration.image_points.tolist(),
            "world_points_m": simple_calibration.world_points.tolist(),
            "check_image_points": [[300, 200]],
            "check_world_points_m": [[2, 4]],
            "check_tolerance_m": 0.05,
        }
        manager = CalibrationManager(dataset_config)
        assert manager.get_calibration("T1P1_Eufy") is not None
        profile["calibration"]["check_world_points_m"] = [[2, 5]]
        assert manager.get_calibration("T1P1_Eufy") is None

    def test_get_calibration_unknown(self, dataset_config):
        mgr = CalibrationManager(dataset_config)
        cal = mgr.get_calibration("nonexistent_profile")
        assert cal is None

    def test_custom_calibration(self, dataset_config):
        mgr = CalibrationManager(dataset_config)
        img_pts = np.array([[0,0],[400,0],[400,300],[0,300]], dtype=np.float32)
        world_pts = np.array([[0,0],[5,0],[5,10],[0,10]], dtype=np.float32)
        cal = mgr.get_calibration_from_custom_points(img_pts, world_pts, "my_cam")
        assert cal is not None
        assert cal.is_valid()

    def test_distance_preservation(self, simple_calibration):
        bottom = np.asarray(simple_calibration.transform(100, 300))
        top = np.asarray(simple_calibration.transform(100, 100))
        assert np.linalg.norm(top - bottom) == pytest.approx(8.0, abs=1e-5)


class TestSpeedFromCalibration:
    def test_speed_calculation(self, simple_calibration):
        from modules.speed_detection import SpeedEstimator
        from modules.speed_detection.vehicle_tracker import TrackState

        state = TrackState(1, 2, "car")
        for i in range(5):
            state.timestamps.append(i * 0.25)
            state.bottom_centers_world.append(simple_calibration.transform(100, 300 - 50 * i))
        assert SpeedEstimator().estimate_speed(state) == pytest.approx(28.8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
