"""
tests/test_speed_estimation.py

Unit tests for the SpeedEstimator module.
"""

import sys
import os
import pytest
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.speed_detection.speed_estimator import SpeedEstimator
from modules.speed_detection.vehicle_tracker import TrackState


def make_track_moving_at_speed(
    track_id: int,
    speed_kmh: float,
    n_frames: int = 30,
    fps: float = 25.0,
    start_x: float = 1.0,
    start_y: float = 0.0,
) -> TrackState:
    """
    Create a TrackState simulating a vehicle moving at a fixed speed.
    Vehicle moves along Y axis (depth into scene).
    
    speed_kmh → speed_mps = speed_kmh / 3.6
    distance per frame = speed_mps / fps metres
    """
    state = TrackState(track_id=track_id, class_id=2, class_name="car")
    speed_mps = speed_kmh / 3.6
    dt = 1.0 / fps
    
    for i in range(n_frames):
        t = i * dt
        y = start_y + speed_mps * t
        x = start_x  # lateral position constant
        
        state.bottom_centers_world.append((x, y))
        state.timestamps.append(t)
        # Fake image coords (not used in calibrated mode)
        state.bottom_centers_image.append((320.0, 400.0 - i * 5))
        state.bboxes.append((300, 380 - i*5, 340, 400 - i*5))
        state.hit_count += 1
    
    return state


class TestSpeedEstimatorCalibrated:

    def test_speed_30kmh(self):
        est = SpeedEstimator(smoothing_window=7, min_samples=5)
        state = make_track_moving_at_speed(1, 30.0, n_frames=40)
        speed = est.estimate_speed(state, calibrated=True)
        assert speed is not None, "Speed should not be None with enough samples"
        assert 20 < speed < 45, f"Expected ~30 km/h, got {speed:.1f}"

    def test_speed_50kmh(self):
        est = SpeedEstimator(smoothing_window=7, min_samples=5)
        state = make_track_moving_at_speed(2, 50.0, n_frames=40)
        speed = est.estimate_speed(state, calibrated=True)
        assert speed is not None
        assert 35 < speed < 70, f"Expected ~50 km/h, got {speed:.1f}"

    def test_speed_zero(self):
        est = SpeedEstimator(smoothing_window=5, min_samples=5)
        state = make_track_moving_at_speed(3, 0.0, n_frames=20)
        speed = est.estimate_speed(state, calibrated=True)
        # Should be 0 or None for stationary
        if speed is not None:
            assert speed < 5, f"Stationary vehicle should have ~0 speed, got {speed:.1f}"

    def test_insufficient_samples_returns_none(self):
        est = SpeedEstimator(min_samples=10)
        state = make_track_moving_at_speed(4, 60.0, n_frames=3)
        speed = est.estimate_speed(state, calibrated=True)
        # With only 3 frames, should return None or cached 0
        assert speed is None or speed == 0.0

    def test_implausible_speed_is_rejected(self):
        est = SpeedEstimator(max_speed_kmh=120.0, min_samples=5)
        # A spurious 500 km/h cannot be used for violation confirmation.
        state = make_track_moving_at_speed(5, 500.0, n_frames=20)
        speed = est.estimate_speed(state, calibrated=True)
        assert speed is None

    def test_outlier_rejection(self):
        """Speed estimate should be robust to occasional spike frames."""
        est = SpeedEstimator(smoothing_window=7, min_samples=5, outlier_sigma=2.0)
        state = make_track_moving_at_speed(6, 46.67, n_frames=50)
        
        # Inject a few outlier world points (huge jumps)
        for _ in range(3):
            state.bottom_centers_world.append((999.0, 999.0))
            state.timestamps.append(state.timestamps[-1] + 0.04)
        
        speed = est.estimate_speed(state, calibrated=True)
        # Should still be near 46.67, not 999 km/h
        if speed is not None:
            assert speed < 200.0, f"Outlier not rejected: {speed:.1f} km/h"

    def test_update_all_tracks(self):
        est = SpeedEstimator(smoothing_window=5, min_samples=5)
        tracks = {}
        for i, expected in enumerate([30.0, 50.0, 70.0]):
            state = make_track_moving_at_speed(i, expected, n_frames=30)
            tracks[i] = state
        
        result = est.update_all_tracks(tracks, calibrated=True)
        assert len(result) == 3
        for i, spd in result.items():
            if spd is not None and spd > 0:
                assert 0 <= spd <= 200.0

    def test_synthetic_constant_speed(self):
        """Exercise the estimator on a known synthetic trajectory."""
        est = SpeedEstimator(smoothing_window=7, min_samples=5)
        state = make_track_moving_at_speed(99, 46.67, n_frames=60, fps=25.0)
        speed = est.estimate_speed(state, calibrated=True)
        assert speed is not None, "Should produce a speed estimate"
        assert abs(speed - 46.67) < 15, (
            f"Benchmark estimation too far off: {speed:.1f} km/h vs 46.67 km/h target"
        )

    def test_sigma_clipping(self):
        est = SpeedEstimator(outlier_sigma=1.5)
        # Use a larger array so sigma clipping has statistical power
        # Normal values: 40-43, outliers: 200, 300
        arr = np.array([40, 42, 41, 43, 40, 42, 41, 43, 40, 42, 200, 300], dtype=float)
        clipped = est._sigma_clip(arr)
        # Clipped result should not contain the gross outliers
        # (200 and 300 are >3 sigma from the mean of 40-43)
        assert len(clipped) > 0, "Clipping should not remove everything"
        # The mean of clipped values should be near 41 (not 200 or 300)
        assert np.mean(clipped) < 100, f"Clipping did not remove outliers: mean={np.mean(clipped):.1f}, arr={clipped}"


class TestSpeedEstimatorUncalibrated:

    def test_pixel_only_track_has_no_speed(self):
        est = SpeedEstimator(min_samples=3)
        state = TrackState(track_id=10, class_id=2, class_name="car")
        
        # Add pixel-only data (no world coords)
        fps = 25.0
        for i in range(20):
            t = i / fps
            state.bottom_centers_image.append((300.0, 400.0 - i * 3))
            state.timestamps.append(t)
            state.bboxes.append((280, 385 - i*3, 320, 400 - i*3))
            state.hit_count += 1
        
        assert est.estimate_speed(state, calibrated=False) is None


def test_variable_timestamps_and_duplicate_timestamp():
    estimator = SpeedEstimator(use_kalman=False)
    state = TrackState(1, 2, "car")
    for t in (0.0, 0.05, 0.15, 0.19, 0.26):
        state.timestamps.append(t)
        state.bottom_centers_world.append((10 * t, 0.0))
    assert estimator.estimate_speed(state) == pytest.approx(36.0)
    state.timestamps.append(0.26)
    state.bottom_centers_world.append((2.6, 0.0))
    assert estimator.estimate_speed(state) is None


def test_smoothing_rejects_a_box_jump():
    estimator = SpeedEstimator(use_kalman=False)
    state = TrackState(1, 2, "car")
    for i in range(7):
        t = i / 20
        state.timestamps.append(t)
        state.bottom_centers_world.append((10 * t, 30.0 if i == 3 else 0.0))
    assert estimator.estimate_speed(state) == pytest.approx(36.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
