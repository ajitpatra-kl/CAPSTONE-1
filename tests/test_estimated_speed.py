"""Regressions for opted-in speed estimation and violation continuity."""

import pytest

from modules.speed_detection.speed_estimator import SpeedEstimator
from modules.speed_detection.vehicle_tracker import TrackState
from modules.speed_detection.violation_detector import ViolationDetector


def test_pixel_speed_uses_aligned_history_and_video_timestamps():
    estimator = SpeedEstimator(pixel_scale_mpp=0.1, use_kalman=False)
    track = TrackState(1, 2, "car")
    track.bottom_centers_image.append((900, 900))
    for t in (0.0, 0.04, 0.11, 0.17, 0.22):
        track.timestamps.append(t)
        track.bottom_centers_image.append((100 * t, 30))
    assert estimator.estimate_speed(track, calibrated=False) == pytest.approx(36)
    track.timestamps.append(1.0)
    track.bottom_centers_image.append((150, 30))
    assert estimator.estimate_speed(track, calibrated=False) is None
    track.timestamps.clear()
    track.timestamps.append(1.1)
    track.bottom_centers_image.append((151, 30))
    assert estimator.estimate_speed(track, calibrated=False) is None


@pytest.mark.parametrize("allow_estimated", [False, True])
def test_estimated_overspeed_requires_opt_in_and_does_not_duplicate(allow_estimated):
    detector = ViolationDetector(min_violation_duration_sec=0.3, allow_estimated=allow_estimated)
    track = TrackState(1, 2, "car")
    track.speed_samples_kmh.extend([54] * 3)
    for frame in range(20):
        detector.update(track, 54, frame, frame / 10, is_calibrated=False)
        if frame == 10:
            detector.interrupt_track(track.track_id)
    assert len(detector.violations) == int(allow_estimated)
    if allow_estimated:
        assert detector.violations[0].is_calibrated is False
