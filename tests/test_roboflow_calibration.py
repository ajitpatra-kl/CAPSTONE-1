import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.speed_detection.camera_calibration import CalibrationManager
from modules.speed_detection.speed_estimator import SpeedEstimator


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "Roboflow_Vehicles"
SOURCE = np.float32([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET = np.float32([[0, 0], [24, 0], [24, 249], [0, 249]])


@pytest.fixture
def manager():
    config = json.loads((ROOT / "config/dataset_config.json").read_text())
    return CalibrationManager(config)


def test_reference_projection_and_coverage(manager):
    calibration = manager.get_calibration(PROFILE)
    assert calibration is not None
    np.testing.assert_allclose(calibration.transform_many(SOURCE), TARGET, atol=1e-4)
    anchors = np.float32([[1800, 1000], [2100, 1800], [1750, 1400]])
    expected = cv2.perspectiveTransform(
        anchors.reshape(-1, 1, 2), cv2.getPerspectiveTransform(SOURCE, TARGET)
    ).reshape(-1, 2)
    np.testing.assert_allclose(calibration.transform_many(anchors), expected, atol=1e-4)
    assert calibration.covers_anchor((1800, 1000))
    assert calibration.covers_anchor((1252, 787))
    assert not calibration.covers_anchor((1800, 700))
    assert manager.get_calibration("T1P1_Kasa") is None


def test_similar_name_and_native_dimensions_do_not_select_profile(manager, tmp_path):
    unrelated = tmp_path / "vehicles.mp4"
    unrelated.write_bytes(b"unrelated video with the same filename")
    assert manager.detect_profile(unrelated, (3840, 2160)) is None


def test_reference_video_identity(manager):
    video = ROOT / "dataset/roboflow/vehicles.mp4"
    if not video.is_file():
        pytest.skip("Optional Roboflow demo video is not installed")
    assert manager.detect_profile(video, (3840, 2160)) == PROFILE
    assert manager.detect_profile(video, (1920, 1080)) is None


def test_reference_projection_speed_uses_elapsed_timestamps(manager):
    calibration = manager.get_calibration(PROFILE)
    timestamps = np.array([0.0, 0.04, 0.08, 0.16, 0.20])
    expected_world = np.column_stack((np.full(5, 12), 100 + 20 * timestamps))
    anchors = cv2.perspectiveTransform(
        expected_world.reshape(-1, 1, 2), np.linalg.inv(calibration.H)
    ).reshape(-1, 2)
    projected = calibration.transform_many(anchors)
    estimator = SpeedEstimator()
    assert estimator._estimate_displacement(timestamps, projected) == pytest.approx(72, abs=0.001)
