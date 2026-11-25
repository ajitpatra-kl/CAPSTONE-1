"""Checks on the original ForeSpeed media, independent of synthetic tests."""

import json
from pathlib import Path

import cv2
import pytest

from modules.speed_detection.camera_calibration import CalibrationManager


@pytest.mark.parametrize(("key", "frame_count"), [("T1P1_Eufy", 988), ("T1P1_Kasa", 900)])
def test_original_variable_rate_video(key, frame_count):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "config/dataset_config.json").read_text(encoding="utf-8"))
    manager = CalibrationManager(config)
    profile = manager.profiles[key]
    video = root / "dataset/forespeed" / profile["path"]
    if not video.is_file():
        pytest.skip("Local ForeSpeed recording unavailable")
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    stamps = []
    try:
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        assert manager.detect_profile(str(video), resolution) == key
        while cap.grab():
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    assert len(stamps) == frame_count
    intervals = [round(end - start, 3) for start, end in zip(stamps, stamps[1:])]
    assert min(intervals) > 0
    assert len(set(intervals)) > 1
    assert manager.get_calibration(key) is None
