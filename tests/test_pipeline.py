"""Pipeline regressions using a controlled trajectory and temporary outputs."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from modules.speed_detection import VideoProcessor
from modules.speed_detection.vehicle_detector import Detection


@pytest.mark.parametrize("mode", ["verified", "assumed road", "pixel scale"])
def test_calibrated_pipeline_confirms_once_and_preserves_original_frame(tmp_path, monkeypatch, mode):
    input_video = tmp_path / "test.mp4"
    size = (320, 160)
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"mp4v"), 20, size)
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()

    class MovingDetection:
        def __init__(self, **kwargs):
            self.x = 15

        def detect(self, frame):
            self.x += 5
            return [Detection(bbox=(self.x, 50, self.x + 50, 100),
                              confidence=0.95, class_id=2, class_name="car")]

    monkeypatch.setattr("modules.speed_detection.vehicle_detector.VehicleDetector", MovingDetection)
    profile = {
        "camera": "Test camera",
        "resolution": size,
        "sha256": hashlib.sha256(input_video.read_bytes()).hexdigest(),
        "calibration_status": "verified",
        "calibration": {
            "source": "Synthetic test geometry",
            "image_points": [[0, 0], [300, 0], [300, 150], [0, 150]],
            "world_points_m": [[0, 0], [45, 0], [45, 22.5], [0, 22.5]],
            "check_image_points": [[100, 100]],
            "check_world_points_m": [[15, 15]],
            "check_tolerance_m": 0.01,
        },
    }
    config = {"forespeed": {"videos": {"test": profile}}}
    options = {"output_dir": tmp_path / "outputs"}
    if mode != "verified":
        config = {}
        options["speed_estimation"] = (
            {"assumed_road": {
                "image_points": profile["calibration"]["image_points"],
                "width_m": 45,
                "length_m": 22.5,
            }} if mode == "assumed road" else {"pixel_scale_mpp": 0.15}
        )
    overlay_labels = []
    put_text = cv2.putText

    def record_overlay_text(image, text, *args, **kwargs):
        if text.startswith("Est. "):
            overlay_labels.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_overlay_text)
    result = VideoProcessor(options, config).process(str(input_video))
    if mode == "assumed road":
        assert result["analytics"]["avg_speed_kmh"] == pytest.approx(54)
        assert any(label.startswith("Est. 54.0 km/h") for label in overlay_labels)

    assert result["calibrated"] is (mode == "verified")
    assert result["estimated_speed_mode"] == {
        "verified": None,
        "assumed road": "assumed homography",
        "pixel scale": "assumed pixel scale",
    }[mode]
    assert result["analytics"]["tracker"] == "ByteTrack"
    assert result["analytics"]["unique_tracks"] == 1
    assert result["analytics"]["processed_frames"] == 40
    assert result["analytics"]["invalid_timestamps"] == 0
    violation, = result["violations"]
    assert violation["is_calibrated"] is (mode == "verified")
    assert violation["speed_kmh"] == pytest.approx(54.0)
    evidence = Path(violation["evidence_path"])
    original = Path(violation["original_frame_path"])
    assert cv2.imdecode(np.frombuffer(evidence.read_bytes(), np.uint8), cv2.IMREAD_COLOR).max() > 0
    assert cv2.imdecode(np.frombuffer(original.read_bytes(), np.uint8), cv2.IMREAD_COLOR).max() == 0
    report = pd.read_csv(result["csv_report_path"], comment="#")
    assert len(report) == 1
    assert report.iloc[0]["Calibrated"] == ("Yes" if mode == "verified" else "No")
    assert len(pd.read_csv(result["tracks_csv_path"])) == 1
    annotated_video = cv2.VideoCapture(result["output_video_path"])
    try:
        assert annotated_video.read()[0]
        assert int(annotated_video.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    finally:
        annotated_video.release()
