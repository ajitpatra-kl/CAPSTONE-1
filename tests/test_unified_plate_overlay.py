"""Unified plate geometry, recognition and overlay regressions."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from modules.anpr.ocr_session import EngineResult
from modules.anpr.plate_detector import PlateDetection
from modules.anpr.plate_recognizer import OCRReading
from modules.anpr.plate_scan import PlateScan
from modules.unified import pipeline


@pytest.mark.parametrize("error", ["", "OCR unavailable"])
def test_event_scan_translates_boxes_and_keeps_detection_when_ocr_fails(error):
    frame = np.zeros((150, 300, 3), np.uint8)
    frame[:, 150:] = 50
    tracks = {7: SimpleNamespace(latest_bbox=(30, 20, 130, 120)),
              8: SimpleNamespace(latest_bbox=(180, 20, 280, 120))}
    detector = Mock()
    detector.detect.side_effect = lambda crop: (
        [PlateDetection((20, 95, 60, 108), .9)] if crop[0, 0, 0] == 0 else []
    )
    recognizer = Mock()
    recognizer.scan.return_value = ({}, {
        "easyocr": EngineResult({} if error else {"original": [OCRReading("OD02AB1234", .95)]}, 1, error)
    })
    scan = PlateScan(detector, recognizer, {"event_scan": True, "crop_padding": .1, "sample_interval_frames": 3})
    scan.observe(frame, 0, tracks)
    scan.observe(frame, 1, tracks)
    scan.sample_track(frame, 0, 7, {tid: track.latest_bbox for tid, track in tracks.items()}, force=True)
    assert detector.detect.call_count == 2
    recognizer.scan.assert_not_called()
    plate, = scan.by_frame[0]
    assert plate.track_id == 7 and plate.bbox == (40, 105, 80, 118)
    result = scan.result(7)
    assert result.plate_bbox == plate.bbox
    assert result.detection_confidence == .9
    assert result.frame_number == 0
    assert result.vehicle_number == ("" if error else "OD02AB1234")
    assert result.status == ("Processing Failed" if error else "Recognized")
    assert scan.result(7) is result
    assert scan.result(8).plate_bbox is None
    recognizer.scan.assert_called_once()


def test_overlays_use_colors_labels_and_exact_frame(monkeypatch, tmp_path):
    canvas = np.zeros((120, 320, 3), np.uint8)
    label = Mock()
    monkeypatch.setattr(pipeline, "draw_label", label)
    pipeline.draw_track(canvas, 1, (10, 10, 100, 95), "car")
    pipeline.draw_track(canvas, 2, (130, 10, 220, 95), "OVERSPEED", pipeline.VIOLATION_COLOR)
    assert tuple(canvas[40, 10]) == pipeline.NORMAL_COLOR
    assert tuple(canvas[40, 130]) == pipeline.VIOLATION_COLOR

    collector = pipeline.EventCollector("session", "video", tmp_path, None, {}, tmp_path)
    plate = SimpleNamespace(track_id=2, bbox=(155, 74, 190, 86), rank=1)
    collector.anpr = SimpleNamespace(scan=SimpleNamespace(by_frame={5: [plate]}))
    collector.draw_track_plates(canvas, 4, 2, (130, 10, 220, 95))
    collector.draw_track_plates(canvas, 5, 1, (10, 10, 100, 95))
    assert tuple(canvas[74, 155]) == (0, 0, 0)
    collector.draw_track_plates(canvas, 5, 2, (130, 10, 220, 95))
    assert tuple(canvas[74, 155]) == pipeline.PLATE_COLOR
    assert label.call_args.args[1] == "PLATE DETECTED"
    collector.active_violations[2] = {"overspeed": 5}
    collector.draw_track_plates(canvas, 5, 2, (130, 10, 220, 95))
    assert label.call_args.args[1] == "PLATE DETECTED - UNREADABLE"
    collector.confirmed_numbers[2] = "OD02AB1234"
    collector.draw_track_plates(canvas, 5, 2, (130, 10, 220, 95))
    assert label.call_args.args[1] == "OD02AB1234"


def test_checksum_selects_roboflow_plate_profile():
    settings = {"anpr": {"enabled": True, "roboflow_scan": {"detector_image_size": 960, "scan_all_tracks": True}}}
    datasets = {"roboflow": {"videos": {"Roboflow_Vehicles": {"sha256": "reference"}}}}
    options = pipeline.anpr_profile(settings, datasets, "reference")
    assert options["event_scan"] and not options["scan_all_tracks"]
    assert options["detector_image_size"] == 960
    assert "detector_image_size" not in pipeline.anpr_profile(settings, datasets, "another video")
