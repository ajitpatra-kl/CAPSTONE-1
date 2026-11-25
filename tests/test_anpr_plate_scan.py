from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from modules.anpr.plate_detector import PlateDetection
from modules.anpr.ocr_session import EngineResult
from modules.anpr.plate_recognizer import OCRReading
from modules.anpr.plate_scan import PlateScan, draw_plates
from modules.anpr.plate_validator import validate_registration


def test_native_mapping_and_exact_frame_deduplication():
    detector = Mock()
    detector.detect.return_value = [PlateDetection((8, 35, 32, 42), 0.8)]
    scan = PlateScan(detector, Mock(), {"sample_interval_frames": 1, "crop_padding": 0.1})
    frame = np.full((100, 160, 3), 150, np.uint8)
    tracks = {7: SimpleNamespace(latest_bbox=(40, 20, 100, 65))}
    scan.observe(frame, 0, tracks)
    scan.observe(frame, 1, tracks)
    assert detector.detect.call_count == 1
    assert scan.records[0].bbox == (42, 50, 66, 57)
    assert scan.records[0].crop.shape == (7, 24, 3)
    canvas = frame.copy()
    draw_plates(canvas, scan.by_frame[0])
    assert tuple(canvas[56, 42]) == (0, 255, 0)


def test_ocr_consensus_counts_distinct_frames_and_retains_raw_text(tmp_path):
    detector = Mock()
    detector.detect.return_value = [PlateDetection((12, 33, 45, 43), 0.8)]
    recognizer = Mock()
    variant = np.full((48, 160), 127, np.uint8)
    recognizer.initialization_errors = {}
    recognizer.scan.return_value = ({"gray": variant}, {"easyocr": EngineResult({
        "gray": [OCRReading("22 BH 1234 A", 0.71)],
        "contrast": [OCRReading("22 BH 1234 A", 0.72)],
    }, 10.0)})
    scan = PlateScan(detector, recognizer, {"sample_interval_frames": 1, "crop_padding": 0})
    track = {7: SimpleNamespace(latest_bbox=(0, 0, 60, 55))}
    first = np.full((55, 60, 3), 100, np.uint8)
    scan.observe(first, 0, track)
    scan.observe(first, 1, track)
    scan.observe(np.full_like(first, 110), 2, track)
    result = scan.result(7)
    assert result.status == "Consistent Candidate — Unverified"
    assert result.candidate == result.candidate_format == "22BH1234A"
    assert result.vehicle_number == ""
    assert result.candidate_frames == (0, 2)
    assert recognizer.scan.call_count == 2
    assert scan.result(7) is result
    _, rows = scan.export(tmp_path)
    assert len(rows) == 2 and rows[0]["raw_text"] == "22 BH 1234 A"
    saved = cv2.imread(rows[0]["crop_path"])
    np.testing.assert_array_equal(saved, first[33:43, 12:45])
    assert validate_registration("22 BH 12O4 A") == ""
