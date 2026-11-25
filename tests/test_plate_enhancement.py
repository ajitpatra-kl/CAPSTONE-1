from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from modules.anpr.ocr_session import OCRSession
from modules.anpr.plate_detector import PlateDetection
from modules.anpr.plate_enhancement import enhance_plate
from modules.anpr.plate_recognizer import OCRReading
from modules.anpr.plate_scan import PlateScan


def test_ocr_receives_original_and_enhanced_variant(monkeypatch, tmp_path):
    engine = Mock()
    engine.read.return_value = {"original": [OCRReading("ABC1234", .9)]}
    monkeypatch.setattr("modules.anpr.ocr_session.PPOCRv4", lambda _: engine)
    session = OCRSession({"ocr_engine": "ppocrv4"}, tmp_path)
    original = np.arange(12 * 40 * 3, dtype=np.uint8).reshape(12, 40, 3)
    native_pixels = original.copy()
    variants, _ = session.scan(original)
    inputs = engine.read.call_args.args[0]
    assert inputs["original"] is original
    np.testing.assert_array_equal(inputs["enhanced"], variants["enhanced"])
    assert variants["enhanced"].shape == (48, 160)
    np.testing.assert_array_equal(original, native_pixels)


def test_export_saves_enhanced_copy_even_when_ocr_was_not_attempted(tmp_path):
    detector = Mock()
    detector.detect.return_value = [PlateDetection((12, 33, 45, 43), 0.8)]
    recognizer = Mock(initialization_errors={})
    scan = PlateScan(detector, recognizer, {"sample_interval_frames": 1, "crop_padding": 0})
    frame = np.full((55, 60, 3), 100, np.uint8)
    scan.observe(frame, 21, {7: SimpleNamespace(latest_bbox=(0, 0, 60, 55))})
    _, rows = scan.export(tmp_path)
    row, = rows
    assert row["status"] == "Plate Detected" and row["ocr_inputs"] == {}
    assert "track7_frame21" in row["enhanced_crop_path"]
    original = frame[33:43, 12:45]
    np.testing.assert_array_equal(cv2.imread(row["crop_path"]), original)
    np.testing.assert_array_equal(
        cv2.imread(row["enhanced_crop_path"], cv2.IMREAD_GRAYSCALE).squeeze(), enhance_plate(original),
    )
    recognizer.scan.assert_not_called()
