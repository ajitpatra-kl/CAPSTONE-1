from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from modules.anpr import ANPRProcessor
from modules.anpr.plate_detector import PlateDetection, PlateDetector
from modules.anpr.plate_recognizer import OCRReading, PlateRecognizer
from modules.anpr.plate_validator import validate_registration
from modules.speed_detection.evidence_generator import EvidenceGenerator
from modules.speed_detection.report_generator import ReportGenerator
from modules.speed_detection.vehicle_detector import Detection
from modules.speed_detection.video_processor import VideoProcessor
from modules.speed_detection.violation_detector import ViolationRecord


@pytest.mark.parametrize(("text", "expected"), [
    ("AP 39 AB 1234", "AP39AB1234"), ("od-02.c-0042", "OD02C0042"),
    ("TS09EA1234", "TS09EA1234"), ("KA01AA1234", "KA01AA1234"),
    ("MH12XY1234", "MH12XY1234"), ("DL8CAF1234", "DL8CAF1234"),
    ("OD02AB12O4", ""), ("ZZ02AB1234", ""), ("OD02AB123", ""),
])
def test_standard_registration(text, expected):
    assert validate_registration(text) == expected


def test_detector_loads_only_plate_weights_and_clips_boxes(tmp_path, monkeypatch):
    weights = tmp_path / "plate.pt"
    weights.touch()
    boxes = SimpleNamespace(xyxy=np.array([[-2.4, 1.5, 48.1, 30], [50, 0, 80, 30]]),
                            conf=np.array([0.8, 0.6]))
    model = SimpleNamespace(names={0: "plate"}, predict=Mock(return_value=[SimpleNamespace(boxes=boxes)]))
    loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=loader))
    detector = PlateDetector(weights, 0.35, 2)
    assert detector.detect(np.zeros((20, 50, 3), np.uint8)) == [PlateDetection((0, 1, 49, 20), 0.8)]
    loader.assert_called_once_with(str(weights), task="detect")
    assert model.predict.call_args.kwargs["conf"] == 0.35
    with pytest.raises(FileNotFoundError):
        PlateDetector(tmp_path / "missing.pt", 0.35, 2)


def test_recognizer_preserves_raw_text_and_uses_no_ocr_detector(tmp_path, monkeypatch):
    (tmp_path / "english_g2.pth").touch()
    recognize = Mock(return_value=[(None, "OD 02 AB 1234", 0.91)])
    factory = Mock(return_value=SimpleNamespace(recognize=recognize))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=factory))
    recognizer = PlateRecognizer(tmp_path)
    assert recognizer.recognize(np.zeros((12, 55, 3), np.uint8)) == [OCRReading("OD 02 AB 1234", 0.91)] * 2
    assert not factory.call_args.kwargs["detector"]
    assert all(call.args[0].shape == (48, 220) for call in recognize.call_args_list)


@pytest.fixture
def processor(monkeypatch, tmp_path):
    detector = Mock()
    detector.detect.return_value = [PlateDetection((4, 4, 44, 19), 0.9)]
    recognizer = Mock()
    monkeypatch.setattr("modules.anpr.anpr_processor.PlateDetector", Mock(return_value=detector))
    monkeypatch.setattr("modules.anpr.anpr_processor.PlateRecognizer", Mock(return_value=recognizer))
    return ANPRProcessor({"sample_interval_frames": 1, "crop_padding": 0}, tmp_path)


def test_consensus_and_track_isolation(processor):
    tracks = {45: SimpleNamespace(latest_bbox=(0, 0, 55, 40)),
              46: SimpleNamespace(latest_bbox=(55, 0, 110, 40))}
    for marker in (10, 20, 30):
        frame = np.full((40, 110, 3), marker, np.uint8)
        frame[:, 55:] = 100
        processor.observe(frame, marker, tracks)
        frame[:] = 0  # Buffered observations own their pixels.

    def recognize(plate):
        marker = plate[0, 0, 0]
        return {10: [OCRReading("OD02AB1234", 0.65)],
                20: [OCRReading("OD02AB1234", 0.70)],
                30: [OCRReading("OD02AB1284", 0.92)],
                100: [OCRReading("KA01AB5678", 0.90)]}[marker]

    processor.recognizer.recognize.side_effect = recognize
    assert processor.recognize_track(45).vehicle_number == "OD02AB1234"
    assert processor.recognize_track(46).vehicle_number == "KA01AB5678"
    assert processor.recognize_track(45).vehicle_number == "OD02AB1234"
    assert processor.detector.detect.call_count == 6


def test_padding_does_not_vote_for_neighbor_and_attempts_are_capped(processor):
    processor.padding = 0.5
    processor.max_attempts = 1
    processor.detector.detect.return_value = [PlateDetection((1, 20, 17, 35), 0.95)]
    active = {45: SimpleNamespace(latest_bbox=(25, 25, 75, 75))}
    frame = np.zeros((100, 100, 3), np.uint8)
    processor.observe(frame, 0, active)
    processor.observe(frame, 1, active)
    assert processor.recognize_track(45).status == "Not Detected"
    processor.observe(frame, 2, active)
    assert processor.recognize_track(45).status == "Not Detected"
    processor.detector.detect.assert_called_once()
    processor.recognizer.recognize.assert_not_called()


@pytest.mark.parametrize(("plates", "reading", "status"), [
    ([], OCRReading("", 0), "Not Detected"),
    ([PlateDetection((0, 0, 10, 5), 0.9)], OCRReading("", 0), "Not Readable"),
    ([PlateDetection((0, 0, 40, 15), 0.9)], OCRReading("OD02AB1234", 0.2), "Low Confidence"),
    ([PlateDetection((0, 0, 40, 15), 0.9)], OCRReading("OD02AB1234", 0.6), "Low Confidence"),
])
def test_unreadable_results_are_empty(processor, plates, reading, status):
    processor.detector.detect.return_value = plates
    processor.recognizer.recognize.return_value = [reading]
    processor.observe(np.zeros((50, 50, 3), np.uint8), 0,
                      {45: SimpleNamespace(latest_bbox=(-5, -4, 50, 50))})
    result = processor.recognize_track(45)
    assert result.status == status
    assert result.vehicle_number == ""
    assert result.vehicle_crop.shape == (50, 50, 3)
    processor.recognize_track(45)
    processor.detector.detect.assert_called_once()


def test_violation_serialization_reports_and_crops(tmp_path, processor):
    frame = np.zeros((120, 160, 3), np.uint8)
    processor.observe(frame, 0, {45: SimpleNamespace(latest_bbox=(0, 0, 55, 50))})
    processor.recognizer.recognize.return_value = [OCRReading("OD 02 AB 1234", 0.91)]
    result = processor.recognize_track(45)
    recognized = ViolationRecord(45, 2, "car", 60, 40, 20, 10, 300, vehicle_number=result.vehicle_number,
                                 plate_confidence=result.plate_confidence, anpr_status=result.status)
    legacy = ViolationRecord(46, 2, "car", 58, 40, 18, 11, 330)
    evidence = EvidenceGenerator(tmp_path)
    recognized.evidence_path = evidence.generate(frame, recognized)
    evidence.save_anpr_crops(recognized, result)
    row = recognized.to_dict()
    assert Path(row["vehicle_crop_path"]).is_file()
    assert Path(row["plate_crop_path"]).is_file()
    assert row["vehicle_number"] == "OD02AB1234"
    assert legacy.to_dict()["vehicle_number"] == ""
    reporter = ReportGenerator(tmp_path)
    report = pd.read_csv(reporter.generate_violations_report([recognized, legacy], "video", 40, {}),
                         comment="#", keep_default_na=False)
    assert report["Track ID"].tolist() == [45, 46]
    assert report["Registration Number"].tolist() == ["OD02AB1234", ""]
    assert report["ANPR Status"].tolist() == ["Recognized", ""]
    assert report["Plate Confidence"].tolist() == ["0.91", ""]
    empty = pd.read_csv(reporter.generate_violations_report([], "video", 40, {}), comment="#")
    assert list(empty.columns) == list(report.columns)


@pytest.mark.parametrize("fail", [False, True])
def test_video_pipeline_associates_anpr_and_survives_failure(tmp_path, monkeypatch, fail):
    input_video = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 160))
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.zeros((160, 320, 3), dtype=np.uint8))
    writer.release()

    class MovingVehicleDetector:
        def __init__(self, **kwargs):
            self.x = 15

        def detect(self, frame):
            self.x += 5
            return [Detection((self.x, 50, self.x + 50, 100), 0.95, 2, "car")]

    engine = Mock()
    if fail:
        engine.recognize_track.side_effect = RuntimeError("test OCR failure")
    else:
        from modules.anpr.anpr_processor import ANPRResult

        engine.recognize_track.return_value = ANPRResult("Recognized", "OD02AB1234", 0.92)
    monkeypatch.setattr("modules.anpr.ANPRProcessor", Mock(return_value=engine))
    monkeypatch.setattr("modules.speed_detection.vehicle_detector.VehicleDetector", MovingVehicleDetector)
    result = VideoProcessor({
        "output_dir": tmp_path / "outputs",
        "anpr": {"enabled": True},
        "speed_estimation": {"pixel_scale_mpp": 0.15},
    }, {}).process(str(input_video))
    assert result["analytics"]["processed_frames"] == 40
    violation, = result["violations"]
    engine.recognize_track.assert_called_once_with(violation["track_id"])
    assert violation["speed_kmh"] == pytest.approx(54)
    assert violation["anpr_status"] == ("Processing Failed" if fail else "Recognized")
    assert violation["vehicle_number"] == ("" if fail else "OD02AB1234")
    assert Path(violation["evidence_path"]).is_file()
