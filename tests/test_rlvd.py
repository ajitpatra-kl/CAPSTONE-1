"""Regression coverage for RLVD timing, geometry and artifact generation."""

import csv

import cv2
import numpy as np
import pytest

from modules.rlvd.configuration import Configuration, save_configuration, video_metadata
from modules.rlvd.signal_detector import SignalHistory, classify_signal
from modules.rlvd.video_processor import process_video
from modules.rlvd.violation_engine import CrossingEngine, Thresholds
from modules.speed_detection.vehicle_detector import Detection


POINTS = [95, 93, 91, 89, 87, 85, 83, 81, 78, 75, 72, 69, 66]
BOX_COLOR = (240, 240, 240)
SIGNAL_COLORS = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "YELLOW": (0, 255, 255)}


def config():
    return Configuration("test", "synthetic", 180, 120, 10, [10, 10, 27, 48],
                         [[55, 80], [125, 80]], "Bottom")


@pytest.mark.parametrize("states,points,x,omissions,expected", [
    (["GREEN"] * 13, POINTS, 95, (), "Legal crossing"),
    (["YELLOW"] * 13, POINTS, 95, (), "Legal crossing"),
    (["GREEN"] * 9 + ["RED"] * 4, POINTS, 95, (), "Legal crossing"),
    (["GREEN"] * 6 + ["RED"] * 7, [65] * 13, 95, (), None),
    (["GREEN"] * 4 + ["RED"] * 9, [95, 93, 91, 89] + [82] * 9, 95, (), None),
    (["GREEN"] * 4 + ["RED"] * 9, POINTS, 95, (), "Violation"),
    (["GREEN"] * 4 + ["RED"] * 3 + ["UNKNOWN"] + ["RED"] * 5, POINTS, 95, (), "Uncertain crossing"),
    (["GREEN"] * 4 + ["RED"] * 9, POINTS, 95, (7,), "Uncertain crossing"),
    (["GREEN"] * 4 + ["RED"] * 9, POINTS, 145, (), None),
    (["GREEN"] * 4 + ["RED"] * 9, POINTS[::-1], 95, (), None),
    (["GREEN"] * 7 + ["RED"] * 6, POINTS, 95, (), "Uncertain crossing"),
    (["RED"] * 13, POINTS, 95, (), "Uncertain crossing"),
])
def test_crossing_timing(states, points, x, omissions, expected):
    settings = config()
    history = SignalHistory(3)
    engine = CrossingEngine(settings, Thresholds(deadband=2), history)
    events = []
    for frame, (color, y) in enumerate(zip(states, points)):
        history.update(color, 1)
        if frame not in omissions:
            event = engine.observe(1, frame, (x-10, y-20, x+10, y), "car", .95)
            if event:
                events.append(event)
    assert [event.status for event in events] == ([expected] if expected else [])
    if expected == "Violation":
        assert events[0].onset == 4
        assert events[0].frame == pytest.approx(7 + 1/3)


def test_lamp_detection_rejects_ambiguous_roi():
    config_data = config()
    frame = np.zeros((120, 180, 3), np.uint8)
    for (state, color), y in zip(SIGNAL_COLORS.items(), (17, 41, 29)):
        frame[:] = 0
        cv2.circle(frame, (18, y), 4, color, -1)
        assert classify_signal(frame, config_data.roi)[0] == state
    cv2.circle(frame, (18, 17), 4, SIGNAL_COLORS["RED"], -1)
    assert classify_signal(frame, config_data.roi)[0] == "UNKNOWN"
    frame[:] = (0, 0, 255)
    assert classify_signal(frame, config_data.roi)[0] == "UNKNOWN"


def test_video_outputs_with_real_bytetrack(tmp_path):
    from dataclasses import asdict
    import json
    from pathlib import Path

    settings = config()
    save_configuration(tmp_path, asdict(settings), "test")
    assert Configuration(**json.loads((tmp_path / "test.json").read_text())) == settings
    source = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (180, 120))
    assert writer.isOpened()
    for index, y in enumerate(POINTS):
        frame = np.zeros((120, 180, 3), np.uint8)
        cv2.circle(frame, (18, 41 if index < 4 else 17), 4,
                   SIGNAL_COLORS["GREEN" if index < 4 else "RED"], -1)
        cv2.rectangle(frame, (83, y-19), (106, y-1), BOX_COLOR, -1)
        writer.write(frame)
    writer.release()

    class FixtureDetector:
        def detect(self, frame):
            # The rectangle is the only bright object in the lane.
            _, top, _, height = cv2.boundingRect(cv2.inRange(frame[:, 55:126], (200,)*3, (255,)*3))
            return [Detection((83, top, 107, top+height), .95, 2, "car")]

    result = process_video(source, settings, Thresholds(deadband=2), tmp_path, detector=FixtureDetector())
    with result["report"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert (tmp_path / "outputs" / "rlvd" / "rlvd_report.csv").read_bytes() == result["report"].read_bytes()
    assert len(rows) == 1
    assert rows[0]["plate_number"] == "Not Readable"
    assert float(rows[0]["red_onset_timestamp"]) == .4
    for key in ("original_image_path", "evidence_image_path", "vehicle_crop_path"):
        assert Path(rows[0][key]).stat().st_size > 0
    assert video_metadata(result["video"]) == (180, 120, 10, len(POINTS))
    with result["uncertain_report"].open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []
