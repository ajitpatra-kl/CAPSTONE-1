"""System A integration tests with real decoding, ByteTrack and rule engines."""

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules.anpr.anpr_processor import ANPRResult
from modules.lcvd.configuration import Configuration as LaneConfig, Settings
from modules.rlvd.configuration import Configuration as RedConfig
from modules.rlvd.violation_engine import Thresholds
from modules.speed_detection.vehicle_detector import Detection
from modules.speed_detection.vehicle_tracker import VehicleTracker
from modules.unified import pipeline
from modules.unified.storage import csv_export, read_records


X = [86]*4 + [91, 96, 99, 102, 105, 108, 110] + [115]*9
Y = [95, 93, 91, 89, 87, 85, 83, 81, 78, 75, 72, 69, 66] + [63, 60, 57, 54, 51, 48, 45]
KINDS = ("overspeed", "red_light", "lane_change")
SETTINGS = {"detection": {"confidence_threshold": .4}, "anpr": {"enabled": True}}


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 140))
    assert writer.isOpened()
    for index in range(len(X)):
        image = np.zeros((140, 200, 3), dtype=np.uint8)
        center, color = ((18, 41), (0, 255, 0)) if index < 4 else ((18, 17), (0, 0, 255))
        cv2.circle(image, center, 4, color, -1)
        writer.write(image)
    writer.release()
    identity = pipeline.video_identity(path)
    lines = [{"kind": "Restricted", "points": [[x, 10], [x, 130]]} for x in (25, 100, 175)]
    lane = LaneConfig("Lane", identity, 200, 140, 10, lines, settings=Settings(stable_frames=3))
    red = RedConfig("Signal", identity, 200, 140, 10, [10, 10, 27, 48], [[55, 80], [125, 80]], "Bottom")
    speed = {"speed_limit": 8., "violation_duration": .2, "speed_estimation": {
        "custom_road": {"image_points": [[0, 0], [190, 0], [190, 130], [0, 130]],
                        "width_m": 38., "length_m": 26.}
    }}
    return path, {"speed_options": speed, "red_config": red, "lane_config": lane}


class Detector:
    def __init__(self, **kwargs):
        self.calls = 0

    def detect(self, image):
        x, y = X[self.calls], Y[self.calls]
        self.calls += 1
        return [Detection((x - 15, y - 22, x + 15, y), .95, 2, "car")]


@pytest.mark.parametrize("selected", [(key,) for key in KINDS] + [KINDS])
def test_unified_session(selected, video, tmp_path, monkeypatch):
    path, options = video
    detector = Detector()
    counts = Counter()
    capture_factory = cv2.VideoCapture

    class Capture:
        def __init__(self, *args):
            counts["captures"] += 1
            self.capture = capture_factory(*args)

        def read(self):
            counts["read"] += 1
            return self.capture.read()

        def __getattr__(self, name):
            return getattr(self.capture, name)

    class ANPR:
        def __init__(self, config, root):
            counts["anpr_init"] += 1
            assert not config["scan_all_tracks"]
            assert config["event_scan"]
            self.scan = SimpleNamespace(by_frame={}, sample_track=lambda *args, **kwargs: None)

        def observe(self, frame, index, active):
            counts["anpr_frames"] += 1
            self.scan.by_frame[index] = [
                SimpleNamespace(track_id=tid, rank=1, bbox=(int(x1)+7, int(y2)-10, int(x2)-7, int(y2)-3))
                for tid, track in active.items() for x1, _, x2, y2 in (track.latest_bbox,)
            ]

        def recognize_track(self, track_id):
            counts["recognize"] += 1
            plate = next(plate for plate in self.scan.by_frame[4] if plate.track_id == track_id)
            return ANPRResult("Text Unreadable", plate_confidence=.2, frame_number=4,
                              plate_bbox=plate.bbox, detection_confidence=.88,
                              plate_crop=np.zeros((10, 20, 3), np.uint8))

    class Tracker(VehicleTracker):
        def __init__(self, **kwargs):
            counts["trackers"] += 1
            super().__init__(**kwargs)

        def update(self, *args):
            counts["updates"] += 1
            return super().update(*args)

    def make_detector(**kwargs):
        counts["detectors"] += 1
        return detector

    def verify(candidate, rule):
        counts["candidates"] += 1
        assert candidate.trajectory[-1]["track_id"] == candidate.track_id
        assert all(Path(path).is_file() for path in candidate.evidence_paths.values())
        return rule()

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(pipeline, "VehicleDetector", make_detector)
    monkeypatch.setattr(pipeline, "ANPRProcessor", ANPR)
    selected_options = {key: value for kind, (key, value) in zip(KINDS, options.items()) if kind in selected}
    result = pipeline.process_video(
        path, tmp_path, SETTINGS, {}, tracker_factory=Tracker, verifier=verify,
        red_thresholds=Thresholds(deadband=2), **selected_options,
    )
    database = tmp_path / "outputs" / "unified" / pipeline.DATABASE
    events = read_records(database, "events", result["session_id"])
    candidates = read_records(database, "candidates", result["session_id"])
    assert {row["violation_type"] for row in events} == set(selected)
    assert {row["track_id"] for row in events} == {1}
    assert {row["session_id"] for row in candidates} == {result["session_id"]}
    assert {row["event_id"] for row in events} <= {row["event_id"] for row in candidates}
    assert all(row["number_plate"] is None and row["plate_status"] == "Text Unreadable"
               and row["ocr_confidence"] == .2 for row in events)
    assert counts == Counter(captures=1, read=len(X)+1, detectors=1, trackers=1, updates=len(X),
                             anpr_init=1, anpr_frames=len(X), recognize=len(events), candidates=len(candidates))
    for row in events:
        paths = row["evidence_paths"]
        for key in ("video", f"frame_{row['frame_number']}", "annotated", "vehicle", "plate", "plate_enhanced"):
            assert Path(paths[key]).is_file()
        annotated = cv2.imread(paths["annotated"])
        assert ((annotated[:, :, 2] > 200) & (annotated[:, :, 1] < 60)).any()
        assert ((annotated[:, :, 2] > 190) & (annotated[:, :, 1] > 190) & (annotated[:, :, 0] < 90)).any()
        assert row["plate_bbox"] and row["plate_detection_confidence"] == .88
        assert row["plate_frame_number"] == 4
        assert Path(paths["plate_enhanced"]).suffix == ".png"
        assert cv2.imread(paths["plate_enhanced"], cv2.IMREAD_GRAYSCALE).shape[0] > 10
    assert detector.calls == result["frames"] == len(X)
    assert result["unique_tracks"] == 1
    assert result["invalid_timestamps"] == 0
    assert read_records(database, "sessions") == [result]
    encoded = capture_factory(result["annotated_video"])
    try:
        assert encoded.read()[0]
        assert encoded.get(cv2.CAP_PROP_FRAME_COUNT) == len(X)
    finally:
        encoded.release()
    assert all(row["verification_status"] == "Confirmed violation" for row in events)
    track_rows = list(csv.DictReader(io.StringIO(Path(result["tracks_csv"]).read_text())))
    assert len(track_rows) == len(X)
    assert {row["track_id"] for row in track_rows} == {"1"}
    if "overspeed" in selected:
        assert "Estimated speeds" in result["calibration"]
        assert any(row["speed_kmh"] for row in track_rows)
    if "red_light" in selected:
        event = next(row for row in events if row["violation_type"] == "red_light")
        assert event["signal_information"]["red_onset_frame"] == 4
        assert event["timestamp"] == pytest.approx(.7333333333333)
        assert len(event["evidence_frames"]) == 2
    if "lane_change" in selected:
        event = next(row for row in events if row["violation_type"] == "lane_change")
        assert event["lane_information"]["boundary_type"] == "Restricted"
        assert len(event["evidence_frames"]) == 2
        assert {row["lane_id"] for row in track_rows} == {"", "1", "2"}
    exported = list(csv.DictReader(io.StringIO(csv_export(events))))
    assert {row["event_id"] for row in exported} == {row["event_id"] for row in events}
    from modules.tcvv.storage import read_results, run_verification

    original_calls = counts.copy()
    evidence = {Path(path): Path(path).stat().st_mtime_ns for row in events
                for path in row["evidence_paths"].values() if path}
    tcvv_run = run_verification(database, result["session_id"])
    decisions = read_results(database, result["session_id"], tcvv_run["run_id"])
    assert {row["candidate_id"] for row in decisions} == {row["event_id"] for row in candidates}
    expected = {"overspeed": "UNCERTAIN", "red_light": "CONFIRMED", "lane_change": "CONFIRMED"}
    assert all(row["decision"] == expected[row["violation_type"]] for row in decisions), decisions
    assert counts == original_calls and detector.calls == len(X)
    assert read_records(database, "events", result["session_id"]) == events
    assert read_records(database, "candidates", result["session_id"]) == candidates
    assert all(path.stat().st_mtime_ns == modified for path, modified in evidence.items())


def test_permitted_transition_and_anpr_failure(video, tmp_path, monkeypatch):
    path, options = video
    lane = options["lane_config"]
    lane.lines[1]["kind"] = "Permitted"

    class BrokenANPR:
        def __init__(self, **kwargs):
            pass

        def observe(self, *args):
            pass

        def recognize_track(self, *args):
            raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(pipeline, "ANPRProcessor", lambda *args: BrokenANPR())
    report = pipeline.process_video(path, tmp_path, SETTINGS, {}, detector=Detector(),
                                    red_thresholds=Thresholds(deadband=2), **options)
    database = tmp_path / "outputs" / "unified" / pipeline.DATABASE
    candidates = read_records(database, "candidates", report["session_id"])
    events = read_records(database, "events", report["session_id"])
    assert any(row["verification_status"] == "Permitted" for row in candidates)
    assert {row["violation_type"] for row in events} == {"overspeed", "red_light"}
    assert all(row["plate_status"] == "Processing Failed" for row in events)
    second = pipeline.process_video(path, tmp_path, {**SETTINGS, "anpr": {}}, {}, detector=Detector(),
                                    red_config=options["red_config"], red_thresholds=Thresholds(deadband=2))
    other, = read_records(database, "events", second["session_id"])
    assert second["session_id"] != report["session_id"]
    assert other["track_id"] == 1 and other["plate_status"] == "Unavailable"
    assert len(read_records(database, "sessions")) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_geometry_validation(kind, video, tmp_path):
    path, options = video
    if kind == "overspeed":
        options["speed_options"]["speed_estimation"]["custom_road"]["image_points"] = [[0, 0]]*4
        invalid = {"speed_options": options["speed_options"]}
        message = "four ordered frame corners"
    elif kind == "red_light":
        invalid = {"red_config": replace(options["red_config"], roi=[0, 0, 1, 1])}
        message = "signal rectangle"
    else:
        invalid = {"lane_config": replace(options["lane_config"], lines=[])}
        message = "at least three boundaries"
    with pytest.raises(ValueError, match=message):
        pipeline.process_video(path, tmp_path, SETTINGS, {}, detector=Detector(), **invalid)
    assert not (tmp_path / "outputs" / "unified" / pipeline.DATABASE).exists()


def test_lane_monitoring_direction(video, tmp_path):
    path, options = video
    run = pipeline.process_video(
        path, tmp_path, {"detection": SETTINGS["detection"]}, {}, detector=Detector(),
        lane_config=options["lane_config"], lane_direction="increasing",
    )
    database = tmp_path / "outputs" / "unified" / pipeline.DATABASE
    candidate, = read_records(database, "candidates", run["session_id"])
    assert candidate["lane_information"]["direction"] == "decreasing"
    assert candidate["verification_status"] == "Outside monitoring direction"
    assert not read_records(database, "events", run["session_id"])


def test_verification_disposition_leaves_candidate_generation_stable(video, tmp_path):
    path, options = video
    database = tmp_path / "outputs" / "unified" / pipeline.DATABASE
    session_options = {"detector": Detector(), "red_thresholds": Thresholds(deadband=2), **options}
    baseline = pipeline.process_video(path, tmp_path, {"detection": SETTINGS["detection"]}, {}, **session_options)
    session_options["detector"] = Detector()
    second = pipeline.process_video(
        path, tmp_path, {"detection": SETTINGS["detection"]}, {}, **session_options,
        verifier=lambda candidate, rule: "Rejected",
    )
    keys = ("video_id", "track_id", "violation_type", "frame_number")
    first_rows = read_records(database, "candidates", baseline["session_id"])
    second_rows = read_records(database, "candidates", second["session_id"])
    assert len(first_rows) == len(second_rows) > 0
    assert [tuple(row[key] for key in keys) for row in first_rows] == [
        tuple(row[key] for key in keys) for row in second_rows
    ]
    assert not read_records(database, "events", second["session_id"])


def test_unified_streamlit_entry_and_report_navigation(monkeypatch, video):
    import streamlit as st

    from modules.unified import dashboard

    source, _ = video
    upload = io.BytesIO(source.read_bytes())
    upload.name = source.name
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.setattr(dashboard, "read_records", lambda *_: [])
    path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(path)).run()
    assert not app.exception
    assert app.multiselect(key="unified_selected").options == [
        "Overspeed", "Red-light violation", "Lane-change violation",
    ]
    assert any(item.label == "Road calibration" for item in app.selectbox)
    assert not any(item.label == "Side vehicles approach from" for item in app.selectbox)
    app.multiselect(key="unified_selected").set_value(["red_light", "lane_change"]).run()
    assert not app.exception
    assert any(item.label == "Side vehicles approach from" for item in app.selectbox)
    assert any(item.label == "Road orientation" for item in app.selectbox)
    assert any(item.label == "Monitoring direction" for item in app.selectbox)
    assert not any(item.label == "Road calibration" for item in app.selectbox)
    app.radio(key="traffic_navigation").set_value("reports").run()
    assert not app.exception
    assert app.selectbox(key="traffic_report_kind").value == "Unified sessions"
    assert any("unified detection run" in item.value for item in app.info)

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)
    app.radio(key="traffic_navigation").set_value("unified").run()
    assert not app.exception
    assert app.multiselect(key="unified_selected").value == ["red_light", "lane_change"]
    assert any(item.label == "Scene preview frame" for item in app.number_input)
    start = next(button for button in app.button if button.label == "Start processing")
    assert start.disabled

    calls = []

    def process_once(path, *args, **options):
        calls.append(options)
        return {"video_id": pipeline.video_identity(path)}

    monkeypatch.setattr(dashboard, "process_video", process_once)
    monkeypatch.setattr(dashboard, "render_summary", lambda *args: None)
    app.multiselect(key="unified_selected").set_value(["overspeed"]).run()
    assert not app.exception and not calls
    next(button for button in app.button if button.label == "Start processing").click().run()
    assert not app.exception and len(calls) == 1
    app.radio(key="traffic_navigation").set_value("reports").run()
    app.radio(key="traffic_navigation").set_value("unified").run()
    assert not app.exception and len(calls) == 1



@pytest.mark.parametrize(("kind", "option", "configure"), [
    ("rlvd", "red_config", "configure_red"),
    ("lcvd", "lane_config", "configure_lane"),
])
def test_unified_applies_matching_saved_geometry(video, tmp_path, kind, option, configure):
    _, settings = video
    scene = settings[option]
    folder = tmp_path / "configs" / kind
    folder.mkdir(parents=True)
    (folder / "test-preset.json").write_text(json.dumps(asdict(scene)), encoding="utf-8")
    script = f"""
from pathlib import Path
import streamlit as st
from modules.unified.dashboard import {configure}
config, _ = {configure}(Path({str(tmp_path)!r}), {scene.video_id!r}, "",
                        {scene.width}, {scene.height}, {scene.fps})
if config.name == {scene.name!r}:
    {"config.validate()" if kind == "rlvd" else "config.geometry()"}
    st.success("Matching geometry loaded")
"""
    app = AppTest.from_string(script).run()
    assert not app.exception and not app.success
    next(widget for widget in app.selectbox if widget.label == "Saved geometry").set_value("test-preset").run()
    assert not app.exception
    assert app.success[0].value == "Matching geometry loaded"
