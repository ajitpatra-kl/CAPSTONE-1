"""LCVD geometry, reliability, persistence and output regression coverage."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.lcvd.configuration import Configuration, Settings
from modules.lcvd.lane_change_detector import LaneChangeDetector, Observation
from modules.lcvd.lane_geometry import LaneGeometry
from modules.lcvd.video_processor import process_video
from modules.rlvd.configuration import video_metadata
from modules.speed_detection.vehicle_detector import Detection


def boundaries(kind="Restricted"):
    return [{"kind": classification, "points": [[x, 10], [x, 130]]}
            for x, classification in ((25, "Restricted"), (100, kind), (175, "Restricted"))]


def configuration(kind="Restricted"):
    return Configuration("test", "test-fixture", 200, 140, 10, boundaries(kind),
                         settings=Settings(stable_frames=3))


def run(engine, points, missing=()):
    events = []
    for frame, x in enumerate(points):
        if frame not in missing:
            event = engine.observe(7, Observation(frame, frame/10, (x-15, 70, x+15, 100), "car", .9))
            if event:
                events.append(event)
    return events


@pytest.mark.parametrize("x,lane", [(50, 1), (140, 2)])
def test_lane_assignment(x, lane):
    geometry = configuration().geometry()
    assert geometry.lane_number(geometry.locate((x, 70), 4)) == lane


def test_curved_horizontal_geometry_and_reverse_numbering():
    rows = [{"kind": "Restricted", "points": [[10, y], [45, y+5], [130, y+10]]}
            for y in (15, 50, 85)]
    geometry = LaneGeometry(150, 120, rows, "Horizontal", True)
    assert geometry.lane_number(geometry.locate((60, 40), 4)) == 2
    assert geometry.lane_number(geometry.locate((60, 72), 4)) == 1
    assert geometry.crossing((40, 45), (55, 65), geometry.boundaries[1]) == pytest.approx(33/65)


def test_intersecting_boundaries_are_rejected():
    rows = boundaries()
    rows[1]["points"][-1][0] = 20
    with pytest.raises(ValueError, match="intersect"):
        LaneGeometry(200, 140, rows)


@pytest.mark.parametrize("kind,status", [("Restricted", "Confirmed violation"), ("Permitted", "Permitted")])
def test_stable_transition_counts_once_and_allows_a_return(kind, status):
    config = configuration(kind)
    engine = LaneChangeDetector(config.geometry(), config.settings)
    points = [86]*4 + [91, 96, 99, 102, 105, 108, 110] + [115]*12
    points += [110, 106, 103, 101, 98, 95, 92, 89] + [85]*8
    events = run(engine, points)
    assert [(event.source, event.target, event.status) for event in events] == [(0, 1, status), (1, 0, status)]
    assert events[0].sweep.timestamp == pytest.approx((6 + 1/3)/10)


@pytest.mark.parametrize("points", [[55]*20, [91]*4+[98, 101, 99, 102, 101, 97, 92]+[91]*5,
                                    [98, 99, 101, 102]*6, [140]*10])
def test_lane_stability_new_tracks_and_ambiguous_boundary(points):
    config = configuration()
    assert not run(LaneChangeDetector(config.geometry(), config.settings), points)


def test_short_tracking_gap_makes_crossing_uncertain():
    config = configuration()
    engine = LaneChangeDetector(config.geometry(), config.settings)
    points = [86]*4 + [91, 96, 99, 102, 106, 110, 114]
    events = run(engine, points, missing=(6, 7))
    assert len(events) == 1 and events[0].status == "Uncertain"
    assert "Tracking gap" in events[0].reason


def test_long_gap_reseeds_track():
    config = configuration()
    engine = LaneChangeDetector(config.geometry(), config.settings)
    points = [70]*5 + [95]*18 + [130]*6
    assert not run(engine, points, missing=range(5, 23))


def test_skipped_lane_cannot_become_a_confirmed_crossing():
    config = configuration()
    config.width = 280
    config.lines.append({"kind": "Restricted", "points": [[250, 10], [250, 130]]})
    engine = LaneChangeDetector(config.geometry(), config.settings)
    events = run(engine, [86]*4 + [92, 98, 198, 146, 145, 144])
    assert len(events) == 1 and events[0].status == "Uncertain"


def test_overlapping_track_during_crossing_cannot_confirm():
    config = configuration()
    engine = LaneChangeDetector(config.geometry(), config.settings)
    events = []
    for index, x in enumerate([86]*4+[92, 99, 102, 107, 110, 115]):
        reason = "Overlapping tracks near the transition" if index == 6 else ""
        event = engine.observe(7, Observation(index, index/10, (x-15, 70, x+15, 100), "car", .9, reason))
        if event:
            events.append(event)
    assert len(events) == 1 and events[0].status == "Uncertain"
    assert events[0].reason == "Overlapping tracks near the transition"


def test_abrupt_crossing_and_incomplete_destination_are_uncertain():
    config = configuration()
    engine = LaneChangeDetector(config.geometry(), config.settings)
    assert run(engine, [38]*4 + [146]*4)[0].status == "Uncertain"
    engine = LaneChangeDetector(config.geometry(), config.settings)
    assert not run(engine, [86]*4+[92, 99, 102, 106])
    events = list(engine.expire(8, final=True))
    assert len(events) == 1 and events[0][1].status == "Uncertain"


def test_config_roundtrip_preserves_native_coordinates(tmp_path):
    original = configuration()
    original.reverse = True
    original.save(tmp_path, "test")
    assert Configuration.load(tmp_path / "test.json") == original


def test_scaled_resolution_preserves_lane_assignment_and_crossing():
    original = configuration().geometry()
    ratio = 3
    lines = boundaries()
    for line in lines:
        line["points"] = [[x*ratio, y*ratio] for x, y in line["points"]]
    resized = LaneGeometry(600, 420, lines)
    for x, y in ((52, 40), (135, 61), (188, 81)):
        assert original.locate((x, y), 3) == resized.locate((x*ratio, y*ratio), 9)
    assert original.crossing((94, 65), (107, 72), original.boundaries[1]) == pytest.approx(
        resized.crossing((282, 195), (321, 216), resized.boundaries[1]))


def test_video_evidence_and_csv_with_real_bytetrack(tmp_path):
    config = configuration()
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 140))
    assert writer.isOpened()
    positions = [74, 74, 75, 78, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116]
    for x in positions:
        frame = np.zeros((140, 200, 3), np.uint8)
        cv2.rectangle(frame, (x-15, 63), (x+14, 99), (245,)*3, -1)
        writer.write(frame)
    writer.release()

    class FixtureDetector:
        def detect(self, frame):
            x, y, width, height = cv2.boundingRect(cv2.inRange(frame, (210,)*3, (255,)*3))
            return [Detection((x, y, x+width, y+height), .95, 2, "car")]

    result = process_video(path, config, tmp_path, detector=FixtureDetector())
    assert video_metadata(result["video"]) == (200, 140, 10, len(positions))
    assert len(result["events"]) == len(result["violations"]) == 1
    assert not result["uncertain"]
    row = result["violations"][0]
    assert (row["source_lane"], row["destination_lane"], row["boundary"]) == (1, 2, 2)
    assert row["frame_number"] == 11
    expected = {"events": 1, "violations": 1, "uncertain": 0}
    for name, report in result["reports"].items():
        with report.open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == expected[name]
    for key in ("original_frame_path", "evidence_image_path", "vehicle_crop_path"):
        assert cv2.imread(row[key]) is not None
    assert (Path(row["evidence_image_path"]).parent / "event.json").is_file()
