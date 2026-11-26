"""TCVV decisions, persistence, evaluation and Streamlit behavior."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules.lcvd.configuration import Configuration as LaneConfig, Settings
from modules.rlvd.configuration import Configuration as RedConfig
from modules.rlvd.violation_engine import Thresholds
from modules.tcvv.context import Candidate, Observation, TemporalContext, availability, contiguous
from modules.tcvv.evaluation import Label, compare, parse_labels, score
from modules.tcvv.storage import read_results, read_runs, run_verification
from modules.tcvv.verification import VERSION, Verifier
from modules.unified.storage import read_records, save_session


def observation(frame, x=45, y=70, speed=50):
    return Observation(frame, frame / 10, True, (x, y), (x-5, y-10, x+5, y), .9, speed, 0, None, None)


def candidate(kind, frame, **options):
    values = dict(event_id="candidate", kind=kind, track_id=1, frame=frame, timestamp=frame/10,
                  original="Confirmed violation", evidence_frames=(frame,), metadata={}, speed_limit=None, lanes={})
    return Candidate(**(values | options))


def test_context_windows_ordering_missing_and_track_isolation():
    rows = [observation(4), replace(observation(1), timestamp=.4), observation(2)]
    context = TemporalContext({1: rows, 2: [observation(1, x=65)]})
    assert [row.frame for row in context.window(1, 1, 4)] == [1, 2, 4]
    assert context.window(2, 1, 4)[0].anchor[0] == 65
    assert context.window(3, 0, 4) == []
    assert context.window(1, 3, 3) == []
    assert not contiguous(context.window(1, 1, 2), 10)
    assert not contiguous(context.window(1, 2, 4), 10)
    assert not contiguous([observation(1), replace(observation(2), missing_frames=1)], 10)


@pytest.mark.parametrize("speeds,calibration,decision,reason", [
    ([50]*6, "Verified calibration: test", "CONFIRMED", "SUSTAINED_VERIFIED_EXCEEDANCE"),
    ([35]*5 + [50], "Verified calibration: test", "REJECTED", "OBSERVED_BELOW_LIMIT"),
    ([50]*5 + [None], "Verified calibration: test", "UNCERTAIN", "INSUFFICIENT_VALID_SPEED"),
    ([50]*5 + [-1], "Verified calibration: test", "UNCERTAIN", "INSUFFICIENT_VALID_SPEED"),
    ([50]*6, "Estimated speeds using assumed homography.", "UNCERTAIN", "ASSUMED_CALIBRATION"),
    ([50]*6, "Reference calibration: test", "UNCERTAIN", "ASSUMED_CALIBRATION"),
    ([50]*6, "Calibration unavailable", "UNCERTAIN", "UNVERIFIED_CALIBRATION"),
    ([50]*2, "Verified calibration: test", "UNCERTAIN", "INSUFFICIENT_VALID_SPEED"),
    ([50]*5 + [150], "Verified calibration: test", "UNCERTAIN", "INCONSISTENT_SPEED"),
])
def test_speed(speeds, calibration, decision, reason):
    session = {"configuration": {"speed": {"violation_duration": .4}}, "nominal_fps": 10,
               "calibration": calibration}
    rows = [observation(i, speed=speed) for i, speed in enumerate(speeds)]
    result = Verifier(session, TemporalContext({1: rows})).evaluate(candidate("overspeed", len(rows)-1, speed_limit=40))
    assert (result["decision"], result["reason_codes"]) == (decision, [reason])


@pytest.mark.parametrize("scenario,decision,reason", [
    ("red", "CONFIRMED", "DIRECTED_CROSSING_AFTER_RED_ONSET"),
    ("green", "REJECTED", "NON_RED_AT_CROSSING"),
    ("approach", "REJECTED", "NO_DIRECTED_CROSSING"),
    ("transition", "UNCERTAIN", "SIGNAL_AMBIGUOUS_AT_CROSSING"),
    ("missing", "UNCERTAIN", "MISSING_SIGNAL_OBSERVATIONS"),
    ("gap", "UNCERTAIN", "INTERRUPTED_CROSSING"),
])
def test_red(scenario, decision, reason):
    config = RedConfig("Signal", "video", 120, 120, 10, [2, 2, 15, 20], [[10, 50], [110, 50]], "Bottom")
    session = {"configuration": {"red_light": asdict(config), "red_thresholds": asdict(Thresholds())},
               "nominal_fps": 10}
    positions = [75, 71, 65, 60, 55, 52, 48, 44, 40, 36]
    if scenario == "approach":
        positions[6:] = [50, 51, 53, 55]
    rows = [observation(i, y=y) for i, y in enumerate(positions) if scenario != "gap" or i != 5]
    color = "GREEN" if scenario == "green" else "RED"
    frames = {}
    for i in range(10):
        state = "UNKNOWN" if scenario == "transition" and i == 6 else color
        frames[i] = {"frame": i, "timestamp": i / 10, "timing_valid": True,
                     "signal": {"state": state, "raw": color, "quality": .8, "onset": 1, "observed_onset": True}}
    if scenario == "missing":
        frames.pop(6)
    result = Verifier(session, TemporalContext({1: rows}, frames)).evaluate(
        candidate("red_light", 6, evidence_frames=(5, 6)))
    assert (result["decision"], result["reason_codes"]) == (decision, [reason])
    if scenario == "red":
        assert result["context"]["crossing_timestamp"] == pytest.approx(.55)
        assert result["context"]["red_onset_timestamp"] == .1


@pytest.mark.parametrize("scenario,decision,reason", [
    ("restricted", "CONFIRMED", "SUSTAINED_RESTRICTED_TRANSITION"),
    ("permitted", "REJECTED", "PERMITTED_BOUNDARY"),
    ("touch", "REJECTED", "NO_LANE_TRANSITION"),
    ("partial", "UNCERTAIN", "INCOMPLETE_LANE_TRANSITION"),
    ("no_lane", "UNCERTAIN", "AMBIGUOUS_LANE_ASSIGNMENT"),
    ("overlap", "UNCERTAIN", "INCOMPLETE_LANE_TRANSITION"),
    ("gap", "UNCERTAIN", "INCOMPLETE_LANE_TRANSITION"),
    ("wrong_direction", "REJECTED", "OUTSIDE_MONITORED_DIRECTION"),
])
def test_lanes(scenario, decision, reason):
    lines = [{"kind": "Permitted" if scenario == "permitted" else "Restricted",
              "points": [[x, 0], [x, 119]]} for x in (10, 50, 90)]
    config = LaneConfig("Lanes", "video", 120, 120, 10, lines, settings=Settings(stable_frames=3))
    session = {"configuration": {"lane_change": asdict(config),
                                 "lane_direction": "increasing" if scenario == "wrong_direction" else "decreasing"},
               "nominal_fps": 10}
    x_positions = [35, 38, 41, 48, 52, 57, 60, 62, 64]
    if scenario == "touch":
        x_positions[4:] = [50, 48, 43, 40, 38]
    if scenario == "partial":
        x_positions = x_positions[:6]
    rows = [observation(i, x=x, y=80-i) for i, x in enumerate(x_positions) if scenario != "gap" or i != 4]
    frames = {i: {"lane_overlap_tracks": [1] if scenario == "overlap" and i == 4 else []} for i in range(9)}
    lanes = {} if scenario == "no_lane" else {"source_lane": 1, "destination_lane": 2}
    result = Verifier(session, TemporalContext({1: rows}, frames)).evaluate(
        candidate("lane_change", 4, evidence_frames=(3, 4), lanes=lanes, metadata={"confirmation_frame": 8}))
    assert (result["decision"], result["reason_codes"]) == (decision, [reason])


@pytest.fixture
def saved(tmp_path):
    database = tmp_path / "violations.sqlite3"
    path = tmp_path / "tracks.csv"
    session = {"session_id": "session", "video_id": "video", "tracks_csv": str(path),
               "configuration": {"speed": {"violation_duration": .4}}, "calibration": "Verified calibration: test",
               "nominal_fps": 10, "frames": 6, "processing_fps": 30}
    row = {"event_id": "event", "session_id": "session", "video_id": "video", "track_id": 1,
           "violation_type": "overspeed", "frame_number": 5, "timestamp": .5, "speed_limit": 40,
           "verification_status": "Confirmed violation", "evidence_frames": [5], "evidence_paths": {},
           "metadata": {"consecutive_frames": 6}, "lane_information": None}
    with path.open("w", encoding="utf-8", newline="") as file:
        fields = ["session_id", "track_id", "frame_number", "timestamp", "timing_valid", "bottom_center",
                  "bbox", "confidence", "speed_kmh", "missing_frames", "lane_id", "stop_line_distance"]
        writer = csv.writer(file)
        writer.writerow(fields)
        for i in range(6):
            sample = observation(i)
            writer.writerow(["session", 1, i, sample.timestamp, True, json.dumps(sample.anchor),
                             json.dumps(sample.bbox), .9, 50, 0, None, None])
    save_session(database, session, [row], [row])
    return database, session, row


def test_runs_are_atomic_versioned_and_idempotent(saved, monkeypatch):
    from modules.tcvv import storage

    database, session, row = saved
    assert read_runs(database, "session") == []
    assert availability(session, [row]) is None
    assert "lacks the full signal history" in availability(session, [row | {"violation_type": "red_light"}])
    progress = []
    first = run_verification(database, "session", run_id="stable", progress=lambda *state: progress.append(state))
    assert progress[0][0] == 0 and progress[-1][0] == 1
    assert run_verification(database, "session", run_id="stable") == first
    monkeypatch.setattr(storage, "VERSION", "experimental")
    second = run_verification(database, "session")
    assert first["version"] == VERSION and second["version"] == "experimental"
    assert read_runs(database, "session") == [second, first]
    assert not read_runs(database, "other")
    assert not read_results(database, "other", "stable")
    for run in (first, second):
        results = read_results(database, "session", run["run_id"])
        assert [(result["candidate_id"], result["decision"]) for result in results] == [("event", "CONFIRMED")]
    for table in ("events", "candidates"):
        assert read_records(database, table, "session") == [row]
    assert read_records(database, "sessions", "session") == [session]


def test_context_rejects_mixed_session_and_video(saved):
    database, session, row = saved
    with pytest.raises(ValueError, match="session or video"):
        Candidate.from_record(row | {"video_id": "other"}, session)
    event = Candidate.from_record(row, session)
    records = Path(session["tracks_csv"])
    records.write_text(records.read_text().replace("session,1", "other,1"), encoding="utf-8")
    with pytest.raises(ValueError, match="another session"):
        TemporalContext.load(session, [event])


def test_incomplete_sidecar_blocks_a_new_run(saved):
    database, session, row = saved
    another = session | {"session_id": "red-session"}
    tracks = Path(session["tracks_csv"])
    tracks.write_text(tracks.read_text().replace("session,1", "red-session,1"), encoding="utf-8")
    sidecar = tracks.with_name("tcvv-context-v1.jsonl")
    sidecar.write_text(json.dumps({"session_id": "red-session", "video_id": "video"}) + "\n", encoding="utf-8")
    red = row | {"event_id": "red-event", "session_id": "red-session", "violation_type": "red_light"}
    save_session(database, another, [red], [])
    with pytest.raises(ValueError, match="sidecar is incomplete"):
        run_verification(database, "red-session")
    assert not read_runs(database, "red-session")
    assert read_records(database, "candidates", "red-session") == [red]


def test_failed_result_insert_rolls_back_run(saved, monkeypatch):
    database, _, row = saved
    monkeypatch.setattr(Verifier, "evaluate", lambda *_: {"value": float("nan")})
    with pytest.raises(ValueError, match="Out of range"):
        run_verification(database, "session")
    assert not read_runs(database, "session")
    assert read_records(database, "events", "session") == [row]


def test_evaluation_deduplicates_and_scores_only_labeled_events():
    def result(identity, track, frame, a, b):
        return {"candidate_id": identity, "track_id": track, "violation_type": "overspeed",
                "event_frame": frame, "event_timestamp": frame/10, "original_decision": a, "decision": b}

    results = [result("a", 1, 10, "Confirmed violation", "REJECTED"),
               result("duplicate", 1, 11, "Uncertain overspeed", "UNCERTAIN"),
               result("b", 2, 20, "Confirmed violation", "CONFIRMED"),
               result("c", 3, 30, "Uncertain overspeed", "CONFIRMED")]
    records = {row["candidate_id"]: {"metadata": {"consecutive_frames": 5}} for row in results}
    values = compare(results, records)
    assert values["Total candidates"] == 4 and values["Events compared"] == 3
    assert values["Confirmed only by System A"] == 1 and values["Confirmed only by System B"] == 1
    labels = [Label(1, "overspeed", 1, False), Label(2, "overspeed", 2, True), Label(4, "overspeed", 4, True)]
    metrics, matched, unscored = score(results, records, labels, .2)
    assert (matched, unscored) == (2, 1)
    assert metrics["System A"]["False-positive rate"] == 1
    assert metrics["System B"]["Precision"] == 1
    assert metrics["System B"]["Recall"] == .5
    assert metrics["System B"]["False-negative rate"] == .5
    assert metrics["System B"]["F1-score"] == pytest.approx(2/3)
    header = "session_id,video_id,track_id,violation_type,timestamp,label\n"
    assert parse_labels(header + "s,v,1,overspeed,1,0\n"*2, "s", "v") == labels[:1]
    with pytest.raises(ValueError, match="selected session"):
        parse_labels(header + "other,v,1,overspeed,1,0\n", "s", "v")


def test_tcvv_streamlit_runs_only_on_click_and_reopens_history(saved):
    database, session, row = saved
    script = f"""
from pathlib import Path
from modules.tcvv.dashboard import render_tcvv
from modules.unified.storage import read_records
database = Path({str(database)!r})
session = read_records(database, 'sessions')[0]
candidates = read_records(database, 'candidates', 'session')
render_tcvv(database, session, candidates, [], 'test', lambda *args: None)
"""
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not read_runs(database, "session")
    app.button(key="test_tcvv_run").click().run()
    assert not app.exception
    assert app.success[0].value == "TCVV verification complete."
    assert app.get("progress")[0].proto.value == 100
    assert [tab.label for tab in app.tabs] == ["TCVV results", "System A vs System B"]
    assert app.dataframe[0].value["TCVV decision"].tolist() == ["CONFIRMED"]
    first, = read_runs(database, "session")
    app.run()
    assert read_runs(database, "session") == [first]
    app.button(key="test_tcvv_run").click().run()
    assert not app.exception
    assert len(read_runs(database, "session")) == 2
    historical = AppTest.from_string(script).run()
    historical.selectbox(key="test_tcvv_selected").set_value(first["run_id"]).run()
    assert not historical.exception
    assert historical.selectbox(key="test_tcvv_selected").value == first["run_id"]
    Path(session["tracks_csv"]).unlink()
    historical.run()
    assert not historical.exception
    assert not historical.button
    assert any("missing its stored" in item.value for item in historical.info)
    assert historical.tabs
