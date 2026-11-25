"""Run RLVD and write isolated video, evidence and CSV artifacts."""

import csv
import json
import logging
from math import ceil, floor
from pathlib import Path
from shutil import copyfile
import subprocess
from uuid import uuid4

import cv2
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

from modules.anpr.anpr_processor import ANPRProcessor
from modules.speed_detection.vehicle_detector import VehicleDetector
from modules.speed_detection.vehicle_tracker import VehicleTracker

from .configuration import video_metadata
from .signal_detector import COLORS, UNKNOWN, SignalHistory, classify_signal
from .violation_engine import CrossingEngine


REPORT_FIELDS = (
    "violation_id", "track_id", "timestamp", "frame_number", "vehicle_class",
    "vehicle_confidence", "signal_state", "red_onset_timestamp", "crossing_timestamp",
    "crossing_status", "plate_number", "plate_read_status", "evidence_image_path",
    "vehicle_crop_path", "original_image_path", "plate_crop_path", "evidence_quality",
    "previous_anchor", "current_anchor", "reason",
)
TRACK_COLORS = {"Normal": (225, 170, 35), "Approaching": (255, 215, 0),
                "Legal crossing": (65, 220, 90), "Violation": (45, 45, 255),
                "Uncertain crossing": (30, 155, 255)}


def draw_geometry(frame, config, state):
    x1, y1, x2, y2 = config.roi
    color = COLORS[state]
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2)
    cv2.line(frame, *[tuple(p) for p in config.line], (255, 245, 200), 3, cv2.LINE_AA)
    if config.lane:
        cv2.polylines(frame, [np.asarray(config.lane, np.int32)], True, (225, 170, 35), 2)
    cv2.putText(frame, f"Signal: {state}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .72, color, 2, cv2.LINE_AA)


def draw_track(frame, track_id, observation, status):
    x1, y1, x2, y2 = map(int, observation.bbox)
    color = TRACK_COLORS[status]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"#{track_id} {observation.vehicle_class} | {status}",
                (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)


def write_image(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())
    return str(path)


def report_row(track_id, event, fps):
    observation = event.after
    return dict(
        track_id=track_id, timestamp=round(observation.frame / fps, 6),
        frame_number=observation.frame, vehicle_class=observation.vehicle_class,
        vehicle_confidence=round(observation.confidence, 4), signal_state=event.signal,
        red_onset_timestamp=round(event.onset / fps, 6) if event.onset >= 0 else "",
        crossing_timestamp=round(event.frame / fps, 6), crossing_status=event.status,
        previous_anchor=event.before.anchor, current_anchor=observation.anchor,
        reason=event.reason,
    )


def evidence(directory, row, event, frame, config, plate):
    directory.mkdir()
    annotated = frame.copy()
    draw_geometry(annotated, config, event.signal)
    draw_track(annotated, row["track_id"], event.after, event.status)
    left, top, right, bottom = event.after.bbox
    crop = frame[max(0, floor(top)):ceil(bottom), max(0, floor(left)):ceil(right)]
    number = plate.vehicle_number if plate else ""
    row.update(
        violation_id=directory.name, plate_number=number or "Not Readable",
        plate_read_status="Read" if number else "Not Readable",
        evidence_quality="Observed red onset and continuous track",
        original_image_path=write_image(directory / "original.jpg", frame),
        evidence_image_path=write_image(directory / "annotated.jpg", annotated),
        vehicle_crop_path=write_image(directory / "vehicle.jpg", crop),
        plate_crop_path=write_image(directory / "plate.jpg", plate.plate_crop)
        if plate is not None and plate.plate_crop is not None else "",
    )
    (directory / "event.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def write_report(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_video(path, config, thresholds, root, anpr_settings=None, progress=lambda *args: None,
                  detector=None):
    config.validate()
    width, height, fps, count = video_metadata(path)
    if (width, height) != (config.width, config.height) or abs(fps - config.fps) > .001:
        raise ValueError("Video metadata differs from the saved configuration.")
    progress(0, "Loading vehicle detector and ByteTrack")
    detector = detector if detector is not None else VehicleDetector(confidence_threshold=.15)
    tracker = VehicleTracker(track_high_thresh=.35, track_low_thresh=.14, new_track_thresh=.5)
    if tracker.name != "ByteTrack":
        raise RuntimeError("ByteTrack is unavailable. Check the existing tracking dependencies.")
    anpr = None
    warnings = []
    if anpr_settings is not None:
        try:
            anpr = ANPRProcessor({**anpr_settings, "scan_all_tracks": False}, root)
        except Exception:
            logging.exception("RLVD ANPR initialization failed")
            warnings.append("ANPR could not start. Reports will show plates as Not Readable.")
    history = SignalHistory(thresholds.signal_frames)
    engine = CrossingEngine(config, thresholds, history)
    directory = root / "outputs" / "rlvd" / uuid4().hex[:12]
    directory.mkdir(parents=True)
    confirmed, uncertain = [], []
    snapshots = {}
    output = directory / "annotated.mp4"
    capture = cv2.VideoCapture(str(path))
    detections_total = legal = index = unknown = 0
    try:
        with (directory / "encoder.log").open("wb") as log:
            command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                       "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(output)]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log) as encoder:
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if frame.shape[:2] != (height, width):
                            raise ValueError(f"Frame {index} changed resolution.")
                        signal = history.update(*classify_signal(frame, config.roi, config.lamp_layout))
                        unknown += signal.state == UNKNOWN
                        detections = detector.detect(frame)
                        detections_total += len(detections)
                        active = tracker.update(detections, frame, index / fps)
                        confidences = {det.bbox: det.confidence for det in detections}
                        if anpr:
                            anpr.observe(frame, index, active)
                        annotated = frame.copy()
                        draw_geometry(annotated, config, signal.state)
                        for track_id, track in active.items():
                            bbox = track.latest_bbox
                            event = engine.observe(track_id, index, bbox, track.class_name,
                                                   confidences.get(bbox, 0))
                            state = engine.tracks[track_id]
                            if state.pending and state.pending.after.frame == index:
                                # Only potentially confirmable crossings need a native-resolution frame.
                                first, last = state.pending.before.frame, state.pending.after.frame
                                snapshots[track_id] = frame if (
                                    state.pending.reliable and history.samples[first].state == "RED"
                                    and history.samples[last].state == "RED"
                                ) else None
                            if state.pending and index - state.pending.after.frame > max(fps, thresholds.crossing_frames):
                                event = engine.finish(state, incomplete=True)
                            if event:
                                row = report_row(track_id, event, fps)
                                if event.status == "Violation":
                                    plate = anpr.recognize_track(track_id) if anpr else None
                                    evidence(directory / f"RLVD-{len(confirmed) + 1:04}", row, event,
                                             snapshots.get(track_id, frame), config, plate)
                                    confirmed.append(row)
                                elif event.status == "Uncertain crossing":
                                    uncertain.append(row)
                                else:
                                    legal += 1
                            if state.pending is None:
                                snapshots.pop(track_id, None)
                            points = [tuple(map(int, obs.anchor)) for obs in list(state.trajectory)[-14:]]
                            if len(points) > 1:
                                cv2.polylines(annotated, [np.asarray(points, np.int32)], False,
                                              TRACK_COLORS[state.status], 1, cv2.LINE_AA)
                            draw_track(annotated, track_id, state.trajectory[-1], state.status)
                        for track_id in snapshots.keys() - active.keys():
                            state = engine.tracks[track_id]
                            if index - state.trajectory[-1].frame > 1:
                                uncertain.append(report_row(track_id, engine.finish(state, True), fps))
                                del snapshots[track_id]
                        encoder.stdin.write(annotated.tobytes())
                        index += 1
                        if index == 1 or index % 10 == 0:
                            progress(min(.99, index / count), f"Processed {index:,} of {count:,} frames")
                finally:
                    encoder.stdin.close()
                if encoder.wait() or not index:
                    raise RuntimeError(f"Could not encode annotated video. See {directory / 'encoder.log'}")
    finally:
        capture.release()
    for track_id in snapshots:
        uncertain.append(report_row(track_id, engine.finish(engine.tracks[track_id], True), fps))
    if index < count:
        warnings.append(f"Decoding ended after {index:,} of {count:,} advertised frames.")
    report = directory / "rlvd_report.csv"
    uncertain_report = directory / "rlvd_uncertain_events.csv"
    write_report(report, confirmed)
    write_report(uncertain_report, uncertain)
    latest = directory.parent / f".{directory.name}-report.tmp"
    try:
        copyfile(report, latest)
        latest.replace(directory.parent / report.name)
    finally:
        latest.unlink(missing_ok=True)
    progress(1, "Processing finished")
    return dict(video=output, report=report, uncertain_report=uncertain_report, confirmed=confirmed,
                uncertain=uncertain, warnings=warnings, metrics={
                    "Vehicle detections": detections_total, "Unique vehicle tracks": len(engine.tracks),
                    "Vehicles approaching": len(engine.approaching),
                    "Vehicles crossing": legal + len(confirmed) + len(uncertain),
                    "Legal crossings": legal, "Confirmed violations": len(confirmed),
                    "Uncertain events": len(uncertain), "Unknown signal frames": unknown,
                })
