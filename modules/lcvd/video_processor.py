"""Frame-by-frame YOLO11n and ByteTrack processing for LCVD."""

from fractions import Fraction
from itertools import combinations
from pathlib import Path
import subprocess
from uuid import uuid4

import cv2
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

from modules.rlvd.configuration import read_frame, video_metadata
from modules.speed_detection.vehicle_detector import VehicleDetector
from modules.speed_detection.vehicle_tracker import VehicleTracker

from .evidence_generator import event_row, save_evidence, write_csv
from .lane_change_detector import LaneChangeDetector, Observation


GREEN, YELLOW, RED = (95, 225, 120), (50, 220, 255), (55, 70, 245)
GRAY = (180, 180, 180)


def label(frame, text, origin, color, scale=.5):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(text, font, scale, 1)
    x = min(max(0, x), max(0, frame.shape[1]-width-8))
    y = min(max(height+4, y), frame.shape[0]-4)
    cv2.rectangle(frame, (x, y-height-4), (x+width+7, y+3), (28, 28, 32), -1)
    cv2.putText(frame, text, (x+3, y), font, scale, color, 1, cv2.LINE_AA)


def draw_geometry(frame, geometry):
    for boundary in geometry.boundaries:
        points = np.int32([geometry.image_point(*point) for point in boundary.points])
        color = RED if boundary.kind == "Restricted" else GREEN
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)
        along = geometry.start + .22 * (geometry.end - geometry.start)
        x, y = map(int, geometry.image_point(along, boundary.lateral_at(along)))
        label(frame, f"B{boundary.number}", (x, y), color, .42)
    along = geometry.start + .63 * (geometry.end - geometry.start)
    for index, (left, right) in enumerate(zip(geometry.boundaries, geometry.boundaries[1:])):
        lateral = (left.lateral_at(along) + right.lateral_at(along)) / 2
        x, y = map(int, geometry.image_point(along, lateral))
        label(frame, f"Lane {geometry.lane_number(index)}", (x-24, y), (245, 245, 245), .44)


def draw_vehicle(frame, observation, text, color):
    x1, y1, x2, y2 = (round(value) for value in observation.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, tuple(map(round, observation.anchor)), 3, color, -1)
    label(frame, text, (x1, y1-4), color, .42)


def compress_snapshot(frame):
    ok, data = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError("Could not retain the crossing frame for evidence.")
    return data


def overlapping_tracks(active):
    boxes = [(key, track.latest_bbox) for key, track in active.items()]
    ambiguous = set()
    for (first, (ax, ay, bx, by)), (second, (cx, cy, dx, dy)) in combinations(boxes, 2):
        overlap = max(0, min(bx, dx)-max(ax, cx)) * max(0, min(by, dy)-max(ay, cy))
        smaller = min((bx-ax)*(by-ay), (dx-cx)*(dy-cy))
        if smaller > 0 and overlap >= .60 * smaller:
            ambiguous.update((first, second))
    return ambiguous


def process_video(path, config, root, progress=lambda *args: None, detector=None):
    geometry = config.geometry()
    width, height, fps, count = video_metadata(path)
    if (width, height) != (config.width, config.height) or abs(fps-config.fps) > .001:
        raise ValueError("The video resolution or frame rate differs from this configuration.")
    progress(0, "Loading YOLO11n and ByteTrack")
    detector = detector if detector is not None else VehicleDetector(confidence_threshold=.15)
    tracker = VehicleTracker(track_high_thresh=.35, track_low_thresh=.14, new_track_thresh=.50)
    if tracker.name != "ByteTrack":
        raise RuntimeError("ByteTrack is unavailable. Check the tracking dependencies.")
    engine = LaneChangeDetector(geometry, config.settings)
    directory = Path(root) / "outputs" / "lcvd" / uuid4().hex[:12]
    directory.mkdir(parents=True)
    output = directory / "annotated.mp4"
    rate = str(Fraction(fps).limit_denominator(100000))
    pixel_format = "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"
    command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", rate, "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", pixel_format,
               "-movflags", "+faststart", str(output)]
    events, violations, uncertain = [], [], []
    snapshots = {}
    previous_frame = None
    total_detections = index = 0
    capture = cv2.VideoCapture(str(path))
    log_path = directory / "encoder.log"

    def record(track_id, event, frame):
        row = event_row(geometry, event, track_id, len(events)+len(uncertain)+1, frame)
        if event.status == "Uncertain":
            uncertain.append(row)
            return
        if event.status == "Confirmed violation":
            before, crossing = (cv2.imdecode(data, cv2.IMREAD_COLOR) for data in snapshots[track_id])

            def annotate(image):
                draw_geometry(image, geometry)
                draw_vehicle(image, event.observation,
                             f"#{track_id} | Lane {row['source_lane']} > {row['destination_lane']} | Restricted",
                             RED)
                label(image, f"{row['violation_id']} | B{row['boundary']} | {row['timestamp_sec']:.3f}s",
                      (8, 20), RED)

            save_evidence(directory / row["violation_id"], row, event, before, crossing, annotate)
            violations.append(row)
        events.append(row)

    try:
        with log_path.open("wb") as log, subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log) as encoder:
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (height, width):
                        raise ValueError(f"Frame {index} changed resolution.")
                    reported = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    timestamp = index / fps
                    if index and reported > 0 and abs(reported - timestamp) > .55 / fps:
                        raise ValueError("This clip has irregular timestamps. Export it at a constant frame rate first.")
                    detections = detector.detect(frame)
                    total_detections += len(detections)
                    active = tracker.update(detections, frame, timestamp)
                    ambiguous = overlapping_tracks(active)
                    confidence = {item.bbox: item.confidence for item in detections}
                    canvas = frame.copy()
                    draw_geometry(canvas, geometry)
                    label(canvas, f"{timestamp:.3f}s | Frame {index}", (8, 19), (245, 245, 245))
                    snapshot = None
                    for track_id, track in active.items():
                        box = track.latest_bbox
                        reason = "Overlapping tracks near the transition" if track_id in ambiguous else ""
                        observation = Observation(index, timestamp, box, track.class_name, confidence.get(box, 0), reason)
                        event = engine.observe(track_id, observation)
                        state = engine.tracks[track_id]
                        sweep = event.sweep if event else (state.pending.sweep if state.pending else state.sweep)
                        if sweep and sweep.after.frame == index and sweep.boundary.kind == "Restricted":
                            if snapshot is None:
                                snapshot = (compress_snapshot(previous_frame), compress_snapshot(frame))
                            snapshots[track_id] = snapshot
                        if event:
                            record(track_id, event, index)
                            state.annotated_until = index + round(fps)
                        if not sweep or event:
                            snapshots.pop(track_id, None)
                        text = f"#{track_id} {track.class_name}"
                        color = GRAY
                        if state.stable is not None:
                            text += f" | Lane {geometry.lane_number(state.stable)}"
                            color = GREEN
                        if state.sweep or state.pending:
                            color, text = YELLOW, text + " | Transition"
                        elif state.latest and index <= state.annotated_until:
                            event = state.latest
                            color = RED if event.status == "Confirmed violation" else (YELLOW if event.status == "Uncertain" else GREEN)
                            event_time = event.sweep.timestamp if event.sweep else event.first.timestamp
                            text += f" | {event.status} {event_time:.2f}s"
                        draw_vehicle(canvas, observation, text, color)
                    for track_id, event in engine.expire(index):
                        snapshots.pop(track_id, None)
                        record(track_id, event, index)
                    for track_id in snapshots.keys() - active.keys():
                        if index - engine.tracks[track_id].last.frame > config.settings.max_gap:
                            snapshots.pop(track_id)
                    encoder.stdin.write(canvas.tobytes())
                    previous_frame = frame
                    index += 1
                    if index == 1 or index % 15 == 0:
                        progress(min(.99, index/count), f"Processed {index:,} of {count:,} frames")
            finally:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    if not encoder.poll():
                        raise
            if encoder.wait():
                raise OSError(f"Video encoding failed. See {log_path}.")
    except BrokenPipeError as error:
        raise OSError(f"Video encoder stopped early. See {log_path}.") from error
    finally:
        capture.release()

    if index != count:
        raise ValueError(f"Decoded {index:,} of {count:,} expected frames. The run is incomplete.")
    out_width, out_height, out_fps, out_count = video_metadata(output)
    if (out_width, out_height, out_count) != (width, height, count) or abs(out_fps-fps) > .001:
        raise OSError("The annotated video failed frame count, resolution, or frame rate verification.")
    read_frame(output, count-1)
    for track_id, event in engine.expire(index, final=True):
        record(track_id, event, index-1)
    reports = {
        "events": directory / "lane_change_events.csv",
        "violations": directory / "lane_change_violations.csv",
        "uncertain": directory / "lane_change_uncertain.csv",
    }
    for path, rows in zip(reports.values(), (events, violations, uncertain)):
        write_csv(path, rows)
    progress(1, f"Complete · verified {count:,} frames")
    return dict(video=output, reports=reports, events=events, violations=violations, uncertain=uncertain,
                metrics={"Total Vehicles Detected": total_detections,
                         "Unique Vehicles Tracked": len(engine.tracks),
                         "Lane Change Events": len(events),
                         "Confirmed Lane Change Violations": len(violations),
                         "Permitted Lane Changes": len(events)-len(violations),
                         "Uncertain Events": len(uncertain)})
