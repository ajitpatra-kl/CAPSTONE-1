"""Decode, detect and track once per unified video session."""

import csv
import hashlib
import json
import logging
import subprocess
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from functools import lru_cache
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

from modules.anpr.anpr_processor import ANPRProcessor
from modules.anpr.plate_enhancement import enhance_plate
from modules.rlvd.violation_engine import Thresholds
from modules.speed_detection.camera_calibration import CalibrationManager
from modules.speed_detection.vehicle_detector import VehicleDetector
from modules.speed_detection.vehicle_tracker import VehicleTracker
from .engines import LaneChange, Overspeed, RedLight
from .schema import VehicleContext, VehicleObservation, ViolationCandidate, verify_with_rules
from .storage import save_session


DATABASE = "violations.sqlite3"


NORMAL_COLOR = (225, 195, 50)
VIOLATION_COLOR = (0, 0, 255)
PLATE_COLOR = (0, 255, 255)
VIOLATION_LABELS = {
    "overspeed": "OVERSPEED",
    "red_light": "RED LIGHT VIOLATION",
    "lane_change": "LANE CHANGE VIOLATION",
}


def anpr_profile(settings, dataset_config, video_id):
    options = settings.get("anpr", {})
    profiles = CalibrationManager(dataset_config).profiles
    reference = profiles.get("Roboflow_Vehicles", {})
    if reference.get("sha256") == video_id:
        options = {**options, **options.get("roboflow_scan", {})}
    return {**options, "enabled": options.get("enabled", bool(options)),
            "scan_all_tracks": False, "event_scan": True}


def video_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_image(path, image):
    if image is None or not image.size:
        return None
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    path.write_bytes(data.tobytes())
    return str(path)


class EventCollector:
    def __init__(self, session_id, video_id, directory, verifier, anpr_settings, root):
        self.session_id = session_id
        self.video_id = video_id
        self.directory = directory
        self.verifier = verifier
        self.contexts = {}
        self.candidates = []
        self.events = []
        self.snapshots = {}
        self.frames = {}
        self.active_boxes = {}
        self.scenes = {}
        self.anpr = None
        self.display_seconds = max(0., float(anpr_settings.get("violation_display_seconds", 3.)))
        self.active_violations = {}
        self.confirmed_numbers = {}
        self.anpr_status = "Unavailable"
        self.anpr_error = None
        if anpr_settings.get("enabled", False):
            try:
                self.anpr = ANPRProcessor(anpr_settings, root)
            except Exception as error:
                self.anpr_error = f"{type(error).__name__}: {error}"
                logging.exception("Unified ANPR initialization failed")
                self.anpr_status = "Processing Failed"

    def observe_anpr(self, frame, index, active):
        self.active_boxes = {tid: track.latest_bbox for tid, track in active.items() if track.latest_bbox is not None}
        if self.anpr is not None:
            try:
                self.anpr.observe(frame, index, active)
            except Exception as error:
                self.anpr_error = f"{type(error).__name__}: {error}"
                logging.exception("Unified ANPR crop buffering failed")
                self.anpr = None
                self.anpr_status = "Processing Failed"

    def snapshot(self, tid, kind, *indices):
        key = (tid, kind)
        if not indices:
            self.snapshots.pop(key, None)
            return
        available = {**self.snapshots.get(key, {}), **self.frames}
        self.snapshots[key] = {index: available[index] for index in indices if index in available}

    def submit(self, observation, kind, rule, **details):
        history = self.contexts[observation.track_id].observations
        values = dict(
            event_id=uuid4().hex, session_id=self.session_id, video_id=self.video_id,
            track_id=observation.track_id, violation_type=kind, timestamp=observation.timestamp,
            frame_number=observation.frame_number, vehicle_class=observation.vehicle_class,
            confidence=observation.confidence, bbox=observation.bbox, evidence_frames=[observation.frame_number],
            trajectory=[asdict(sample) for sample in history],
        )
        candidate = ViolationCandidate(**{**values, **details})
        directory = self.directory / "evidence" / str(len(self.candidates) + 1)
        self.save_evidence(candidate, directory)
        # Keep the original engine state independent of a verifier's disposition.
        original_rule = lru_cache(maxsize=1)(rule)
        decision = self.verifier(candidate, original_rule)
        rule_decision = original_rule()
        if decision == "Confirmed violation":
            decision = rule_decision
        row = asdict(candidate)
        row["rule_status"] = rule_decision
        row["verification_status"] = decision
        self.candidates.append(row)
        if decision == "Confirmed violation":
            event = {**row, "evidence_paths": dict(row["evidence_paths"])}
            event["evidence_paths"]["video"] = str(self.directory / "annotated.mp4")
            self.recognize(candidate, event, directory, observation.frame_number)
            if event["number_plate"]:
                self.confirmed_numbers[candidate.track_id] = event["number_plate"]
            self.active_violations.setdefault(candidate.track_id, {})[kind] = observation.frame_number
            self.annotate_evidence(candidate, directory, confirmed=True)
            self.events.append(event)
        return decision

    def save_evidence(self, candidate, directory):
        directory.mkdir(parents=True)
        available = {**self.snapshots.get((candidate.track_id, candidate.violation_type), {}), **self.frames}
        for index in candidate.evidence_frames:
            frame = available.get(index)
            if frame is not None:
                candidate.evidence_paths[f"frame_{index}"] = write_image(directory / f"frame_{index}.jpg", frame)
        frame = available.get(candidate.frame_number)
        if frame is None:
            return
        x1, y1, x2, y2 = candidate.bbox
        crop = frame[max(0, floor(y1)):ceil(y2), max(0, floor(x1)):ceil(x2)]
        path = write_image(directory / "vehicle.jpg", crop)
        if path:
            candidate.evidence_paths["vehicle"] = path
        self.annotate_evidence(candidate, directory)

    def annotate_evidence(self, candidate, directory, confirmed=False):
        available = {**self.snapshots.get((candidate.track_id, candidate.violation_type), {}), **self.frames}
        frame = available.get(candidate.frame_number)
        if frame is None:
            return
        annotated = frame.copy()
        scene = self.scenes.get(candidate.violation_type)
        if scene is not None:
            if candidate.signal_information:
                scene.draw(annotated, candidate.signal_information["samples"][-1]["state"])
            else:
                scene.draw(annotated)
        label = VIOLATION_LABELS[candidate.violation_type] if confirmed else "Candidate"
        draw_track(annotated, candidate.track_id, candidate.bbox, label,
                   VIOLATION_COLOR if confirmed else NORMAL_COLOR)
        if confirmed:
            self.draw_track_plates(annotated, candidate.frame_number, candidate.track_id, candidate.bbox)
        candidate.evidence_paths["annotated"] = write_image(directory / "annotated.jpg", annotated)

    def recognize(self, candidate, event, directory, confirmation_frame):
        event.update(number_plate=None, ocr_confidence=None, plate_status=self.anpr_status,
                     anpr_error=self.anpr_error)
        if self.anpr is None:
            return
        try:
            scan = getattr(self.anpr, "scan", None)
            if scan is not None and confirmation_frame in self.frames:
                scan.sample_track(self.frames[confirmation_frame], confirmation_frame,
                                  candidate.track_id, self.active_boxes, force=True)
            result = self.anpr.recognize_track(candidate.track_id)
            event.update(number_plate=result.vehicle_number or None, ocr_confidence=result.plate_confidence,
                         plate_status=result.status, plate_frame_number=result.frame_number,
                         plate_bbox=result.plate_bbox, plate_detection_confidence=result.detection_confidence,
                         plate_candidate=result.candidate or None, anpr_error=result.error or None)
        except Exception as error:
            event["anpr_error"] = f"{type(error).__name__}: {error}"
            logging.exception("Unified ANPR failed for track %s", candidate.track_id)
            if not event["number_plate"]:
                event["plate_status"] = "Processing Failed"
            return

        paths = event["evidence_paths"]
        try:
            if result.vehicle_crop is not None and result.frame_number != candidate.frame_number:
                paths["plate_source_vehicle"] = write_image(directory / "plate_source_vehicle.png", result.vehicle_crop)
            if result.plate_crop is not None and result.plate_crop.size:
                paths["plate"] = write_image(directory / "plate.png", result.plate_crop)
                paths["plate_enhanced"] = write_image(directory / "plate_enhanced.png", enhance_plate(result.plate_crop))
        except (OSError, cv2.error):
            logging.exception("Could not save plate evidence for track %s", candidate.track_id)


    def draw_track_plates(self, canvas, index, track_id, vehicle_box):
        scan = getattr(self.anpr, "scan", None)
        if scan is None:
            return
        matching = [plate for plate in scan.by_frame.get(index, ()) if plate.track_id == track_id]
        if matching:
            best = max(matching, key=lambda plate: plate.rank)
            draw_plate(canvas, best.bbox, vehicle_box, self.confirmed_numbers.get(track_id),
                       unreadable=track_id in self.active_violations)


def draw_label(canvas, text, x, y, color):
    scale = max(.48, canvas.shape[1] / 2200)
    thickness = max(1, round(scale))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, min(x, canvas.shape[1] - width - 8))
    top = max(0, y - height - baseline - 8)
    cv2.rectangle(canvas, (x, top), (x + width + 8, top + height + baseline + 6), (15, 15, 15), -1)
    cv2.putText(canvas, text, (x + 4, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_track(canvas, tid, bbox, text, color=NORMAL_COLOR):
    x1, y1, x2, y2 = map(int, bbox)
    thickness = max(2, round(canvas.shape[1] / 1000))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    draw_label(canvas, f"#{tid} {text}", x1, y1, color)


def draw_plate(canvas, bbox, vehicle_box, number=None, *, unreadable=False):
    left, top, right, bottom = vehicle_box
    x1, y1 = max(bbox[0], ceil(left), 0), max(bbox[1], ceil(top), 0)
    x2 = min(bbox[2], floor(right), canvas.shape[1])
    y2 = min(bbox[3], floor(bottom), canvas.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), PLATE_COLOR, 2)
    label = number or ("PLATE DETECTED - UNREADABLE" if unreadable else "PLATE DETECTED")
    draw_label(canvas, label, x1, y1, PLATE_COLOR)


def process_video(path, root, settings, dataset_config, *, speed_options=None, red_config=None,
                  lane_config=None, lane_direction=None, red_thresholds=None, video_name=None, detector=None,
                  tracker_factory=VehicleTracker, verifier=verify_with_rules, progress=lambda *args: None):
    """Process selected engines. Optional geometry selects the corresponding engine."""
    if speed_options is None and red_config is None and lane_config is None:
        raise ValueError("Select at least one violation type.")
    root = Path(root)
    path = Path(path)
    video_id = video_identity(path)
    session_id = uuid4().hex
    output_root = (root / "outputs" / "unified").resolve()
    directory = output_root / session_id[:16]
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the input video.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(width, height) <= 0 or not isfinite(fps) or fps <= 0:
            raise ValueError("The video needs valid dimensions and a positive nominal frame rate.")
        for config in (red_config, lane_config):
            if config is not None and ((config.width, config.height) != (width, height)
                                       or abs(config.fps - fps) > .001 or config.video_id != video_id):
                raise ValueError("Scene configuration does not match this video.")
        if red_config is not None:
            red_config.validate()
        thresholds = red_thresholds or Thresholds()
        red = RedLight(red_config, thresholds, fps) if red_config is not None else None
        lane = LaneChange(lane_config, lane_direction) if lane_config is not None else None
        speed = (Overspeed(speed_options, dataset_config, str(path), (width, height))
                 if speed_options is not None else None)

        # Both geometry workflows already share this detector/tracker profile.
        # A speed-only run retains the existing speed confidence profile.
        if red or lane:
            confidence, high, low, new = .15, .35, .14, .50
        else:
            confidence = speed_options.get("confidence", settings["detection"]["confidence_threshold"])
            high, low, new = confidence, min(.1, confidence), confidence
        progress(0, "Loading YOLOv11 and ByteTrack")
        if detector is None:
            model = root / "models" / settings["detection"].get("model", "yolo11n.pt")
            detector = VehicleDetector(model_path=str(model), confidence_threshold=confidence)
        tracker = tracker_factory(track_high_thresh=high, track_low_thresh=low, new_track_thresh=new)
        if tracker.name != "ByteTrack":
            raise RuntimeError("ByteTrack is unavailable. Check the tracking dependencies.")
        directory.mkdir(parents=True)
        anpr_options = anpr_profile(settings, dataset_config, video_id)
        collector = EventCollector(session_id, video_id, directory, verifier, anpr_options, root)
        collector.scenes = {engine.kind: engine for engine in (red, lane) if engine is not None}
        output = directory / "annotated.mp4"
        command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                   "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(output)]
        start = perf_counter()
        with (directory / "encoder.log").open("wb") as log, subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=log,
        ) as encoder, (directory / "tracks.csv").open("w", encoding="utf-8", newline="") as tracks:
            try:
                frames, detections, invalid, elapsed = process_frames(
                    capture, encoder.stdin, tracks, detector, tracker, collector,
                    speed, red, lane, fps, expected, (height, width), progress,
                )
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise OSError(f"Annotated video encoding failed. See {directory / 'encoder.log'}.")
        if not frames or (expected > 0 and frames != expected):
            raise ValueError(f"Decoded {frames} of {expected} expected frames.")
        wall_time = perf_counter() - start
        summary = dict(
            schema_version=1, system="A", session_id=session_id, video_id=video_id,
            video_name=video_name or path.name, frames=frames, detections=detections,
            unique_tracks=len(collector.contexts), invalid_timestamps=invalid,
            processing_fps=frames / wall_time, average_frame_latency_ms=elapsed * 1000 / frames,
            nominal_fps=fps, candidate_counts=dict(Counter(row["violation_type"] for row in collector.candidates)),
            confirmed_counts=dict(Counter(row["violation_type"] for row in collector.events)),
            annotated_video=str(output), tracks_csv=str(directory / "tracks.csv"),
            calibration=speed.status if speed else None, speed_evaluated=speed.available if speed else None,
            configuration={
                "detector": {"confidence": confidence, "model": settings["detection"].get("model", "yolo11n.pt")},
                "tracker": {"high": high, "low": low, "new": new},
                "speed": ({**speed_options, "speed_limit": speed.motion.speed_limit,
                           "smoothing_window": speed.motion.smoothing_window,
                           "violation_duration": speed.motion.violation_duration} if speed else None),
                "red_light": asdict(red_config) if red else None,
                "red_thresholds": asdict(thresholds) if red else None,
                "lane_change": asdict(lane_config) if lane else None,
                "lane_direction": lane_direction,
                "anpr": anpr_options,
            },
            evaluation={"precision": None, "recall": None, "f1": None, "false_positive_rate": None},
        )
        save_session(output_root / DATABASE, summary, collector.candidates, collector.events)
        progress(1, "Processing complete")
        return summary
    finally:
        capture.release()


def process_frames(capture, output, tracks, detector, tracker, collector, speed, red, lane,
                   fps, expected, shape, progress):
    writer = csv.writer(tracks)
    writer.writerow(["session_id", "frame_number", "timestamp", "track_id", "vehicle_class",
                     "confidence", "bbox", "bottom_center", "speed_kmh", "lane_id",
                     "signal_state", "stop_line_distance", "timing_valid", "track_age_frames", "missing_frames"])
    index = total = invalid = 0
    elapsed = 0.0
    previous_pts = None
    previous_image = None
    while True:
        start = perf_counter()
        ok, image = capture.read()
        if not ok:
            break
        if image.shape[:2] != shape:
            raise ValueError(f"Frame {index} changed resolution.")
        pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        valid = isfinite(pts) and pts >= 0 and (previous_pts is None or pts > previous_pts)
        nominal = index / fps
        if lane and index and pts > 0 and abs(pts - nominal) > .55 / fps:
            raise ValueError("Lane-change analysis requires constant frame rate video.")
        if valid:
            previous_pts = pts
        else:
            invalid += 1
        timestamp = pts if valid else nominal
        detections = detector.detect(image)
        total += len(detections)
        calibration = speed.calibration if speed and valid else None
        active = tracker.update(detections, image, timestamp, calibration)
        collector.frames = {index: image}
        if previous_image is not None:
            collector.frames[index - 1] = previous_image
        collector.observe_anpr(image, index, active)
        speeds = speed.prepare(active, valid) if speed else {}
        if red:
            red.prepare(image, timestamp)
        if lane:
            lane.prepare(active)
        canvas = image.copy()
        for scene in collector.scenes.values():
            scene.draw(canvas)
        confidences = {detection.bbox: float(detection.confidence) for detection in detections}
        for tid, track in active.items():
            context = collector.contexts.setdefault(tid, VehicleContext(track, first_frame=index))
            observation = VehicleObservation(
                collector.session_id, index, timestamp, tid, track.class_name,
                confidences.get(track.latest_bbox, 0.0), track.latest_bbox,
                track.latest_bottom_center_image, speeds.get(tid), timing_valid=valid,
                track_age_frames=index - context.first_frame + 1,
                missing_frames=(index - context.observations[-1].frame_number - 1
                                if context.observations else 0),
            )
            context.observations.append(observation)
            if red:
                red.observe(observation, collector.submit, collector.snapshot)
            if lane:
                lane.observe(observation, collector.submit, collector.snapshot)
            if speed:
                speed.observe(observation, track, collector.submit)
            text = observation.vehicle_class
            if observation.estimated_speed is not None:
                text += f" | {speed.speed_label}{observation.estimated_speed:.1f} km/h"
            if observation.lane_id is not None:
                text += f" | Lane {observation.lane_id}"
            confirmed = [VIOLATION_LABELS[kind] for kind, started in
                         collector.active_violations.get(tid, {}).items()
                         if index - started <= collector.display_seconds * fps]
            if confirmed:
                text += " | " + ", ".join(confirmed)
            draw_track(canvas, tid, observation.bbox, text, VIOLATION_COLOR if confirmed else NORMAL_COLOR)
            collector.draw_track_plates(canvas, index, tid, observation.bbox)
            writer.writerow([
                collector.session_id, index, timestamp, tid, observation.vehicle_class,
                observation.confidence, json.dumps(observation.bbox), json.dumps(observation.bottom_center),
                observation.estimated_speed, observation.lane_id,
                observation.signal_context["state"] if observation.signal_context else None,
                observation.stop_line_distance, valid, observation.track_age_frames, observation.missing_frames,
            ])
        if red:
            red.expire(index, collector.contexts, collector.submit, collector.snapshot)
        if lane:
            lane.expire(index, collector.contexts, collector.submit, collector.snapshot)
            for tid, kind in list(collector.snapshots):
                if kind == lane.kind and index - lane.engine.tracks[tid].last.frame > lane.engine.settings.max_gap:
                    collector.snapshot(tid, kind)
        output.write(canvas.tobytes())
        previous_image = image
        index += 1
        elapsed += perf_counter() - start
        if index == 1 or index % 30 == 0:
            progress(min(.99, index / max(expected, index)), f"Processed {index:,} frames")
    if index:
        if red:
            red.expire(index - 1, collector.contexts, collector.submit, collector.snapshot, final=True)
        if lane:
            lane.expire(index - 1, collector.contexts, collector.submit, collector.snapshot, final=True)
    return index, total, invalid, elapsed
