from collections import Counter, deque
from dataclasses import dataclass, field
import logging
from math import ceil, floor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .plate_detector import PlateDetector
from .plate_recognizer import PlateRecognizer
from .plate_validator import validate_registration


logger = logging.getLogger(__name__)


@dataclass
class ANPRResult:
    status: str
    vehicle_number: str = ""
    plate_confidence: Optional[float] = None  # OCR confidence
    raw_text: str = ""
    vehicle_crop: Optional[np.ndarray] = field(default=None, repr=False)
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)
    frame_number: Optional[int] = None
    plate_bbox: Optional[tuple] = None
    detection_confidence: Optional[float] = None
    vehicle_bbox: Optional[tuple] = None
    candidate: str = ""
    candidate_format: str = ""
    candidate_frames: tuple = ()
    character_agreement: Optional[float] = None
    engine_results: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class VehicleObservation:
    frame_number: int
    crop: np.ndarray
    inner_bbox: tuple[int, int, int, int]

    @property
    def area(self):
        height, width = self.crop.shape[:2]
        return height * width


@dataclass
class TrackObservations:
    crops: deque
    last_sample: int
    attempts: int = 0
    result: Optional[ANPRResult] = None


class ANPRProcessor:
    def __init__(self, config, project_root: Path):
        self.padding = max(0.0, float(config.get("crop_padding", 0.08)))
        self.sample_interval = max(1, int(config.get("sample_interval_frames", 5)))
        self.buffer_size = max(1, int(config.get("buffer_size", 12)))
        self.max_attempts = max(1, int(config.get("max_attempts_per_track", 8)))
        self.ocr_threshold = float(config.get("ocr_confidence", 0.45))
        self.single_threshold = float(config.get("single_read_confidence", 0.85))
        self.min_width = max(1, int(config.get("min_plate_width", 28)))
        self.min_height = max(1, int(config.get("min_plate_height", 9)))
        self.detector = PlateDetector(
            project_root / config.get("detector_weights", "models/anpr/yolo_best.pt"),
            float(config.get("detection_confidence", 0.35)),
            max(1, int(config.get("max_plates_per_crop", 2))),
            int(config.get("detector_image_size", 640)),
        )
        self.scan = None
        if config.get("scan_all_tracks", False) or config.get("event_scan", False):
            from .ocr_session import OCRSession
            from .plate_scan import PlateScan

            self.recognizer = OCRSession(config, project_root)
            self.scan = PlateScan(self.detector, self.recognizer, config)
        else:
            self.recognizer = PlateRecognizer(project_root / config.get("ocr_model_dir", "models/anpr"))
        self.tracks = {}
        self.active_ids = set()

    def observe(self, frame, frame_number, active_tracks):
        if self.scan is not None:
            self.scan.observe(frame, frame_number, active_tracks)
            return
        active_ids = set(active_tracks)
        for track_id in self.active_ids - active_ids:
            self.tracks[track_id].crops.clear()
        self.active_ids = active_ids

        height, width = frame.shape[:2]
        for track_id, track in active_tracks.items():
            observations = self.tracks.get(track_id)
            if observations is None:
                observations = TrackObservations(deque(maxlen=self.buffer_size), frame_number - self.sample_interval)
                self.tracks[track_id] = observations
            if observations.attempts >= self.max_attempts or track.latest_bbox is None:
                continue
            if observations.result and observations.result.vehicle_number:
                continue
            if frame_number - observations.last_sample < self.sample_interval:
                continue
            left, top, right, bottom = track.latest_bbox
            pad_x, pad_y = (right - left) * self.padding, (bottom - top) * self.padding
            x1, y1 = max(0, floor(left - pad_x)), max(0, floor(top - pad_y))
            x2, y2 = min(width, ceil(right + pad_x)), min(height, ceil(bottom + pad_y))
            if x1 >= x2 or y1 >= y2:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            # Use native crop area as a cheap proxy for available plate detail.
            observations.crops.append(VehicleObservation(
                frame_number, crop,
                (floor(left) - x1, floor(top) - y1, ceil(right) - x1, ceil(bottom) - y1),
            ))
            observations.last_sample = frame_number

    def recognize_track(self, track_id):
        if self.scan is not None:
            return self.scan.result(track_id)
        observations = self.tracks.get(track_id)
        if observations is None:
            return ANPRResult("Not Detected")
        if observations.result and observations.result.vehicle_number:
            return observations.result

        remaining = self.max_attempts - observations.attempts
        crops = sorted(observations.crops, key=lambda item: (item.area, item.frame_number), reverse=True)
        observations.crops.clear()
        votes = []
        fallback = observations.result or ANPRResult("Not Detected")
        for observation in crops[:remaining]:
            observations.attempts += 1
            try:
                candidate = self._read_crop(observation)
            except Exception:
                logger.exception("ANPR inference failed for track %s", track_id)
                return ANPRResult("Processing Failed", vehicle_crop=observation.crop,
                                  frame_number=observation.frame_number)
            candidate.frame_number = observation.frame_number
            if candidate.vehicle_number:
                votes.append(candidate)
            fallback = max((candidate, fallback), key=self._diagnostic_rank)

        counts = Counter(read.vehicle_number for read in votes)
        if counts:
            number, support = counts.most_common(1)[0]
            best = max((read for read in votes if read.vehicle_number == number),
                       key=lambda read: read.plate_confidence)
            tied = sum(count == support for count in counts.values()) > 1
            if not tied and (support > 1 or best.plate_confidence >= self.single_threshold):
                observations.result = best
                return best
            fallback = ANPRResult("Low Confidence", plate_confidence=best.plate_confidence,
                                  raw_text=best.raw_text, vehicle_crop=best.vehicle_crop,
                                  plate_crop=best.plate_crop, frame_number=best.frame_number)

        # Keep the outcome between events without retaining diagnostic images for departed tracks.
        observations.result = ANPRResult(fallback.status, plate_confidence=fallback.plate_confidence,
                                         raw_text=fallback.raw_text)
        return fallback

    @staticmethod
    def _diagnostic_rank(result):
        return (result.status == "Low Confidence", result.plate_crop is not None,
                result.vehicle_crop is not None, result.plate_confidence or 0)

    def _read_crop(self, observation):
        vehicle = observation.crop
        left, top, right, bottom = observation.inner_bbox
        best = ANPRResult("Not Detected", vehicle_crop=vehicle)
        candidates = []
        for detection in sorted(self.detector.detect(vehicle), key=lambda item: item.confidence, reverse=True):
            x1, y1, x2, y2 = detection.bbox
            # Reject nearby plates whose centers fall only in the crop padding.
            if not (left <= (x1 + x2) / 2 < right and top <= (y1 + y2) / 2 < bottom):
                continue
            plate = vehicle[y1:y2, x1:x2].copy()
            if best.plate_crop is None:
                best = ANPRResult("Not Readable", vehicle_crop=vehicle, plate_crop=plate)
            if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
                continue
            for reading in self.recognizer.recognize(plate):
                number = validate_registration(reading.raw_text)
                if number and self.ocr_threshold <= reading.confidence <= 1:
                    candidates.append(ANPRResult(
                        "Recognized", number, reading.confidence, reading.raw_text, vehicle, plate,
                    ))
                elif reading.raw_text.strip():
                    diagnostic = ANPRResult(
                        "Low Confidence" if number else "Not Readable",
                        plate_confidence=reading.confidence, raw_text=reading.raw_text,
                        vehicle_crop=vehicle, plate_crop=plate,
                    )
                    best = max((diagnostic, best), key=self._diagnostic_rank)
        return max(candidates, key=lambda read: read.plate_confidence) if candidates else best
