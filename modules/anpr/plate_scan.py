"""Sample native vehicle crops and read the strongest plate observations per track."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from hashlib import blake2b
import json
import logging
from math import ceil, floor, sqrt

import cv2
import numpy as np

from .anpr_processor import ANPRResult
from .ocr_candidates import candidate_text, engine_report, rank_candidates
from .plate_enhancement import enhance_plate
from .plate_validator import validate_registration


logger = logging.getLogger(__name__)


@dataclass
class PlateObservation:
    track_id: int
    frame: int
    bbox: tuple
    vehicle_bbox: tuple
    confidence: float
    crop: np.ndarray = field(repr=False)
    sharpness: float
    rank: float
    status: str = "Plate Detected"
    raw_text: str = ""
    ocr_confidence: float | None = None
    text_agreement: bool = False
    variants: dict = field(default_factory=dict, repr=False)
    engines: dict = field(default_factory=dict, repr=False)
    error: str = ""
    readings: dict = field(default_factory=dict, repr=False)


def draw_plates(image, plates):
    green = (0, 255, 0)
    thickness = max(2, round(image.shape[1] / 1100))
    scale = max(0.45, image.shape[1] / 4000)
    for plate in plates:
        x1, y1, x2, y2 = plate.bbox
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), green, thickness)
        label = f"#{plate.track_id} plate {plate.confidence:.0%}"
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y1 - height - baseline - 6)
        cv2.rectangle(image, (x1, top), (x1 + width + 6, top + height + baseline + 6), (0, 0, 0), -1)
        cv2.putText(image, label, (x1 + 3, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, green, 1, cv2.LINE_AA)


class PlateScan:
    def __init__(self, detector, recognizer, config):
        self.detector = detector
        self.recognizer = recognizer
        self.interval = max(1, int(config.get("sample_interval_frames", 8)))
        self.padding = max(0.0, float(config.get("crop_padding", 0.08)))
        self.best_count = max(1, int(config.get("ocr_frames_per_track", 5)))
        self.min_width = max(1, int(config.get("min_plate_width", 20)))
        self.min_height = max(1, int(config.get("min_plate_height", 7)))
        self.ocr_threshold = float(config.get("ocr_confidence", 0.45))
        self.primary = config.get("ocr_engine", "easyocr")
        self.streaming = config.get("event_scan", False)
        self.single_threshold = float(config.get("single_read_confidence", .85))
        self.generations = {}
        self.result_generations = {}
        self.history = max(self.best_count, int(config.get("buffer_size", 12)))
        self.last_sample = {}
        self.fingerprints = defaultdict(lambda: deque(maxlen=self.history))
        self.plate_fingerprints = defaultdict(lambda: deque(maxlen=self.history))
        self.records = []
        self.by_track = defaultdict(list)
        self.rolling = defaultdict(lambda: deque(maxlen=self.history))
        self.by_frame = defaultdict(list)
        self.attempts = 0
        self.hit_attempts = 0
        self.results = {}
        self.errors = {}

    def observe(self, frame, frame_number, tracks):
        if self.streaming:
            cutoff = frame_number - max(60, self.interval * self.history)
            while self.by_frame and next(iter(self.by_frame)) < cutoff:
                self.by_frame.pop(next(iter(self.by_frame)))
        boxes = {tid: track.latest_bbox for tid, track in tracks.items() if track.latest_bbox is not None}
        for tid in boxes:
            self.sample_track(frame, frame_number, tid, boxes)

    def sample_track(self, frame, frame_number, tid, boxes, *, force=False):
        last = self.last_sample.get(tid, frame_number - self.interval)
        if tid not in boxes or last == frame_number or (not force and frame_number - last < self.interval):
            return
        self.last_sample[tid] = frame_number
        try:
            self._sample(frame, frame_number, tid, boxes[tid], boxes)
        except Exception as error:
            self.errors[tid] = f"{type(error).__name__}: {error}"
            logger.exception("Plate detection failed for track %s frame %s", tid, frame_number)

    def _sample(self, frame, frame_number, tid, box, boxes):
        left, top, right, bottom = box
        h, w = frame.shape[:2]
        width, height = right - left, bottom - top
        if width < 24 or height < 20:
            return
        pad_x, pad_y = width * self.padding, height * self.padding
        x1, y1 = max(0, floor(left - pad_x)), max(0, floor(top - pad_y))
        x2, y2 = min(w, ceil(right + pad_x)), min(h, ceil(bottom + pad_y))
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        fingerprint = (crop.shape, blake2b(crop.tobytes(), digest_size=12).digest())
        if not self.streaming and fingerprint in self.fingerprints[tid]:
            return
        self.fingerprints[tid].append(fingerprint)
        self.attempts += 1
        hits = []
        for detection in self.detector.detect(crop):
            a, b, c, d = detection.bbox
            pw, ph = c - a, d - b
            cx, cy = x1 + (a + c) / 2, y1 + (b + d) / 2
            if not (left <= cx < right and top + 0.48 * height <= cy < bottom):
                continue
            if not (10 <= pw <= 0.8 * width and 4 <= ph <= 0.28 * height and 1.55 <= pw / ph <= 7.5):
                continue
            if any(other != tid and l <= cx < r and t <= cy < bt for other, (l, t, r, bt) in boxes.items()):
                continue
            if self.streaming:
                # Translate the padded crop coordinates and clip to the actual tracked vehicle.
                a, b = max(a, ceil(left) - x1), max(b, ceil(top) - y1)
                c, d = min(c, floor(right) - x1), min(d, floor(bottom) - y1)
                if c <= a or d <= b:
                    continue
                pw, ph = c - a, d - b
            plate = crop[b:d, a:c].copy()
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            rank = sqrt(pw * ph) * detection.confidence * (0.5 + 0.5 * min(sharpness / 300, 1))
            observation = PlateObservation(
                tid, frame_number, (x1 + a, y1 + b, x1 + c, y1 + d), (x1, y1, x2, y2),
                detection.confidence, plate, sharpness, rank,
            )
            hits.append(observation)
        if not hits:
            return
        self.hit_attempts += 1
        for plate in hits:
            if not self.streaming:
                self.records.append(plate)
            track_plates = self.by_track[tid]
            track_plates.append(plate)
            if self.streaming:
                del track_plates[:-self.history]
            self.by_frame[frame_number].append(plate)
        eligible = [p for p in hits if p.crop.shape[1] >= self.min_width and p.crop.shape[0] >= self.min_height]
        if eligible:
            best = max(eligible, key=lambda plate: plate.rank)
            key = (best.crop.shape, blake2b(best.crop.tobytes(), digest_size=12).digest())
            if key not in self.plate_fingerprints[tid]:
                self.plate_fingerprints[tid].append(key)
                finalists = self.rolling[tid]
                finalists.append(best)
                self.generations[tid] = frame_number

    def result(self, track_id):
        generation = self.generations.get(track_id)
        cached = self.results.get(track_id)
        if cached is not None and (not self.streaming or cached.vehicle_number
                                   or self.result_generations.get(track_id) == generation):
            if cached.plate_bbox is not None or not self.by_track.get(track_id):
                return cached
        self.result_generations[track_id] = generation
        plates = self.by_track.get(track_id, ())
        if not plates:
            error = self.errors.get(track_id, "")
            result = ANPRResult("Processing Failed" if error else "Plate Not Detected", error=error)
            self.results[track_id] = result
            return result
        candidates = sorted(self.rolling.get(track_id, ()), key=lambda plate: plate.rank, reverse=True)[:self.best_count]
        observed = []
        verified_reads = []
        for plate in candidates:
            plate.status = "OCR Processing"
            try:
                if not self.streaming or not plate.readings:
                    plate.variants, engines = self.recognizer.scan(plate.crop)
                    if self.streaming:
                        plate.readings = engines
                else:
                    engines = plate.readings
            except Exception as error:
                plate.error = f"{type(error).__name__}: {error}"
                plate.status = "Processing Failed"
                logger.exception("OCR failed for track %s frame %s", track_id, plate.frame)
                continue
            plate.engines = {name: engine_report(engine) for name, engine in engines.items()}
            if all(engine.error for engine in engines.values()):
                plate.error = "; ".join(f"{name}: {engine.error}" for name, engine in engines.items())
                plate.status = "Processing Failed"
                continue
            plate.status = "Text Unreadable"
            for name in dict.fromkeys((self.primary, "easyocr", "ppocrv4")):
                if name not in engines:
                    continue
                usable = [
                    read for reads in engines[name].readings.values() for read in reads
                    if read.raw_text.strip() and 0 <= read.confidence <= 1
                ]
                if not usable:
                    continue
                best = max(usable, key=lambda read: read.confidence)
                if not plate.raw_text:
                    plate.raw_text, plate.ocr_confidence = best.raw_text, best.confidence
                eligible = [(candidate_text(read.raw_text), plate, read) for read in usable
                            if read.confidence >= self.ocr_threshold]
                eligible = [entry for entry in eligible if entry[0]]
                if eligible:
                    observed.extend(eligible)
                    if self.streaming:
                        valid = [(validate_registration(read.raw_text), plate, read) for _, _, read in eligible]
                        valid = [entry for entry in valid if entry[0]]
                        if valid:
                            verified_reads.append(max(valid, key=lambda entry: entry[2].confidence))
                    plate.status = "Candidate Extracted — Unverified"
                    break
        ranked = rank_candidates(observed)
        selected = max(plates, key=lambda plate: (bool(plate.variants), plate.rank))
        candidate, agreement, frames = "", None, ()
        if ranked:
            consistent, _, agreement, candidate, matches = ranked
            selected, read = max(matches.values(), key=lambda pair: pair[0].rank * pair[1].confidence)
            selected.raw_text, selected.ocr_confidence = read.raw_text, read.confidence
            frames = tuple(sorted(matches))
            if consistent:
                for plate, _ in matches.values():
                    plate.status = "Consistent Candidate — Unverified"
                    plate.text_agreement = True
        number = ""
        if self.streaming and verified_reads:
            votes = Counter(text for text, _, _ in verified_reads)
            winner, support = votes.most_common(1)[0]
            if list(votes.values()).count(support) == 1:
                _, best_plate, best_read = max(
                    (entry for entry in verified_reads if entry[0] == winner),
                    key=lambda entry: entry[2].confidence,
                )
                if support > 1 or best_read.confidence >= self.single_threshold:
                    number, selected = winner, best_plate
                    selected.raw_text, selected.ocr_confidence = best_read.raw_text, best_read.confidence
                    selected.status = "Recognized"
        status = selected.status if selected.status != "Plate Detected" else "Text Unreadable"
        result = ANPRResult(
            status, vehicle_number=number, plate_confidence=selected.ocr_confidence, raw_text=selected.raw_text,
            plate_crop=selected.crop, frame_number=selected.frame,
            plate_bbox=selected.bbox, detection_confidence=selected.confidence,
            vehicle_bbox=selected.vehicle_bbox, candidate=candidate,
            candidate_format=validate_registration(candidate), candidate_frames=frames,
            character_agreement=agreement, engine_results=selected.engines, error=selected.error,
        )
        self.results[track_id] = result
        return result

    def summary(self):
        return {
            "attempts": self.attempts,
            "attempts_with_detections": self.hit_attempts,
            "detected_plates": len(self.records),
            "tracks_with_detections": len(self.by_track),
            "ocr_attempted": sum(plate.status != "Plate Detected" for plate in self.records),
            "recognized_plates": sum(plate.status == "Registration Verified" for plate in self.records),
            "recognized_tracks": sum(bool(result.vehicle_number) for result in self.results.values()),
            "unreadable_plates": sum(plate.status == "Text Unreadable" for plate in self.records),
            "unattempted_plates": sum(plate.status == "Plate Detected" for plate in self.records),
            "ocr_errors": sum(bool(plate.error) for plate in self.records),
            "tracks_with_detection_errors": len(self.errors),
            "candidate_tracks": sum(bool(result.candidate) for result in self.results.values()),
            "consistent_tracks": sum(result.status == "Consistent Candidate — Unverified" for result in self.results.values()),
        }

    @staticmethod
    def save_png(path, image):
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"Could not encode {path}")
        path.write_bytes(data.tobytes())

    def export(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, plate in enumerate(self.records):
            name = f"track{plate.track_id}_frame{plate.frame}_{index}"
            path = directory / f"{name}.png"
            self.save_png(path, plate.crop)
            enhanced = plate.variants.get("enhanced")
            if enhanced is None:
                enhanced = enhance_plate(plate.crop)
            enhanced_path = directory / f"{name}_enhanced.png"
            self.save_png(enhanced_path, enhanced)
            ocr_inputs = {}
            for variant, image in plate.variants.items():
                if variant == "enhanced":
                    ocr_inputs[variant] = str(enhanced_path)
                    continue
                variant_path = directory / f"{name}_{variant}.png"
                self.save_png(variant_path, image)
                ocr_inputs[variant] = str(variant_path)
            rows.append({
                "track_id": plate.track_id, "frame": plate.frame, "bbox": plate.bbox,
                "size": plate.crop.shape[1::-1], "detection_confidence": plate.confidence,
                "sharpness": plate.sharpness, "status": plate.status, "raw_text": plate.raw_text,
                "ocr_confidence": plate.ocr_confidence, "registration": "",
                "text_agreement": plate.text_agreement, "error": plate.error,
                "crop_path": str(path), "enhanced_crop_path": str(enhanced_path), "ocr_inputs": ocr_inputs,
                "engines": plate.engines,
            })
        manifest = directory / "plates.json"
        manifest.write_text(json.dumps({
            "summary": self.summary(), "plates": rows, "detection_errors": self.errors,
            "tracks": {tid: result.status for tid, result in self.results.items()},
            "candidates": {tid: {"text": result.candidate, "format_valid": bool(result.candidate_format),
                                 "frames": result.candidate_frames, "agreement": result.character_agreement,
                                 "status": result.status} for tid, result in self.results.items()},
            "ocr_initialization_errors": self.recognizer.initialization_errors,
        }, indent=2), encoding="utf-8")
        return str(manifest), rows
