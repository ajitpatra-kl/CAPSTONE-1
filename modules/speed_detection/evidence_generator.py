"""
modules/speed_detection/evidence_generator.py

Generates annotated evidence images for each confirmed overspeed violation.
Saves frames with:
 - Vehicle bounding box (red border)
 - Track ID, class, speed, speed limit
 - Violation timestamp
 - Overspeed status banner
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from uuid import uuid4
from typing import Optional, Tuple

from .violation_detector import ViolationRecord

logger = logging.getLogger(__name__)


class EvidenceGenerator:
    """Generates and saves visual evidence for overspeed violations."""

    # Colors (BGR)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 220)
    GREEN = (0, 200, 0)
    ORANGE = (0, 140, 255)

    def __init__(self, output_dir: str, image_format: str = "jpg"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format.lower().lstrip(".")
        self._evidence_count = 0

    def generate(
        self,
        frame: np.ndarray,
        violation: ViolationRecord,
        video_name: str = "video",
    ) -> Optional[str]:
        """
        Generate and save an evidence image for a violation.
        
        Args:
            frame: Original BGR video frame at violation time
            violation: ViolationRecord with all violation details
            video_name: Source video filename (for naming evidence)
        
        Returns:
            Path to the saved evidence image, or None if failed
        """
        try:
            annotated = self._annotate_frame(frame.copy(), violation)
            
            fname = f"frame{violation.frame_number}_{uuid4().hex[:10]}.{self.image_format}"
            out_path = self.output_dir / fname
            
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.image_format == "jpg" else []
            original_path = out_path.with_name(f"{out_path.stem}_original.png")
            for path, image in ((out_path, annotated), (original_path, frame)):
                encoded, data = cv2.imencode(path.suffix, image, params if path == out_path else [])
                if not encoded:
                    raise OSError(f"Failed to encode evidence for {path}")
                path.write_bytes(data.tobytes())

            violation.original_frame_path = str(original_path)
            self._evidence_count += 1
            logger.info(f"Evidence saved: {out_path}")
            return str(out_path)
        
        except Exception as e:
            logger.error(f"Failed to generate evidence for track #{violation.track_id}: {e}")
            return None

    def save_anpr_crops(self, violation: ViolationRecord, result, *, enhanced=None) -> None:
        """Save the original crops from the selected ANPR observation."""
        evidence_path = Path(violation.evidence_path)
        plate_stem = f"{evidence_path.stem}_track{violation.track_id}"
        for field, crop, suffix in (
            ("vehicle_crop_path", result.vehicle_crop, "vehicle"),
            ("plate_crop_path", result.plate_crop, "plate"),
        ):
            if crop is None:
                continue
            stem = plate_stem if suffix == "plate" else evidence_path.stem
            path = evidence_path.with_name(f"{stem}_{suffix}_f{result.frame_number}.png")
            try:
                encoded, data = cv2.imencode(".png", crop)
                if not encoded:
                    raise OSError(f"Could not encode {suffix} crop")
                path.write_bytes(data.tobytes())
                setattr(violation, field, str(path))
            except (OSError, cv2.error):
                logger.exception("Could not save %s crop for track %s", suffix, violation.track_id)
        if result.plate_crop is not None:
            from modules.anpr.plate_enhancement import enhance_plate

            path = evidence_path.with_name(f"{plate_stem}_plate_enhanced_f{result.frame_number}.png")
            try:
                image = enhanced if enhanced is not None else enhance_plate(result.plate_crop)
                ok, data = cv2.imencode(".png", image)
                if not ok:
                    raise OSError(f"Could not encode {path}")
                path.write_bytes(data.tobytes())
                violation.plate_enhanced_path = str(path)
            except (OSError, cv2.error):
                logger.exception("Could not save enhanced crop for track %s", violation.track_id)

    def finalize_plate_evidence(self, violation, result, scan, capture):
        from modules.anpr.plate_scan import draw_plates

        if not violation.evidence_path or not violation.original_frame_path:
            return
        evidence = Path(violation.evidence_path)
        if result.plate_bbox is not None:
            plate = next(p for p in scan.by_frame[result.frame_number]
                         if p.track_id == violation.track_id and p.bbox == result.plate_bbox)
            capture.set(cv2.CAP_PROP_POS_FRAMES, result.frame_number)
            ok, source = capture.read()
            if ok:
                x1, y1, x2, y2 = result.vehicle_bbox
                result.vehicle_crop = source[y1:y2, x1:x2].copy()
                plates = [p for p in scan.by_frame[result.frame_number] if p.track_id == violation.track_id]
                draw_plates(source, plates)
                path = evidence.with_name(f"{evidence.stem}_plate_evidence_f{result.frame_number}.png")
                scan.save_png(path, source)
                violation.plate_evidence_path = str(path)
            self.save_anpr_crops(violation, result, enhanced=plate.variants.get("enhanced"))

        native = np.frombuffer(Path(violation.original_frame_path).read_bytes(), np.uint8)
        frame = cv2.imdecode(native, cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Could not decode original evidence for track {violation.track_id}")
        annotated = self._annotate_frame(frame, violation)
        draw_plates(annotated, scan.by_frame.get(violation.frame_number, ()))
        ok, data = cv2.imencode(evidence.suffix, annotated)
        if not ok:
            raise OSError(f"Could not update {evidence}")
        evidence.write_bytes(data.tobytes())

    def _annotate_frame(self, frame: np.ndarray, v: ViolationRecord) -> np.ndarray:
        """Draw violation annotations on the frame."""
        h, w = frame.shape[:2]
        
        # Darken the frame slightly for overlay contrast
        frame = cv2.addWeighted(frame, 0.85, np.zeros_like(frame), 0.15, 0)
        
        # Draw violation bounding box
        if v.bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            # Thick red box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.RED, 4)
            # Corner accent marks
            corner_len = 20
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), self.RED, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), self.RED, 3)
        
        # Top banner background
        banner_h = 90
        banner = np.full((banner_h, w, 3), (30, 10, 10), dtype=np.uint8)
        frame[:banner_h] = cv2.addWeighted(frame[:banner_h], 0.4, banner, 0.6, 0)
        
        # Banner text
        cv2.putText(frame, "OVERSPEED VIOLATION", (10, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.RED, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Track #{v.track_id}  |  {v.class_name.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Speed: {v.speed_kmh:.1f} km/h  |  Limit: {v.speed_limit_kmh:.0f} km/h  |  +{v.excess_kmh:.1f} km/h OVER",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.YELLOW, 1, cv2.LINE_AA)
        
        # Bottom info strip
        bottom_h = 68 if v.anpr_status or v.vehicle_number else 50
        bottom_start = h - bottom_h
        btm_bg = np.full((bottom_h, w, 3), (10, 10, 30), dtype=np.uint8)
        frame[bottom_start:] = cv2.addWeighted(frame[bottom_start:], 0.3, btm_bg, 0.7, 0)
        
        cv2.putText(frame, f"Frame: {v.frame_number}  |  Time: {v.timestamp_sec:.2f}s",
                    (10, bottom_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
        
        calib_note = "CALIBRATED" if v.is_calibrated else "UNCALIBRATED (approximate)"
        calib_col = self.GREEN if v.is_calibrated else self.ORANGE
        cv2.putText(frame, f"Speed measurement: {calib_note}",
                    (10, bottom_start + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_col, 1, cv2.LINE_AA)
        if v.anpr_status or v.vehicle_number:
            plate_label = v.vehicle_number or v.anpr_status
            if plate_label == "Text Unreadable":
                plate_label = "Plate Detected - Text Unreadable"
            cv2.putText(frame, f"Number Plate: {plate_label}",
                        (10, bottom_start + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
        
        # Speed label above/below bbox
        if v.bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            label = f"{v.speed_kmh:.1f} km/h"
            lx = max(0, x1)
            ly = max(banner_h + 5, y1 - 10)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (lx, ly - th - 5), (lx + tw + 6, ly + 4), self.RED, -1)
            cv2.putText(frame, label, (lx + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2, cv2.LINE_AA)
        
        return frame

    def annotate_live_frame(
        self,
        frame: np.ndarray,
        tracks: dict,
        speeds: dict,
        speed_limit: float,
        violation_detector,
        calibrated: bool = True,
        estimated: bool = False,
    ) -> np.ndarray:
        """
        Annotate a live processing frame with all track bboxes, IDs, and speeds.
        Used during video processing, NOT for evidence images.
        """
        out = frame.copy()
        
        for tid, state in tracks.items():
            if state.latest_bbox is None:
                continue
            
            x1, y1, x2, y2 = [int(c) for c in state.latest_bbox]
            speed = speeds.get(tid) if calibrated or estimated else None
            is_speeding = (speed is not None and speed > speed_limit)
            
            # Box color: red if speeding, green if ok, yellow if unknown speed
            if speed is None:
                color = (0, 180, 255)  # orange/yellow = no speed yet
            elif is_speeding:
                color = (0, 0, 220)   # red
            else:
                color = (0, 200, 0)   # green
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            
            # Build label
            speed_str = f"{speed:.1f}" if speed is not None else "--"
            label_top = f"#{tid} {state.class_name}"
            label_bot = f"{speed_str} km/h" if speed is not None else "Speed unavailable"
            if estimated and speed is not None:
                label_bot = f"Est. {label_bot}"
            if is_speeding:
                label_bot += " CONFIRMED" if violation_detector.is_event_confirmed(tid) else " ABOVE LIMIT"
            
            # Label backgrounds
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw1, th1), _ = cv2.getTextSize(label_top, font, 0.45, 1)
            (tw2, th2), _ = cv2.getTextSize(label_bot, font, 0.45, 1)
            
            lx = max(0, x1)
            ly = max(0, y1 - 1)
            
            # Top label: class + ID
            cv2.rectangle(out, (lx, ly - th1 - 20), (lx + max(tw1,tw2) + 6, ly - th1 + 4), (30, 30, 30), -1)
            cv2.putText(out, label_top, (lx + 3, ly - th1 - 8), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(out, label_bot, (lx + 3, ly - 3), font, 0.45, color, 1, cv2.LINE_AA)
            
            # Bottom-centre reference point (for homography)
            bc_x = int((x1 + x2) / 2)
            bc_y = y2
            cv2.circle(out, (bc_x, bc_y), 4, color, -1)
        
        # Speed limit overlay (top-right)
        limit_label = f"Limit: {speed_limit:.0f} km/h"
        (lw, lh), _ = cv2.getTextSize(limit_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        fw = frame.shape[1]
        cv2.rectangle(out, (fw - lw - 20, 5), (fw - 5, lh + 20), (10, 10, 60), -1)
        cv2.putText(out, limit_label, (fw - lw - 14, lh + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Calibration status
        if not calibrated:
            cv2.putText(out, "ESTIMATED SPEED" if estimated else "SPEED UNAVAILABLE", (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        
        return out

    @property
    def evidence_count(self) -> int:
        return self._evidence_count
