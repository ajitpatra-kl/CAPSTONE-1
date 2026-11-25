"""
modules/speed_detection/report_generator.py

Generates CSV reports for all detected vehicles and confirmed violations.
"""

import csv
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from .violation_detector import ViolationRecord

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates CSV and JSON reports from processing results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_violations_report(
        self,
        violations: List[ViolationRecord],
        video_name: str,
        speed_limit_kmh: float,
        analytics: Dict[str, Any],
        benchmark: Optional[Dict] = None,
    ) -> str:
        """
        Generate a comprehensive CSV report.
        
        Returns path to the CSV file.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = Path(video_name).stem if video_name else "unknown"
        fname = f"report_{stem}_{ts}.csv"
        out_path = self.output_dir / fname

        # Build rows
        rows = []
        for v in violations:
            rows.append({
                "Track ID": v.track_id,
                "Vehicle Class": v.class_name.capitalize(),
                "Estimated Speed (km/h)": round(v.speed_kmh, 2),
                "Speed Limit (km/h)": v.speed_limit_kmh,
                "Excess Speed (km/h)": round(v.excess_kmh, 2),
                "Timestamp (sec)": round(v.timestamp_sec, 3),
                "Frame Number": v.frame_number,
                "Calibrated": "Yes" if v.is_calibrated else "No",
                "Evidence Image": os.path.basename(v.evidence_path) if v.evidence_path else "",
                "Original Frame": os.path.basename(v.original_frame_path) if v.original_frame_path else "",
                "Status": "Confirmed",
                "Registration Number": v.vehicle_number,
                "Plate Confidence": round(v.plate_confidence, 4) if v.plate_confidence is not None else None,
                "ANPR Status": v.anpr_status,
                "Plate Detector Confidence": round(v.plate_detection_confidence, 4) if v.plate_detection_confidence is not None else None,
                "Plate Frame": v.plate_frame_number,
                "Plate Box": v.plate_bbox,
                "Plate Evidence": os.path.basename(v.plate_evidence_path) if v.plate_evidence_path else "",
                "OCR Candidate (Unverified)": v.plate_candidate,
                "Indian Format Match": "Yes" if v.plate_candidate_format else ("No" if v.plate_candidate else ""),
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Track ID","Vehicle Class","Estimated Speed (km/h)",
                     "Speed Limit (km/h)","Excess Speed (km/h)","Timestamp (sec)",
                     "Frame Number","Calibrated","Evidence Image","Original Frame","Status",
                     "Registration Number","Plate Confidence","ANPR Status",
                     "Plate Detector Confidence","Plate Frame","Plate Box","Plate Evidence",
                     "OCR Candidate (Unverified)","Indian Format Match"]
        )

        # Header metadata
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write(f"# AI Traffic Violation Detection Report\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Video: {video_name}\n")
            f.write(f"# Speed Limit: {speed_limit_kmh} km/h\n")
            f.write(f"# Total Violations: {len(violations)}\n")
            f.write(f"# Tracker: {analytics.get('tracker', 'unavailable')}\n")
            f.write(f"# Total Unique Tracks: {analytics.get('unique_tracks', 0)}\n")
            for label, key in (("Average Speed (km/h)", "avg_speed_kmh"),
                               ("Max Speed (km/h)", "max_speed_kmh")):
                value = analytics.get(key)
                f.write(f"# {label}: {value:.2f}\n" if value is not None else f"# {label}: unavailable\n")
            if benchmark:
                for label, key in (("Reference Speed (km/h)", "reference_speed_kmh"),
                                   ("Estimated Speed (km/h)", "estimated_speed_kmh"),
                                   ("Absolute Error (km/h)", "abs_error_kmh"),
                                   ("Percentage Error (%)", "pct_error")):
                    value = benchmark.get(key)
                    f.write(f"# {label}: {value:.2f}\n" if value is not None else f"# {label}: unavailable\n")
            f.write("#\n")
            df.to_csv(f, index=False)

        logger.info(f"Report saved: {out_path}")
        return str(out_path)

    def generate_all_tracks_report(
        self,
        tracks: Dict[int, Any],
        speed_limit_kmh: float,
        video_name: str,
    ) -> str:
        """Generate a CSV with data for every tracked vehicle (not just violations)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = Path(video_name).stem if video_name else "unknown"
        fname = f"all_tracks_{stem}_{ts}.csv"
        out_path = self.output_dir / fname

        rows = []
        for tid, state in tracks.items():
            speeds = list(state.speed_samples_kmh)
            avg_spd = sum(speeds) / len(speeds) if speeds else 0
            max_spd = max(speeds) if speeds else 0
            rows.append({
                "Track ID": tid,
                "Vehicle Class": state.class_name.capitalize(),
                "Total Frames Tracked": state.hit_count,
                "Average Speed (km/h)": round(avg_spd, 2) if speeds else "",
                "Max Speed (km/h)": round(max_spd, 2) if speeds else "",
                "Speed Limit (km/h)": speed_limit_kmh,
                "Violation Confirmed": "Yes" if state.violation_confirmed else "No",
            })

        df = pd.DataFrame(rows, columns=["Track ID", "Vehicle Class", "Total Frames Tracked",
                                        "Average Speed (km/h)", "Max Speed (km/h)",
                                        "Speed Limit (km/h)", "Violation Confirmed"])
        df.to_csv(out_path, index=False)
        logger.info(f"All-tracks report saved: {out_path}")
        return str(out_path)
