"""
modules/speed_detection/video_processor.py

Main orchestrator for vehicle speed estimation pipeline.

Pipeline per frame:
  1. Read frame + accurate PTS timestamp
  2. Detect vehicles (YOLO11)
  3. Track vehicles (ByteTrack)
  4. Transform positions to world coordinates (Homography)
  5. Estimate speed (smoothed, calibrated)
  6. Detect violations (threshold + duration)
  7. Annotate frame
  8. Save evidence for violations
  9. Write output video

Returns:
  - Annotated output video path
  - List of ViolationRecord dicts
  - Analytics dict (counts, speeds, benchmark)
  - CSV report path
"""

from .speed_session import SpeedSession

import cv2
import numpy as np
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict
import math
import subprocess
from uuid import uuid4

logger = logging.getLogger(__name__)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


class VideoProcessor(SpeedSession):
    """
    End-to-end video processing pipeline for vehicle speed estimation.
    
    Orchestrates: detection → tracking → calibration → speed estimation
                  → violation detection → evidence generation → reporting
    """

    def __init__(self, proc_config: Dict[str, Any], dataset_config: Dict[str, Any]):
        """
        Args:
            proc_config: Processing parameters (speed_limit, confidence, etc.)
            dataset_config: ForeSpeed dataset configuration
        """
        super().__init__(proc_config, dataset_config)
        self.confidence = proc_config.get("confidence", 0.4)
        self.model_path = proc_config.get("model_path")
        self.output_dir = Path(proc_config.get("output_dir", "outputs"))
        self.video_name = proc_config.get("video_name")
        
        # Lazy-init subcomponents
        self._detector = None
        self._tracker = None
        self._evidence_generator = None
        self._report_generator = None
        self._anpr = None
        self._anpr_failed = False
        self._plate_scan = None

    def _initialize_components(self, video_path: str, resolution, progress_cb: Optional[Callable] = None):
        """Initialize all pipeline components."""
        
        def _prog(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)
            else:
                logger.info(msg)
        
        _prog(0.02, "🔍 Loading YOLO11 detector...")
        from .vehicle_detector import VehicleDetector
        self._detector = VehicleDetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence,
        )
        
        _prog(0.08, "🔄 Initializing ByteTrack tracker...")
        from .vehicle_tracker import VehicleTracker
        self._tracker = VehicleTracker(
            track_high_thresh=self.confidence,
            track_low_thresh=min(0.1, self.confidence),
            new_track_thresh=self.confidence,
        )
        _prog(0.10, f"Active tracker: {self._tracker.name}")

        self.initialize_speed(video_path, resolution, _prog)
        key = self._profile_key

        evidence_dir = self.output_dir / "evidence"
        from .evidence_generator import EvidenceGenerator
        self._evidence_generator = EvidenceGenerator(
            output_dir=str(evidence_dir),
        )
        
        from .report_generator import ReportGenerator
        self._report_generator = ReportGenerator(
            output_dir=str(self.output_dir / "reports")
        )
        self._anpr = None
        self._anpr_failed = False
        self._plate_scan = None
        options = self.proc_config.get("anpr", {})
        if key == "Roboflow_Vehicles":
            options = {**options, **options.get("roboflow_scan", {})}
        if options.get("enabled", bool(options)):
            try:
                from modules.anpr import ANPRProcessor

                self._anpr = ANPRProcessor(options, Path(__file__).resolve().parents[2])
                if options.get("scan_all_tracks"):
                    self._plate_scan = self._anpr.scan
            except Exception:
                self._anpr_failed = True
                logger.exception("ANPR initialization failed. Continuing speed processing.")

    def _recognize_violation(self, violation):
        if self._anpr_failed:
            violation.anpr_status = "Processing Failed"
        if self._anpr is None:
            return None
        try:
            result = self._anpr.recognize_track(violation.track_id)
            violation.vehicle_number = result.vehicle_number
            violation.plate_confidence = result.plate_confidence
            violation.anpr_status = result.status
            violation.plate_bbox = result.plate_bbox
            violation.plate_frame_number = result.frame_number
            violation.plate_detection_confidence = result.detection_confidence
            violation.plate_raw_text = result.raw_text
            violation.plate_candidate = result.candidate
            violation.plate_candidate_format = result.candidate_format
            violation.plate_candidate_frames = result.candidate_frames
            violation.plate_character_agreement = result.character_agreement
            violation.plate_engine_results = result.engine_results
            return result
        except Exception:
            violation.anpr_status = "Processing Failed"
            logger.exception("ANPR failed for track %s", violation.track_id)
            return None

    def process(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file through the full pipeline.
        
        Args:
            video_path: Path to input video file
            progress_callback: fn(progress_0_to_1, message, log_text="")
        
        Returns:
            Results dict containing violations, analytics, paths, benchmark
        """
        def _prog(pct, msg, log=""):
            if progress_callback:
                progress_callback(pct, msg, log)
            logger.info(msg)

        from imageio_ffmpeg import get_ffmpeg_exe

        # Validate video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.video_name = self.proc_config.get("video_name") or Path(video_path).name
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        playback_fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        _prog(0.01, f"Video: {frame_w}×{frame_h}, {fps:.2f} nominal FPS, {total_frames} frames")
        writer = None
        working_path = None
        encoded_path = None
        videos_dir = self.output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.video_name).stem
        out_vid_path = videos_dir / f"{stem}_{uuid4().hex[:10]}_annotated.mp4"
        try:
            self._initialize_components(video_path, (frame_w, frame_h), _prog)
            working_path = out_vid_path.with_suffix(".working.mp4")
            writer = cv2.VideoWriter(str(working_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     playback_fps, (frame_w, frame_h))
            if not writer.isOpened():
                raise RuntimeError("Cannot create annotated video")
            first_timestamp = None
            previous_timestamp = None
            previous_tracks = {}
            traced_track = None
            traced_reasons = set()
            invalid_timestamps = 0
            # Tracking state
            all_speeds: List[float] = []
            track_speed_history: Dict[int, Dict] = defaultdict(lambda: {"times": [], "speeds": []})
            total_detections = 0
            processed_frames = 0
            t_start = time.time()
        
            _prog(0.20, "🚀 Starting frame-by-frame processing...")
        
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                # Accurate PTS timestamp
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = pts_ms / 1000.0
                timing_valid = math.isfinite(timestamp_sec) and timestamp_sec >= 0 and (
                    previous_timestamp is None or timestamp_sec > previous_timestamp
                )
                if timing_valid:
                    if first_timestamp is None:
                        first_timestamp = timestamp_sec
                    previous_timestamp = timestamp_sec
                else:
                    invalid_timestamps += 1
            
                # === DETECTION ===
                detections = self._detector.detect(frame)
                total_detections += len(detections)
            
                # === TRACKING ===
                active_tracks = self._tracker.update(
                    detections, frame, timestamp_sec, self._calibration if timing_valid else None
                )
                if self._anpr is not None:
                    try:
                        self._anpr.observe(frame, frame_idx, active_tracks)
                    except Exception:
                        self._anpr = None
                        self._anpr_failed = True
                        logger.exception("ANPR crop buffering failed. Continuing speed processing.")
            
                speeds = self.update_speeds(active_tracks, previous_tracks, timing_valid)
                previous_tracks = active_tracks

                if self._estimated_speed_mode == "assumed homography" and active_tracks:
                    if traced_track is None:
                        traced_track = next(iter(active_tracks))
                    if traced_track in active_tracks:
                        self._trace_speed_sample(
                            active_tracks[traced_track], timestamp_sec, timing_valid,
                            speeds.get(traced_track), traced_reasons,
                        )

                # Collect speed data
                for tid, spd in speeds.items():
                    if spd is not None:
                        all_speeds.append(spd)
                        track_speed_history[tid]["times"].append(timestamp_sec)
                        track_speed_history[tid]["speeds"].append(spd)
            
                # === VIOLATION DETECTION ===
                for tid, state in active_tracks.items():
                    spd = speeds.get(tid)
                    viol = self._violation_detector.update(
                        state=state,
                        current_speed_kmh=spd,
                        frame_number=frame_idx,
                        timestamp_sec=timestamp_sec,
                        is_calibrated=self._calibrated,
                    )
                    if viol is not None:
                        plate_result = self._recognize_violation(viol) if self._plate_scan is None else None
                        # Generate evidence image
                        ev_path = self._evidence_generator.generate(frame, viol, self.video_name)
                        if not ev_path:
                            raise RuntimeError(f"Could not save evidence for track {tid} on frame {frame_idx}")
                        viol.evidence_path = ev_path
                        if plate_result is not None:
                            try:
                                self._evidence_generator.save_anpr_crops(viol, plate_result)
                            except Exception:
                                logger.exception("ANPR crop saving failed for track %s", tid)
            
                # === FRAME ANNOTATION ===
                annotated = self._evidence_generator.annotate_live_frame(
                    frame, active_tracks, speeds, self.speed_limit,
                    self._violation_detector, calibrated=self._calibrated,
                    estimated=self._estimated_speed_mode is not None,
                )
                if self._plate_scan is not None:
                    from modules.anpr.plate_scan import draw_plates

                    draw_plates(annotated, self._plate_scan.by_frame.get(frame_idx, ()))
            
                # Frame counter overlay
                cv2.putText(
                    annotated,
                    f"Frame: {frame_idx+1}/{total_frames}  |  t={timestamp_sec:.2f}s  |  Tracks: {len(active_tracks)}",
                    (5, frame_h - 30 if self._calibrated else frame_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
                )
            
                writer.write(annotated)
                processed_frames += 1
                frame_idx += 1
            
                # Progress update every 30 frames
                if frame_idx % 30 == 0:
                    prog = 0.20 + 0.75 * min(frame_idx / max(total_frames, 1), 1.0)
                    viol_count = len(self._violation_detector.violations)
                    active_count = len(active_tracks)
                    fps_proc = processed_frames / max(time.time() - t_start, 1e-6)
                    _prog(
                        prog,
                        f"⚙️ Frame {frame_idx}/{total_frames} | "
                        f"Tracks: {active_count} | Violations: {viol_count} | "
                        f"Speed: {fps_proc:.1f} FPS",
                    )
        
            if not processed_frames:
                raise RuntimeError("No video frames could be decoded")
            writer.release()
            _prog(0.95, "Encoding browser-compatible annotated video")
            encoded_path = out_vid_path.with_suffix(".encoding.mp4")
            if not invalid_timestamps and processed_frames > 1:
                playback_fps = (processed_frames - 1) / (previous_timestamp - first_timestamp)
            subprocess.run(
                [get_ffmpeg_exe(), "-v", "error", "-y", "-r", f"{playback_fps:.9f}",
                 "-i", str(working_path),
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt",
                 "yuv420p", "-movflags", "+faststart", str(encoded_path)],
                check=True, capture_output=True,
            )
            encoded_path.replace(out_vid_path)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if working_path is not None:
                working_path.unlink(missing_ok=True)
            if encoded_path is not None:
                encoded_path.unlink(missing_ok=True)

        # Finalize
        t_elapsed = time.time() - t_start
        proc_fps = processed_frames / max(t_elapsed, 1e-6)
        all_tracks = self._tracker.get_all_tracks()
        violations = self._violation_detector.get_violations()
        plate_scan = self._finish_plate_scan(video_path, out_vid_path.stem, all_tracks, violations, _prog)
        
        # Analytics
        avg_speed = float(np.mean(all_speeds)) if all_speeds else None
        max_speed = float(np.max(all_speeds)) if all_speeds else None
        
        analytics = {
            "total_detections": total_detections,
            "unique_tracks": len(all_tracks),
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "processing_fps": proc_fps,
            "speed_history": all_speeds,
            "track_speeds": {
                str(tid): {"times": d["times"], "speeds": d["speeds"]}
                for tid, d in track_speed_history.items()
                if len(d["speeds"]) > 1
            },
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "invalid_timestamps": invalid_timestamps,
            "tracker": self._tracker.name,
            "calibrated": self._calibrated,
        }

        # Benchmark validation (ForeSpeed T1P1)
        benchmark = self._run_benchmark_validation(track_speed_history)
        
        # Reports
        _prog(0.96, "📄 Generating reports...")
        viol_dicts = [v.to_dict() for v in violations]
        
        csv_path = self._report_generator.generate_violations_report(
            violations=violations,
            video_name=self.video_name,
            speed_limit_kmh=self.speed_limit,
            analytics=analytics,
            benchmark=benchmark,
        )
        
        tracks_csv_path = self._report_generator.generate_all_tracks_report(
            tracks=all_tracks,
            speed_limit_kmh=self.speed_limit,
            video_name=self.video_name,
        )
        
        _prog(1.0, f"✅ Done! Processed {processed_frames} frames in {t_elapsed:.1f}s | {len(violations)} violations")
        
        return {
            "output_video_path": str(out_vid_path),
            "tracks_csv_path": tracks_csv_path,
            "calibration_status": self._calibration_status,
            "estimated_speed_mode": self._estimated_speed_mode,
            "violations": viol_dicts,
            "analytics": analytics,
            "benchmark": benchmark,
            "csv_report_path": csv_path,
            "plate_scan": plate_scan,
            "calibrated": self._calibrated,
        }

    def _finish_plate_scan(self, video_path, run_name, tracks, violations, progress):
        scan = self._plate_scan
        if scan is None:
            return None
        progress(0.96, "OCR Processing — reading the best plate crops by track")
        for track_id in tracks:
            scan.result(track_id)
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        try:
            for violation in violations:
                result = self._recognize_violation(violation)
                if result is not None:
                    try:
                        self._evidence_generator.finalize_plate_evidence(violation, result, scan, cap)
                    except (OSError, cv2.error):
                        logger.exception("Could not save plate evidence for track %s", violation.track_id)
        finally:
            cap.release()
        try:
            manifest, rows = scan.export(self.output_dir / "evidence" / run_name)
        except (OSError, cv2.error):
            logger.exception("Could not export plate observations")
            return {"summary": scan.summary(), "plates": [], "error": "Plate crop export failed"}
        return {"summary": scan.summary(), "plates": rows, "manifest": manifest,
                "tracks": {tid: result.status for tid, result in scan.results.items()}}

    def _trace_speed_sample(self, state, timestamp, timing_valid, speed, seen):
        """Log the first occurrence of each estimate outcome for one track."""
        estimator = self._speed_estimator
        count = min(len(state.timestamps), len(state.bottom_centers_world),
                    max(estimator.min_samples, estimator.smoothing_window))
        points = np.asarray(list(state.bottom_centers_world)[-count:]) if count else None
        displacement = float(np.linalg.norm(points[-1] - points[0])) if count > 1 else None
        if not timing_valid:
            status = "invalid frame timestamp"
        elif count < estimator.min_samples:
            status = "insufficient consecutive projected positions"
        elif not np.isfinite(points).all():
            status = "non-finite projected position"
        elif not np.all((np.diff(list(state.timestamps)[-count:]) > 0)
                        & (np.diff(list(state.timestamps)[-count:]) <= 0.5)):
            status = "invalid timestamp interval"
        elif speed is None:
            status = f"raw speed non-finite or above {estimator.max_speed_kmh:g} km/h"
        else:
            status = "estimate accepted"
        if status in seen:
            return
        seen.add(status)
        logger.info(
            "SpeedEstimator input: track=%s pixel=%s world=%s timestamp_s=%.6f "
            "positions=%s stored_speed_samples=%s window_displacement_m=%s speed_kmh=%s status=%s",
            state.track_id, state.latest_bottom_center_image, tuple(points[-1]) if count else None,
            timestamp, count, len(state.speed_samples_kmh), displacement, speed, status,
        )

    def _run_benchmark_validation(self, track_speed_history):
        mph = self._video_profile.get("reference_speed_mph")
        if mph is None:
            return None
        reference = mph * 1.609344
        track_id = self._video_profile.get("benchmark_track_id")
        samples = track_speed_history.get(track_id, {}).get("speeds", [])
        estimate = float(np.median(samples)) if self._calibrated and len(samples) >= 5 else None
        error = abs(estimate - reference) if estimate is not None else None
        return {
            "reference_speed_kmh": reference,
            "estimated_speed_kmh": estimate,
            "abs_error_kmh": error,
            "pct_error": error / reference * 100 if error is not None else None,
            "best_track_id": track_id,
            "diagnostics": "Benchmark comparison requires verified calibration." if not self._calibrated else
                           "Comparison requires sufficient samples from the identified benchmark vehicle.",
            "calibrated": self._calibrated,
        }
