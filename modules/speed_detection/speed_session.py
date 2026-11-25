"""Shared speed calibration and per-frame continuity rules."""

import logging

import cv2
import numpy as np

from .camera_calibration import CalibrationManager, HomographyCalibration
from .speed_estimator import SpeedEstimator
from .violation_detector import ViolationDetector


logger = logging.getLogger(__name__)


class SpeedSession:
    def __init__(self, proc_config, dataset_config):
        self.proc_config = proc_config
        self.dataset_config = dataset_config
        self.speed_limit = proc_config.get("speed_limit", 40.0)
        self.smoothing_window = proc_config.get("smoothing_window", 7)
        self.violation_duration = proc_config.get("violation_duration", 1.0)
        self._calibration = None
        self._calibrated = False
        self._speed_estimator = None
        self._violation_detector = None

    def initialize_speed(self, video_path, resolution, _prog=lambda *args: None):
        manager = CalibrationManager(self.dataset_config)
        key = manager.detect_profile(video_path, resolution)
        self._profile_key = key
        self._video_profile = manager.profiles.get(key, {})
        self._calibration = manager.get_calibration(key) if key else None
        self._calibrated = self._calibration is not None
        self._calibration_status = (
            f"Reference calibration: {self._video_profile['camera']} (estimated speeds)."
            if self._calibrated and self._video_profile.get("calibration_status") == "reference" else
            f"Verified calibration: {self._video_profile['camera']}"
            if self._calibrated else
            f"Calibration unavailable: {self._video_profile['camera']} lacks verified road correspondences."
            if key else "Uncalibrated video: no matching verified camera profile."
        )
        speed_options = self.proc_config.get("speed_estimation", {})

        self._estimated_speed_mode = None
        pixel_scale = None
        custom_road = speed_options.get("custom_road")
        if custom_road is not None and custom_road.get("image_points") is None:
            raise ValueError("Custom road calibration requires four image points.")
        if not self._calibrated or custom_road is not None:
            if custom_road is not None:
                self._calibrated = False
                self._calibration = None
            assumed_road = custom_road or speed_options.get("assumed_road") or {}
            if assumed_road.get("image_points") is not None:
                try:
                    self._calibration = self._assumed_homography(assumed_road, resolution)
                except (ValueError, TypeError, RuntimeError, cv2.error):
                    logger.exception("Assumed homography generation failed")
                    raise
                logger.info("Assumed homography generated: %s", self._calibration.H.tolist())
                self._estimated_speed_mode = "assumed homography"
            elif speed_options.get("pixel_scale_mpp") is not None:
                pixel_scale = speed_options["pixel_scale_mpp"]
                self._estimated_speed_mode = "assumed pixel scale"
            if self._estimated_speed_mode:
                self._calibration_status = f"Estimated speeds using {self._estimated_speed_mode}."
        if self._calibration is None:
            logger.info("No homography generated for this run")
        _prog(0.15, self._calibration_status)

        _prog(0.16, "📊 Initializing speed estimator...")
        self._speed_estimator = SpeedEstimator(
            smoothing_window=self.smoothing_window,
            min_samples=5,
            use_kalman=False,
            max_speed_kmh=200.0,
            outlier_sigma=2.5,
            pixel_scale_mpp=pixel_scale,
        )
        
        _prog(0.17, "🚨 Initializing violation detector...")
        self._violation_detector = ViolationDetector(
            speed_limit_kmh=self.speed_limit,
            min_violation_duration_sec=self.violation_duration,
            confirmation_frames=3,
            allow_estimated=self._estimated_speed_mode is not None,
        )
        
    def update_speeds(self, active_tracks, previous_tracks, timing_valid):
        interrupted = previous_tracks.keys() - active_tracks.keys()
        if not timing_valid:
            interrupted |= active_tracks.keys()
            for state in active_tracks.values():
                state.bottom_centers_world.clear()
                state.timestamps.clear()
        for tid in interrupted:
            self._speed_estimator.reset_track(tid)
            self._violation_detector.interrupt_track(tid)

        if timing_valid and self._calibration is not None and self._calibration.reference_polygon:
            for tid, state in active_tracks.items():
                if not self._calibration.covers_anchor(state.bottom_centers_image[-1]):
                    state.bottom_centers_world.clear()
                    state.timestamps.clear()
                    self._speed_estimator.reset_track(tid)
                    self._violation_detector.interrupt_track(tid)

        # === SPEED ESTIMATION ===
        return self._speed_estimator.update_all_tracks(
            active_tracks, calibrated=self._calibration is not None
        ) if timing_valid else {}

    @staticmethod
    def _assumed_homography(road, resolution):
        """Pair four native-frame corners with the configured road rectangle."""
        image = np.asarray(road["image_points"], dtype=np.float32)
        dimensions = np.asarray([road.get("width_m"), road.get("length_m")], dtype=float)
        if (image.shape != (4, 2) or not np.isfinite(image).all()
                or not np.isfinite(dimensions).all() or (dimensions <= 0).any()
                or (image < 0).any() or (image >= resolution).any()
                or not cv2.isContourConvex(image)):
            raise ValueError("Assumed road needs four ordered frame corners and positive width and length")
        width, length = dimensions
        world = np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)
        calibration = HomographyCalibration(image, world, "assumed road dimensions")
        if not calibration.is_valid():
            raise ValueError("Invalid assumed road homography")
        return calibration

