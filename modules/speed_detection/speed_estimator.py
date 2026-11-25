"""
modules/speed_detection/speed_estimator.py

Real-world vehicle speed estimation using homography-calibrated coordinates.

Speed formula:
    Speed (km/h) = Distance (metres) / Time (seconds) × 3.6

Key design decisions:
 - Uses world-space (metre) coordinates from homography transformation
 - Uses a median of timestamp-aware displacement slopes to suppress box jitter
 - Rejects outlier measurements using σ-clipping
 - Requires minimum N samples before reporting speed
 - Allows a pixel-based estimate when an explicit scale is provided
"""

import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
from collections import deque

from .vehicle_tracker import TrackState

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """
    Estimates real-world vehicle speed for each tracked vehicle.
    
    Uses calibrated trajectory samples and video presentation timestamps.
    """

    def __init__(
        self,
        smoothing_window: int = 7,
        min_samples: int = 5,
        max_speed_kmh: float = 200.0,
        outlier_sigma: float = 2.5,
        use_kalman: bool = True,
        pixel_scale_mpp: Optional[float] = None,
    ):
        """
        Args:
            smoothing_window: Recent trajectory samples to fit (at least 5)
            min_samples: Minimum displacement samples before reporting speed
            max_speed_kmh: Upper bound on accepted speed measurements
            outlier_sigma: Standard deviations beyond which samples are rejected
            use_kalman: Apply a simple Kalman-like smoothing on top of SG
            pixel_scale_mpp: Optional assumed metres per pixel for uncalibrated video
        """
        # Ensure smoothing window is odd and at least 5
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.smoothing_window = max(5, self.smoothing_window)
        self.min_samples = min_samples
        self.max_speed_kmh = max_speed_kmh
        self.outlier_sigma = outlier_sigma
        self.use_kalman = use_kalman
        if pixel_scale_mpp is not None and (not np.isfinite(pixel_scale_mpp) or pixel_scale_mpp <= 0):
            raise ValueError("pixel_scale_mpp must be positive and finite")
        self.pixel_scale_mpp = pixel_scale_mpp
        
        # Per-track speed buffer (after smoothing)
        self._speed_buffers: Dict[int, deque] = {}
        # Per-track Kalman state (alpha filter)
        self._kalman_state: Dict[int, float] = {}

    def estimate_speed(
        self,
        state: TrackState,
        calibrated: bool = True,
    ) -> Optional[float]:
        """
        Estimate the current speed (km/h) for a tracked vehicle.
        
        Args:
            state: TrackState with position and timestamp history
            calibrated: True if world-space coordinates are available
        
        Returns:
            Smoothed speed estimate in km/h, or None if insufficient data
        """
        tid = state.track_id
        
        if calibrated:
            speed = self._estimate_calibrated(state)
        elif self.pixel_scale_mpp is not None:
            speed = self._estimate_displacement(
                state.timestamps, state.bottom_centers_image, self.pixel_scale_mpp
            )
        else:
            return None
        if speed is None:
            self.reset_track(tid)
            return None

        if not np.isfinite(speed) or speed > self.max_speed_kmh:
            return None
        
        # Kalman-style alpha filter
        if self.use_kalman:
            alpha = 0.4
            if tid not in self._kalman_state or self._kalman_state[tid] == 0:
                self._kalman_state[tid] = speed
            else:
                speed = alpha * speed + (1 - alpha) * self._kalman_state[tid]
                self._kalman_state[tid] = speed
        
        # Store in speed buffer
        if tid not in self._speed_buffers:
            self._speed_buffers[tid] = deque(maxlen=30)
        self._speed_buffers[tid].append(speed)
        
        # Return median of recent buffer for stability
        buf = list(self._speed_buffers[tid])
        if len(buf) >= 3:
            speed = float(np.median(buf[-min(7, len(buf)):]))
        
        state.smoothed_speed_kmh = speed
        return speed

    def _estimate_calibrated(self, state: TrackState) -> Optional[float]:
        """Fit road-plane position against elapsed time over a recent window."""
        return self._estimate_displacement(state.timestamps, state.bottom_centers_world)

    def _estimate_displacement(self, timestamps, positions, metres_per_unit=1.0) -> Optional[float]:
        # Track timestamps may have been cleared on reacquisition while image history remains.
        count = min(len(timestamps), len(positions), max(self.min_samples, self.smoothing_window))
        if count < self.min_samples:
            return None
        times = np.asarray(list(timestamps)[-count:], dtype=float)
        points = np.asarray(list(positions)[-count:], dtype=float)
        intervals = np.diff(times)
        if not np.all((intervals > 0) & (intervals <= 0.5)) or not np.isfinite(points).all():
            return None
        # The median of pairwise velocity slopes tolerates isolated box jumps.
        first, last = np.triu_indices(count, k=1)
        velocities = (points[last] - points[first]) / (times[last] - times[first])[:, None]
        velocity = np.median(velocities, axis=0)
        return float(np.linalg.norm(velocity) * metres_per_unit * 3.6)

    def _sigma_clip(self, arr: np.ndarray) -> np.ndarray:
        """Remove outliers beyond outlier_sigma standard deviations."""
        if len(arr) < 3:
            return arr
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-6:
            return arr
        mask = np.abs(arr - mean) < self.outlier_sigma * std
        result = arr[mask]
        return result if len(result) > 0 else arr

    def update_all_tracks(
        self,
        tracks: Dict[int, TrackState],
        calibrated: bool = True,
    ) -> Dict[int, Optional[float]]:
        """
        Update speed estimates for all active tracks.
        Returns dict of track_id → speed_kmh (or None).
        """
        speeds = {}
        for tid, state in tracks.items():
            speed = self.estimate_speed(state, calibrated=calibrated)
            speeds[tid] = speed
            if speed is not None:
                state.speed_samples_kmh.append(speed)
        return speeds

    def get_peak_speed(self, track_id: int) -> Optional[float]:
        """Return the peak observed speed for a track."""
        buf = self._speed_buffers.get(track_id)
        if not buf:
            return None
        return float(max(buf))

    def get_average_speed(self, track_id: int) -> Optional[float]:
        """Return the average observed speed for a track."""
        buf = self._speed_buffers.get(track_id)
        if not buf or len(buf) < 3:
            return None
        return float(np.mean(list(buf)))

    def reset_track(self, track_id: int):
        """Clear speed history for a specific track."""
        self._speed_buffers.pop(track_id, None)
        self._kalman_state.pop(track_id, None)
