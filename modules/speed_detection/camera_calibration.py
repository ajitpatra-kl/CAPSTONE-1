"""
modules/speed_detection/camera_calibration.py

Homography-based camera calibration for real-world speed estimation.

The ForeSpeed dataset provides videos from multiple camera perspectives.
This module handles:
 - Loading calibration profiles for known cameras
 - Computing homography matrices from road reference points
 - Transforming image pixel coordinates → real-world road coordinates (metres)

Reference: ForeSpeed paper (https://www.mdpi.com/2313-433X/12/10/488)

"""

import cv2
import numpy as np
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class HomographyCalibration:
    """
    Encapsulates a homography matrix H that maps image points (u,v) to
    real-world ground-plane coordinates (X,Y) in metres.

    Usage:
        cal = HomographyCalibration(src_pts, dst_pts)
        X, Y = cal.transform(u, v)       # single point
        pts_world = cal.transform_many(pts_image)  # Nx2 array
    """

    def __init__(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        profile_name: str = "unknown",
        notes: str = "",
        *,
        reference_polygon: bool = False,
    ):
        """
        Args:
            image_points: (N,2) float array of pixel coords [u, v]
            world_points: (N,2) float array of real-world coords [X, Y] in metres
            profile_name: Identifier string for this calibration
            notes: Human-readable description / caveats
        """
        if (image_points.ndim != 2 or image_points.shape[1] != 2
                or image_points.shape != world_points.shape or len(image_points) < 4
                or not np.isfinite(image_points).all() or not np.isfinite(world_points).all()):
            raise ValueError("Homography requires at least 4 point correspondences.")

        self.profile_name = profile_name
        self.notes = notes
        self.image_points = image_points.astype(np.float32)
        self.world_points = world_points.astype(np.float32)

        self.reference_polygon = reference_polygon
        if reference_polygon:
            self.H = cv2.getPerspectiveTransform(self.image_points, self.world_points)
            inlier_count = len(self.image_points)
        else:
            self.H, mask = cv2.findHomography(
                self.image_points, self.world_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            inlier_count = int(mask.sum()) if mask is not None else 0
        if self.H is None:
            raise RuntimeError(
                f"Homography computation failed for profile '{profile_name}'. "
                "Check that reference points are not collinear."
            )

        logger.info(
            f"Homography computed for '{profile_name}' | "
            f"{inlier_count}/{len(image_points)} inliers | "
            f"Notes: {notes}"
        )

        # Compute reprojection error for validation
        self.reprojection_error_m = self._compute_reprojection_error()

    def _compute_reprojection_error(self) -> float:
        """Compute mean reprojection error in metres."""
        try:
            pred = self.transform_many(self.image_points)
            errors = np.linalg.norm(pred - self.world_points, axis=1)
            return float(np.mean(errors))
        except Exception:
            return float("inf")

    def transform(self, u: float, v: float) -> Tuple[float, float]:
        """Transform single image point (u,v) → world point (X,Y) in metres."""
        pt = np.array([[[u, v]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.H)
        return float(world_pt[0, 0, 0]), float(world_pt[0, 0, 1])

    def transform_many(self, pts: np.ndarray) -> np.ndarray:
        """Transform Nx2 image points → Nx2 world points in metres."""
        pts_f = pts.astype(np.float32).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pts_f, self.H)
        return world.reshape(-1, 2)

    def covers_anchor(self, point: Tuple[float, float]) -> bool:
        """Limit reference-profile speeds to the notebook's road polygon."""
        return not self.reference_polygon or cv2.pointPolygonTest(self.image_points, point, False) >= 0

    def is_valid(self) -> bool:
        """Returns True if calibration is usable (H computed, reasonable error)."""
        return self.H is not None and np.isfinite(self.H).all() and np.linalg.matrix_rank(self.H) == 3

    def summary(self) -> str:
        return (
            f"CalibrationProfile: {self.profile_name}\n"
            f"  Reprojection error: {self.reprojection_error_m:.4f} m\n"
            f"  Notes: {self.notes}"
        )


class CalibrationManager:
    """Bind camera profiles to video bytes and decoded dimensions."""

    def __init__(self, dataset_config: Optional[Dict] = None):
        config = dataset_config or {}
        self.profiles = {
            **config.get("forespeed", {}).get("videos", {}),
            **config.get("roboflow", {}).get("videos", {}),
        }
        self._calibrations: Dict[str, HomographyCalibration] = {}

    def detect_profile(self, video_path: str, resolution: Tuple[int, int]) -> Optional[str]:
        candidates = {
            key: profile for key, profile in self.profiles.items()
            if tuple(profile["resolution"]) == tuple(resolution)
        }
        if not candidates:
            return None
        digest = hashlib.sha256()
        with Path(video_path).open("rb") as video:
            for block in iter(lambda: video.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        return next((key for key, profile in candidates.items()
                     if profile["sha256"] == checksum), None)

    def get_calibration(self, profile_key: str) -> Optional[HomographyCalibration]:
        """Load an official reference mapping or an independently checked calibration."""
        profile = self.profiles.get(profile_key, {})
        status = profile.get("calibration_status")
        if status not in ("verified", "reference"):
            return None
        calibration = profile.get("calibration", {})
        try:
            if not calibration["source"].strip():
                return None
            image = np.asarray(calibration["image_points"], dtype=np.float32)
            world = np.asarray(calibration["world_points_m"], dtype=np.float32)
            if status == "reference":
                if image.shape != (4, 2) or world.shape != (4, 2):
                    return None
                cal = HomographyCalibration(
                    image, world, profile_key, calibration["source"], reference_polygon=True
                )
                if cal.is_valid() and cal.reprojection_error_m < 0.001:
                    return cal
                return None
            check_image = np.asarray(calibration["check_image_points"], dtype=np.float32)
            check_world = np.asarray(calibration["check_world_points_m"], dtype=np.float32)
            if (check_image.ndim != 2 or check_image.shape[1] != 2
                    or not len(check_image) or check_world.shape != check_image.shape
                    or any(np.any(np.linalg.norm(image - point, axis=1) < 1)
                           for point in check_image)):
                return None
            cal = HomographyCalibration(image, world, profile_key, calibration["source"])
            errors = np.linalg.norm(cal.transform_many(check_image) - check_world, axis=1)
            tolerance = float(calibration["check_tolerance_m"])
            if (cal.is_valid() and np.isfinite(tolerance) and 0 < tolerance
                    and cal.reprojection_error_m <= tolerance and np.all(errors <= tolerance)):
                return cal
        except (KeyError, TypeError, ValueError, RuntimeError, cv2.error):
            logger.exception("Invalid calibration for %s", profile_key)
        return None

    def get_calibration_from_custom_points(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        name: str = "custom",
    ) -> HomographyCalibration:
        """
        Build calibration from user-supplied point correspondences.
        image_points: Nx2 pixel coords
        world_points: Nx2 real-world coords in metres
        """
        cal = HomographyCalibration(
            image_points=image_points,
            world_points=world_points,
            profile_name=name,
            notes="User-supplied calibration",
        )
        self._calibrations[name] = cal
        return cal

    @staticmethod
    def pixel_speed_fallback(
        pixel_distance: float,
        time_seconds: float,
        pixels_per_metre: float,
    ) -> float:
        """
        Fallback speed calculation when no calibration is available.
        Uses a pixel-to-metre ratio (unreliable without proper calibration).

        Args:
            pixel_distance: displacement in pixels
            time_seconds: elapsed time
            pixels_per_metre: scale factor (highly approximate)
        Returns:
            speed in km/h (marked as unreliable)
        """
        if time_seconds <= 0 or pixels_per_metre <= 0:
            return 0.0
        speed_mps = (pixel_distance / pixels_per_metre) / time_seconds
        return speed_mps * 3.6
