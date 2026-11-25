"""
modules/speed_detection/__init__.py
Speed violation detection module initialization.
"""

from .video_processor import VideoProcessor
from .vehicle_detector import VehicleDetector
from .vehicle_tracker import VehicleTracker
from .camera_calibration import CalibrationManager, HomographyCalibration
from .speed_estimator import SpeedEstimator
from .violation_detector import ViolationDetector
from .evidence_generator import EvidenceGenerator
from .report_generator import ReportGenerator

__all__ = [
    "VideoProcessor",
    "VehicleDetector",
    "VehicleTracker",
    "CalibrationManager",
    "HomographyCalibration",
    "SpeedEstimator",
    "ViolationDetector",
    "EvidenceGenerator",
    "ReportGenerator",
]

__version__ = "1.0.0"
__description__ = "Vehicle speed estimation and overspeed violation detection module"
