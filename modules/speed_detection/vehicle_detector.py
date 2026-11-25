"""
modules/speed_detection/vehicle_detector.py

Vehicle detection using Ultralytics YOLO11 pretrained model.
Detects cars, motorcycles, buses, and trucks.
Downloads model weights automatically on first use.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """Single vehicle detection result."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Bottom-centre point — used for road-plane transformation."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_xyxy(self) -> np.ndarray:
        return np.array(self.bbox, dtype=np.float32)
    
    def to_xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


class VehicleDetector:
    """
    YOLO11-based vehicle detector.
    
    Uses ultralytics YOLO with pretrained weights.
    Model is downloaded automatically if not found locally.
    """

    DEFAULT_MODEL = "yolo11n.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to .pt weights file. Defaults to yolo11n.pt (auto-downloaded).
            confidence_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            device: 'auto', 'cpu', 'cuda', '0', etc.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path or str(Path(__file__).resolve().parents[2] / "models" / self.DEFAULT_MODEL)
        self.device = self._resolve_device(device)
        self.model = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """Load YOLO11 model, downloading weights if necessary."""
        try:
            from ultralytics import YOLO
            
            model_p = Path(self.model_path)
            
            # If path given is absolute and exists, use it
            # Otherwise let ultralytics handle download
            if model_p.is_absolute() and not model_p.exists():
                # Download to the specified directory
                model_p.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Model not found at {model_p}. Downloading {model_p.name}...")
                self.model = YOLO(model_p.name)  # triggers auto-download
                # Move to desired location if downloaded to cwd
                import shutil, os
                downloaded = Path(model_p.name)
                if downloaded.exists() and not model_p.exists():
                    shutil.move(str(downloaded), str(model_p))
            else:
                self.model = YOLO(str(model_p) if model_p.exists() else model_p.name)
            
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device=self.device)
            
            logger.info(
                f"YOLO11 model loaded: {self.model_path} | device={self.device}"
            )

        except ImportError:
            raise ImportError(
                "ultralytics package not installed. Run: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image (H,W,3) numpy array
        Returns:
            List of Detection objects for detected vehicles
        """
        if self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=list(VEHICLE_CLASS_IDS.keys()),
                agnostic_nms=True,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            raise RuntimeError(f"Vehicle detection inference failed: {e}") from e

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                try:
                    cls_id = int(boxes.cls[i].item())
                    if cls_id not in VEHICLE_CLASS_IDS:
                        continue
                    
                    conf = float(boxes.conf[i].item())
                    if conf < self.confidence_threshold:
                        continue
                    
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Clamp to frame dimensions
                    h, w = frame.shape[:2]
                    x1 = max(0.0, float(x1))
                    y1 = max(0.0, float(y1))
                    x2 = min(float(w), float(x2))
                    y2 = min(float(h), float(y2))
                    
                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        continue  # skip tiny boxes
                    
                    det = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=VEHICLE_CLASS_IDS[cls_id],
                    )
                    detections.append(det)
                    
                except Exception as e:
                    logger.debug(f"Error parsing detection {i}: {e}")
                    continue

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        return [self.detect(f) for f in frames]

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                out, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
        return out

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "vehicle_classes": VEHICLE_CLASS_IDS,
        }
