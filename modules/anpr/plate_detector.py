from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path


@dataclass(frozen=True)
class PlateDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


class PlateDetector:
    def __init__(self, weights: Path, confidence: float, max_plates: int, image_size: int = 640):
        if not weights.is_file():
            raise FileNotFoundError(f"Plate detector weights not found: {weights}")
        if weights.name.lower() == "yolo11n.pt":
            raise ValueError("Configure a dedicated plate detector checkpoint")
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        if len(self.model.names) != 1:
            raise ValueError("Expected a single-class plate detector")
        self.confidence = confidence
        self.max_plates = max_plates
        self.image_size = image_size

    def detect(self, vehicle_crop):
        height, width = vehicle_crop.shape[:2]
        results = self.model.predict(
            vehicle_crop, imgsz=self.image_size, conf=self.confidence, max_det=self.max_plates,
            device="cpu", verbose=False,
        )
        detections = []
        for result in results:
            for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                left, top, right, bottom = box
                bbox = (max(0, floor(left)), max(0, floor(top)),
                        min(width, ceil(right)), min(height, ceil(bottom)))
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    detections.append(PlateDetection(bbox, float(score)))
        return detections
