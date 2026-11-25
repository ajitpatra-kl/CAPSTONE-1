from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class OCRReading:
    raw_text: str
    confidence: float


class PlateRecognizer:
    def __init__(self, model_dir: Path):
        if not (model_dir / "english_g2.pth").is_file():
            raise FileNotFoundError(f"EasyOCR english_g2.pth not found in {model_dir}")
        from easyocr import Reader

        self.reader = Reader(
            ["en"], gpu=False, detector=False, download_enabled=False,
            model_storage_directory=str(model_dir), user_network_directory=str(model_dir),
        )

    @staticmethod
    def preprocess(plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 48:
            scale = 48 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
        return gray, sharpened

    def recognize(self, plate_crop):
        readings = []
        for image in self.preprocess(plate_crop):
            for _, raw_text, score in self.reader.recognize(image, reformat=False):
                readings.append(OCRReading(raw_text, float(score)))
        return readings

    def read(self, images):
        readings = {}
        for name, image in images.items():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            readings[name] = [
                OCRReading(raw_text, float(score))
                for _, raw_text, score in self.reader.recognize(gray, reformat=False)
            ]
        return readings

    def scan_readings(self, plate_crop):
        gray, sharpened = self.preprocess(plate_crop)
        native_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        variants = {"native-gray": native_gray, "gray": gray, "contrast": contrast, "sharpened": sharpened}
        readings = {}
        seen = set()
        for name, image in variants.items():
            key = (image.shape, image.tobytes())
            if key in seen:
                continue
            seen.add(key)
            readings[name] = [
                OCRReading(raw_text, float(score))
                for _, raw_text, score in self.reader.recognize(image, reformat=False)
            ]
        return variants, readings
