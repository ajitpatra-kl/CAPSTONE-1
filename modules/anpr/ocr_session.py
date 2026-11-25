"""Session-local recognition with the bundled PP-OCRv4 ONNX recognizer."""

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import cv2

from .ocr_candidates import candidate_text
from .plate_enhancement import enhance_plate
from .plate_recognizer import OCRReading, PlateRecognizer


logger = logging.getLogger(__name__)


class PPOCRv4:
    def __init__(self, model_path=None):
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer

        path = model_path or Path(rapidocr_onnxruntime.__file__).parent / "models/ch_PP-OCRv4_rec_infer.onnx"
        if not Path(path).is_file():
            raise FileNotFoundError(f"PP-OCRv4 recognition model missing: {path}")
        self.recognizer = TextRecognizer({
            "model_path": str(path), "rec_img_shape": [3, 48, 320], "rec_batch_num": 6,
            "use_cuda": False, "use_dml": False,
            "intra_op_num_threads": 2, "inter_op_num_threads": 1,
        })

    def read(self, images):
        inputs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
                  for img in images.values()]
        readings, _ = self.recognizer(inputs)
        return {key: [OCRReading(text, float(score))]
                for key, (text, score) in zip(images, readings)}


@dataclass
class EngineResult:
    readings: dict
    elapsed_ms: float
    error: str = ""


def plate_variants(crop):
    native = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray, sharpened = PlateRecognizer.preprocess(crop)
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    return {
        "original": crop, "enhanced": enhance_plate(crop), "native-gray": native, "upscaled": gray,
        "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray),
        "sharpened": sharpened, "denoised": denoised,
        "adaptive-threshold": cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        ),
    }


class OCRSession:
    def __init__(self, config, project_root):
        self.primary = config.get("ocr_engine", "easyocr")
        if self.primary not in ("easyocr", "ppocrv4"):
            raise ValueError(f"Unknown OCR engine {self.primary!r}; use easyocr or ppocrv4")
        self.compare = bool(config.get("compare_ocr_engines", False))
        self.threshold = float(config.get("ocr_confidence", 0.45))
        self.engines = {}
        self.initialization_errors = {}
        custom = config.get("ppocr_model")
        self.factories = {
            "easyocr": lambda: PlateRecognizer(project_root / config.get("ocr_model_dir", "models/anpr")),
            "ppocrv4": lambda: PPOCRv4(project_root / custom if custom else None),
        }
        self._load(self.primary)
        if self.compare or self.primary not in self.engines:
            self._load("easyocr")
        if self.compare:
            self._load("ppocrv4")

    def _load(self, name):
        if name in self.engines or name in self.initialization_errors:
            return
        try:
            self.engines[name] = self.factories[name]()
        except Exception as error:
            self.initialization_errors[name] = f"{type(error).__name__}: {error}"
            logger.exception("Could not initialize OCR engine %s", name)

    def scan(self, crop):
        variants = plate_variants(crop)
        images = {}
        seen = set()
        for name, image in variants.items():
            key = (image.shape, image.tobytes())
            if key not in seen:
                images[name] = image
                seen.add(key)
        names = [self.primary]
        if self.compare:
            names += [name for name in self.factories if name != self.primary]
        results = {name: self._read(name, images) for name in names}
        primary = results[self.primary]
        if self.primary != "easyocr" and (primary.error or not any(
                item.confidence >= self.threshold and candidate_text(item.raw_text)
                for batch in primary.readings.values() for item in batch)):
            if "easyocr" not in results:
                results["easyocr"] = self._read("easyocr", images)
        return variants, results

    def _read(self, name, images):
        self._load(name)
        if name not in self.engines:
            return EngineResult({}, 0, self.initialization_errors[name])
        start = perf_counter()
        try:
            return EngineResult(self.engines[name].read(images), (perf_counter() - start) * 1000)
        except Exception as error:
            logger.exception("OCR inference failed for %s", name)
            return EngineResult({}, (perf_counter() - start) * 1000, f"{type(error).__name__}: {error}")
