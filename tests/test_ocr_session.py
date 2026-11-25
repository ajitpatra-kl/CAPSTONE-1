from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from modules.anpr.ocr_candidates import rank_candidates
from modules.anpr.ocr_session import OCRSession, plate_variants
from modules.anpr.plate_recognizer import OCRReading, PlateRecognizer
from modules.anpr.plate_validator import validate_registration


def test_primary_initialization_error_is_recorded_and_fallback_loads_once(monkeypatch, tmp_path):
    pp = Mock(side_effect=FileNotFoundError("missing ONNX checkpoint"))
    easy = Mock(return_value=SimpleNamespace(read=lambda images: {"original": [OCRReading("ABC1234", .8)]}))
    easy.preprocess = PlateRecognizer.preprocess
    monkeypatch.setattr("modules.anpr.ocr_session.PPOCRv4", pp)
    monkeypatch.setattr("modules.anpr.ocr_session.PlateRecognizer", easy)
    session = OCRSession({"ocr_engine": "ppocrv4"}, tmp_path)
    crop = np.full((12, 40, 3), 130, np.uint8)
    for _ in range(2):
        _, engines = session.scan(crop)
        assert engines["ppocrv4"].error == "FileNotFoundError: missing ONNX checkpoint"
        assert engines["easyocr"].readings["original"][0].raw_text == "ABC1234"
    pp.assert_called_once()
    easy.assert_called_once()


def test_character_agreement_returns_an_observed_string_and_distinct_frames():
    frames = [SimpleNamespace(frame=frame, rank=rank) for frame, rank in [(5, 4), (10, 4), (15, 2)]]
    reads = [
        ("4SH4775", frames[0], OCRReading("4SH 4775", .8)),
        ("4SH4775", frames[0], OCRReading("4SH4775", .7)),
        ("4SH4775", frames[1], OCRReading("4SH4775", .8)),
        ("LSH4775", frames[2], OCRReading("LSH4775", .9)),
    ]
    consistent, _, agreement, text, matches = rank_candidates(reads)
    assert consistent and agreement > .5
    assert text == "4SH4775" and set(matches) == {5, 10}
    assert not rank_candidates(reads[:2])[0]


def test_enhancement_keeps_original_and_vintage_is_only_a_format_check():
    original = np.arange(12 * 40 * 3, dtype=np.uint8).reshape(12, 40, 3)
    variants = plate_variants(original)
    assert variants["original"] is original
    assert variants["upscaled"].shape == (48, 160)
    assert set(np.unique(variants["adaptive-threshold"])) <= {0, 255}
    assert validate_registration("MH VA AB 0042") == "MHVAAB0042"
    assert validate_registration("MH VA AB 0000") == ""
