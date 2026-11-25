"""Create a readable display copy without reconstructing plate characters."""

import cv2
import numpy as np


def enhance_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # Cap enlargement to keep tiny or mislocalized detections from dominating memory.
    scale = min(4, max(1, 64 / height), max(1, 640 / width))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, smooth)
    if np.median(residual) >= 3 and min(gray.shape) >= 7:
        gray = cv2.fastNlMeansDenoising(gray, None, 4, 5, 11)
    if scale > 1:
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LANCZOS4)
    midpoint = np.median(gray)
    if midpoint < 90 or midpoint > 205:
        gamma = 0.85 if midpoint < 90 else 1.12
        lut = np.round(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
        gray = cv2.LUT(gray, lut)
    low, high = np.percentile(gray, (10, 90))
    if high - low < 110:
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
    return cv2.addWeighted(gray, 1.18, blurred, -0.18, 0)
