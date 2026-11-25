"""Video-specific geometry and saved configurations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import hypot, isfinite
from pathlib import Path
import json
from uuid import uuid4

import cv2
import numpy as np


DIRECTIONS = {"Bottom": (0, 1), "Top": (0, -1), "Left": (-1, 0), "Right": (1, 0)}
LAMP_LAYOUTS = ("Automatic", "Vertical, red on top", "Horizontal, red on left",
                "Horizontal, red on right", "Color only")


@dataclass
class Configuration:
    name: str
    video_id: str
    width: int
    height: int
    fps: float
    roi: list
    line: list
    approach: str
    lane: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    association: str = "Selected signal controls the configured stop line and monitored approach"
    lamp_layout: str = "Automatic"

    def validate(self):
        if not self.name.strip() or not self.video_id or not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("A name and valid video metadata are required.")
        if len(self.roi) != 4:
            raise ValueError("Draw a rectangle around the traffic signal.")
        if self.lamp_layout not in LAMP_LAYOUTS:
            raise ValueError("Select a supported traffic lamp layout.")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height
                and x2 - x1 >= 3 and y2 - y1 >= 3):
            raise ValueError("Draw a signal rectangle at least 3 × 3 pixels inside the frame.")
        if len(self.line) != 2 or any(not self.inside(point) for point in self.line):
            raise ValueError("Draw two stop-line endpoints inside the frame.")
        (ax, ay), (bx, by) = self.line
        dx, dy = bx - ax, by - ay
        ux, uy = DIRECTIONS[self.approach]
        if hypot(dx, dy) < 8 or abs(dx * uy - dy * ux) < 0.2 * hypot(dx, dy):
            raise ValueError("Use a stop line at least 8 pixels long that crosses the approach direction.")
        if self.lane and (len(self.lane) < 3 or any(not self.inside(p) for p in self.lane)
                          or cv2.contourArea(np.asarray(self.lane, np.float32)) < 8):
            raise ValueError("Use at least three vertices around the monitored lane or clear it.")

    def inside(self, point):
        return len(point) == 2 and 0 <= point[0] < self.width and 0 <= point[1] < self.height


def save_configuration(directory: Path, settings: dict, config_id: str):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config_id}.json"
    temporary = directory / f"{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError("Could not decode the video or read its frame rate and duration.")
        return frame.shape[1], frame.shape[0], fps, count
    finally:
        capture.release()


def read_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not decode frame {index}.")
        return frame
    finally:
        capture.release()
