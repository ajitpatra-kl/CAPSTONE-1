"""HSV lamp classification and conservative temporal confirmation."""

from dataclasses import dataclass

import cv2
import numpy as np


UNKNOWN = "UNKNOWN"
RANGES = {"RED": ((0, 10), (170, 179)), "YELLOW": ((11, 38),),
          "GREEN": ((39, 100),)}
COLORS = {"RED": (50, 50, 245), "YELLOW": (0, 220, 255),
          "GREEN": (75, 225, 75), UNKNOWN: (180, 180, 180)}


def classify_signal(frame, roi, layout="Automatic"):
    x1, y1, x2, y2 = roi
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    if layout == "Automatic":
        layout = "Color only"
        if height >= width * 1.35:
            layout = "Vertical, red on top"
        elif width >= height * 1.35:
            layout = "Horizontal, red on left"
    brightness_floor = int(max(145, np.percentile(hsv[:, :, 2], 50) + 18))
    masks = {}
    for color, ranges in RANGES.items():
        mask = np.zeros((height, width), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, (lo, 65, brightness_floor), (hi, 255, 255))
        masks[color] = mask
    scores = {}
    for color, mask in masks.items():
        # Clipped red lamps can have a warm core surrounded by red pixels.
        if color == "RED" and layout != "Color only":
            mask = mask | masks["YELLOW"]
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        scores[color] = 0.0
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if not (max(3, height * width * .002) <= area <= height * width * .55
                    and .32 <= w / h <= 3.1 and area / (w * h) >= .44):
                continue
            # Broad backgrounds and long streaks should not dominate a lamp.
            if (w == width and h == height) or (h == height and w > width * .72):
                continue
            component = labels[y:y+h, x:x+w] == index
            values = hsv[y:y+h, x:x+w, 2][component]
            cx, cy = centers[index]
            if layout != "Color only":
                position = cy / height if layout.startswith("Vertical") else cx / width
                if layout.endswith("right"):
                    position = 1 - position
                expected = {"RED": .17, "YELLOW": .50, "GREEN": .83}[color]
                if abs(expected - position) > .19:
                    continue
                if color == "RED" and np.count_nonzero(masks["RED"][y:y+h, x:x+w][component]) < max(3, area * .045):
                    continue
            centered = 1 - .25 * min(1, abs(cx / width - .5) + abs(cy / height - .5))
            score = (area / (w * h)) * float(values.mean()) / 255 * centered
            scores[color] = max(scores[color], score)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    state, best = ranked[0]
    margin = best - ranked[1][1]
    if best < .35 or margin < .13:
        return UNKNOWN, 0.0
    return state, min(1.0, margin / .65)


@dataclass(frozen=True)
class SignalSample:
    raw: str
    state: str
    quality: float
    onset: int
    confirmed_at: int
    observed_onset: bool


class SignalHistory:
    def __init__(self, confirmation_frames):
        self.confirmation_frames = confirmation_frames
        self.samples = []
        self.last_raw = UNKNOWN
        self.confirmed_color = UNKNOWN
        self.onset = 0
        self.observed_onset = False

    def update(self, raw, quality):
        index = len(self.samples)
        if raw != self.last_raw:
            self.onset = index
            self.observed_onset = self.last_raw in ("YELLOW", "GREEN") and self.last_raw == self.confirmed_color
            self.last_raw = raw
            self.confirmed_color = UNKNOWN
        confirmed_at = self.onset + self.confirmation_frames - 1
        state = raw if index >= confirmed_at else UNKNOWN
        self.confirmed_color = state
        sample = SignalSample(raw, state, quality, self.onset, confirmed_at, self.observed_onset)
        self.samples.append(sample)
        return sample
