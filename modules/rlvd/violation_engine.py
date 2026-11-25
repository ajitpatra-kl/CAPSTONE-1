"""Directed finite-segment crossings and signal-time decisions."""

from collections import deque
from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np

from .configuration import DIRECTIONS
from .signal_detector import UNKNOWN


@dataclass(frozen=True)
class Thresholds:
    signal_frames: int = 3
    approach_frames: int = 3
    crossing_frames: int = 2
    deadband: float = 3.0
    confidence: float = .35
    anchor_fraction: float = 1.0


@dataclass(frozen=True)
class Observation:
    frame: int
    anchor: tuple
    bbox: tuple
    vehicle_class: str
    confidence: float
    distance: float


@dataclass
class Crossing:
    before: Observation
    after: Observation
    fraction: float
    reliable: bool
    status: str = ""
    reason: str = ""
    signal: str = UNKNOWN
    onset: int = -1

    @property
    def frame(self):
        return self.before.frame + self.fraction * (self.after.frame - self.before.frame)


@dataclass
class Track:
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    approach_hits: int = 0
    armed: bool = False
    pending: Crossing = None
    far_hits: int = 0
    status: str = "Normal"


class CrossingEngine:
    def __init__(self, config, thresholds, signal_history):
        self.start, self.end = (np.asarray(p, dtype=float) for p in config.line)
        self.vector = self.end - self.start
        self.length = hypot(*self.vector)
        self.normal = np.array([-self.vector[1], self.vector[0]]) / self.length
        self.normal *= np.sign(self.normal @ DIRECTIONS[config.approach])
        self.lane = np.asarray(config.lane, np.float32) if config.lane else None
        self.thresholds = thresholds
        self.history = signal_history
        self.tracks = {}
        self.approaching = set()

    def observe(self, track_id, frame, bbox, vehicle_class, confidence, decide=None):
        track = self.tracks.setdefault(track_id, Track())
        x1, y1, x2, y2 = bbox
        anchor = ((x1 + x2) / 2, y1 + (y2 - y1) * self.thresholds.anchor_fraction)
        offset = np.asarray(anchor) - self.start
        distance = float(offset @ self.normal)
        along = float(offset @ self.vector / self.length**2)
        current = Observation(frame, anchor, bbox, vehicle_class, confidence, distance)
        previous = track.trajectory[-1] if track.trajectory else None
        track.trajectory.append(current)
        if track.status in ("Violation", "Legal crossing", "Uncertain crossing"):
            return None
        continuous = previous is not None and previous.frame == frame - 1
        qualified = confidence >= self.thresholds.confidence
        inside = self.lane is None or cv2.pointPolygonTest(self.lane, anchor, False) >= 0
        if distance > self.thresholds.deadband:
            track.pending = None
            track.far_hits = 0
            track.approach_hits = (track.approach_hits * continuous + 1) if qualified and inside else 0
            track.armed = track.approach_hits >= self.thresholds.approach_frames
            track.status = "Normal"
            if track.armed and 0 <= along <= 1 and distance <= max(25, self.length * .35):
                track.status = "Approaching"
                self.approaching.add(track_id)
            return None

        if track.armed and track.pending is None and previous is not None and previous.distance >= 0 > distance:
            fraction = previous.distance / (previous.distance - distance)
            intersection = (1 - fraction) * np.asarray(previous.anchor) + fraction * np.asarray(anchor)
            location = (intersection - self.start) @ self.vector / self.length**2
            if 0 <= location <= 1:
                track.pending = Crossing(previous, current, fraction,
                                         continuous and qualified and previous.confidence >= self.thresholds.confidence
                                         and track.approach_hits >= self.thresholds.approach_frames)
                self.approaching.add(track_id)
            else:
                track.armed = False

        if track.pending is None:
            if track.armed and (not continuous or not qualified):
                track.approach_hits = 0
            return None
        if distance >= 0:
            track.pending = None
            track.far_hits = 0
            return None
        track.pending.reliable &= continuous and qualified
        track.far_hits = (track.far_hits + 1 if continuous else 1) if distance < -self.thresholds.deadband else 0
        if track.far_hits < self.thresholds.crossing_frames:
            return None
        return self.finish(track, decide=decide)

    def finish(self, track, incomplete=False, decide=None):
        event = track.pending
        track.pending = None
        if decide is None:
            self.classify(event, incomplete)
        else:
            decide(event, lambda: self.classify(event, incomplete))
        track.status = event.status
        return event

    def classify(self, event, incomplete=False):
        samples = self.history.samples[event.before.frame:event.after.frame + 1]
        event.signal = samples[-1].state
        event.reason = "Signal state or onset ambiguous at crossing"
        event.status = "Uncertain crossing"
        if incomplete or not event.reliable:
            event.reason = "Insufficient continuous tracking around crossing"
        elif all(sample.state == "RED" and sample.observed_onset and
                 sample.onset == samples[0].onset for sample in samples):
            if event.frame > samples[0].confirmed_at:
                event.status, event.reason = "Violation", "Continuous crossing after observed and confirmed red"
                event.onset = samples[0].onset
        elif all(sample.state in ("GREEN", "YELLOW") for sample in samples):
            event.status, event.reason = "Legal crossing", "Confirmed green or yellow during crossing"
        return event.status
