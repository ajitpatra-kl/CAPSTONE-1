"""Read compact observations scoped to one completed unified session."""

import csv
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    frame: int
    timestamp: float
    timing_valid: bool
    anchor: tuple
    bbox: tuple
    confidence: float
    speed: float | None
    missing_frames: int
    lane: int | None
    stop_distance: float | None


@dataclass(frozen=True)
class Candidate:
    event_id: str
    kind: str
    track_id: int
    frame: int
    timestamp: float
    original: str
    evidence_frames: tuple
    metadata: dict
    speed_limit: float | None
    lanes: dict

    @classmethod
    def from_record(cls, row, session):
        if (row["session_id"], row["video_id"]) != (session["session_id"], session["video_id"]):
            raise ValueError("Candidate belongs to another session or video.")
        return cls(row["event_id"], row["violation_type"], row["track_id"], row["frame_number"],
                   row["timestamp"], row["verification_status"], tuple(row["evidence_frames"]),
                   row["metadata"], row["speed_limit"], row["lane_information"] or {})


def availability(session, records):
    if not records:
        return "This session has no stored candidates to verify."
    track_path = Path(session["tracks_csv"])
    if not track_path.is_file():
        return "This session is missing its stored track observations."
    if any(row["violation_type"] in {"red_light", "lane_change"} for row in records):
        if not track_path.with_name("tcvv-context-v1.jsonl").is_file():
            return ("This session lacks the full signal history or lane-overlap observations required "
                    "for TCVV. These observations are captured in new unified processing sessions.")
    return None


class TemporalContext:
    def __init__(self, histories, frames=None):
        self.histories = {tid: sorted(rows, key=lambda obs: obs.frame) for tid, rows in histories.items()}
        self.indices = {tid: [obs.frame for obs in rows] for tid, rows in self.histories.items()}
        self.frames = frames or {}
        for rows in self.indices.values():
            if any(a == b for a, b in zip(rows, rows[1:])):
                raise ValueError("Track observations contain duplicate frames.")

    def window(self, track, first, last):
        indices = self.indices.get(track, ())
        return self.histories.get(track, [])[bisect_left(indices, first):bisect_right(indices, last)]

    @classmethod
    def load(cls, session, candidates):
        required_tracks = {row.track_id for row in candidates}
        histories = defaultdict(list)
        track_path = Path(session["tracks_csv"])
        with track_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["session_id"] != session["session_id"]:
                    raise ValueError("Track export belongs to another session.")
                track = int(row["track_id"])
                if track not in required_tracks:
                    continue
                speed = float(row["speed_kmh"]) if row["speed_kmh"] else None
                histories[track].append(Observation(
                    int(row["frame_number"]), float(row["timestamp"]), row["timing_valid"] == "True",
                    tuple(json.loads(row["bottom_center"])), tuple(json.loads(row["bbox"])),
                    float(row["confidence"]), speed if speed is not None and isfinite(speed) else None,
                    int(row["missing_frames"]), int(row["lane_id"]) if row["lane_id"] else None,
                    float(row["stop_line_distance"]) if row["stop_line_distance"] else None,
                ))
        frames = {}
        if any(row.kind in {"red_light", "lane_change"} for row in candidates):
            with track_path.with_name("tcvv-context-v1.jsonl").open(encoding="utf-8") as file:
                header = json.loads(file.readline())
                if any(header[key] != session[key] for key in ("session_id", "video_id")):
                    raise ValueError("Temporal sidecar belongs to another session or video.")
                for index, line in enumerate(file):
                    frame = json.loads(line)
                    if frame["frame"] != index:
                        raise ValueError("Temporal sidecar has missing or unordered frames.")
                    frames[index] = frame
            if len(frames) != session["frames"]:
                raise ValueError("Temporal sidecar is incomplete.")
        return cls(histories, frames)


def contiguous(rows, fps, *, timed=True):
    return bool(rows) and all(
        b.frame == a.frame + 1 and not b.missing_frames
        and (not timed or 0 < b.timestamp - a.timestamp <= 2 / fps)
        for a, b in zip(rows, rows[1:])
    ) and (not timed or all(row.timing_valid and isfinite(row.timestamp) for row in rows))
