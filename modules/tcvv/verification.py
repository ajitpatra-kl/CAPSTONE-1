"""Deterministic TCVV policies applied to the shared System A candidate population."""

from dataclasses import asdict, dataclass
from math import hypot, isfinite
from statistics import median

import cv2
import numpy as np

from modules.lcvd.configuration import Configuration as LaneConfiguration, Settings
from modules.rlvd.configuration import Configuration as RedConfiguration, DIRECTIONS
from modules.rlvd.violation_engine import Thresholds
from .context import contiguous


VERSION = "1.0.0"


@dataclass(frozen=True)
class Policy:
    speed_min_samples: int = 3
    speed_max_relative_deviation: float = .30
    signal_guard_frames: int = 1
    signal_min_quality: float = .20
    crossing_search_frames: int = 60


@dataclass
class Finding:
    decision: str
    reason_codes: list
    explanation: str
    observations: list
    context: dict
    limitations: list


def finding(decision, code, explanation, rows, **context):
    return Finding(decision, [code], explanation, [asdict(row) for row in rows], context,
                   [explanation] if decision == "UNCERTAIN" else [])


class Verifier:
    def __init__(self, session, context, policy=Policy()):
        self.configuration = session["configuration"]
        self.fps = session["nominal_fps"]
        self.calibration = session.get("calibration") or ""
        self.context = context
        self.policy = policy
        self.red = None
        self.lane = None
        if self.configuration.get("red_light"):
            self.red = RedConfiguration(**self.configuration["red_light"])
            self.red.validate()
            self.red_thresholds = Thresholds(**self.configuration["red_thresholds"])
        if self.configuration.get("lane_change"):
            options = self.configuration["lane_change"].copy()
            self.lane_settings = Settings(**options.pop("settings"))
            self.lane = LaneConfiguration(**options, settings=self.lane_settings).geometry()

    def evaluate(self, event):
        method = {"overspeed": self.speed, "red_light": self.red_light, "lane_change": self.lane_change}[event.kind]
        return asdict(method(event))

    def speed(self, event):
        options = self.configuration.get("speed") or {}
        duration = options.get("violation_duration")
        limit = event.speed_limit
        if duration is None or limit is None or not isfinite(limit) or limit <= 0:
            return finding("UNCERTAIN", "MISSING_SPEED_CONFIGURATION", "The speed limit or required duration is unavailable.", [])
        length = max(round(duration * self.fps) + 3, event.metadata.get("consecutive_frames", 0))
        rows = self.context.window(event.track_id, event.frame - length + 1, event.frame)
        details = dict(calibration=self.calibration, speed_limit_kmh=limit, required_seconds=duration,
                       smoothing_window=options.get("smoothing_window"))
        if not self.calibration.startswith("Verified calibration:"):
            assumed = self.calibration.startswith(("Estimated speeds", "Reference calibration"))
            return finding("UNCERTAIN", "ASSUMED_CALIBRATION" if assumed else "UNVERIFIED_CALIBRATION",
                           "The speeds rely on calibration assumptions." if assumed else
                           "Verified camera calibration is unavailable.", rows, **details)

        current = []
        for row in rows:
            if (not row.timing_valid or not isfinite(row.timestamp)
                    or row.speed is None or not isfinite(row.speed) or row.speed < 0):
                current = []
                continue
            if current and not contiguous([current[-1], row], self.fps):
                current = []
            current.append(row)
        valid = current if current and current[-1].frame == event.frame else []
        if len(valid) < self.policy.speed_min_samples:
            return finding("UNCERTAIN", "INSUFFICIENT_VALID_SPEED", "Too few consecutive valid speed measurements reach the candidate frame.", rows, **details)
        above = []
        for row in reversed(valid):
            if row.speed <= limit:
                break
            above.append(row)
        above.reverse()
        seconds = above[-1].timestamp - above[0].timestamp if above else 0
        details.update(exceedance_seconds=seconds, consecutive_above=len(above))
        if len(above) >= self.policy.speed_min_samples and seconds + 1e-9 >= duration:
            center = median(row.speed for row in above)
            if max(abs(row.speed - center) for row in above) > center * self.policy.speed_max_relative_deviation:
                return finding("UNCERTAIN", "INCONSISTENT_SPEED", "Speed changes exceed the consistency tolerance.", above, **details)
            return finding("CONFIRMED", "SUSTAINED_VERIFIED_EXCEEDANCE", "Consecutive calibrated speed estimates exceed the limit for the required duration.", above, **details)
        if len(valid) > len(above):
            return finding("REJECTED", "OBSERVED_BELOW_LIMIT", "A valid measurement at or below the limit interrupts the required exceedance.", valid, **details)
        return finding("UNCERTAIN", "INCOMPLETE_SPEED_WINDOW", "The continuous speed window is too short to establish the required duration.", rows, **details)

    def red_light(self, event):
        if not self.red or len(event.evidence_frames) != 2:
            return finding("UNCERTAIN", "MISSING_STOP_LINE_BRACKET", "A stop-line crossing bracket is unavailable.", [])
        before, after = event.evidence_frames
        t = self.red_thresholds
        margin = self.policy.crossing_search_frames
        rows = self.context.window(event.track_id, before - margin, after + margin)
        by_frame = {row.frame: row for row in rows}
        pair = [by_frame[index] for index in (before, after) if index in by_frame]
        start, end = self.red.line
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        ux, uy = DIRECTIONS[self.red.approach]
        orientation = 1 if -dy * ux + dx * uy > 0 else -1

        def anchor(row):
            x1, y1, x2, y2 = row.bbox
            return (x1 + x2) / 2, y1 + (y2 - y1) * t.anchor_fraction

        def distance(row):
            x, y = anchor(row)
            return orientation * (dx * (y - start[1]) - dy * (x - start[0])) / length

        def qualified(window):
            return contiguous(window, self.fps) and all(row.confidence >= t.confidence for row in window)

        if len(pair) != 2 or not qualified(pair):
            return finding("UNCERTAIN", "INTERRUPTED_CROSSING", "The crossing bracket has missing, low-confidence or invalidly timed track observations.", pair)
        a, b = map(distance, pair)
        trajectory = self.context.window(event.track_id, before - t.approach_frames + 1, after + t.crossing_frames - 1)
        if not a >= 0 > b:
            if a < 0 <= b:
                return finding("REJECTED", "OPPOSITE_CROSSING_DIRECTION", "The track crosses toward the configured approach side.", pair)
            if not qualified(trajectory) or len(trajectory) != t.approach_frames + t.crossing_frames:
                return finding("UNCERTAIN", "INCOMPLETE_STOP_LINE_TRAJECTORY", "Tracking ends before the stop-line movement can be resolved.", trajectory)
            if all(distance(row) >= 0 for row in trajectory) or all(distance(row) < 0 for row in trajectory):
                return finding("REJECTED", "NO_DIRECTED_CROSSING", "The observed trajectory stays on one side of the stop line.", trajectory)
            return finding("UNCERTAIN", "CROSSING_OUTSIDE_BRACKET", "The observed side change falls outside the candidate crossing bracket.", trajectory)

        fraction = a / (a - b)
        (x0, y0), (x1, y1) = map(anchor, pair)
        x, y = x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)
        if not 0 <= ((x - start[0]) * dx + (y - start[1]) * dy) / length**2 <= 1:
            return finding("REJECTED", "OUTSIDE_STOP_LINE_SEGMENT", "The trajectory crosses the line extension outside the configured segment.", pair)

        # Walk outward within the continuous approach and departure. Samples in
        # the line deadband establish continuity but do not qualify either side.
        approach = [pair[0]]
        departure = [pair[1]]
        for chain, step, side, count in ((approach, -1, 1, t.approach_frames),
                                          (departure, 1, -1, t.crossing_frames)):
            hits = int(side * distance(chain[0]) > t.deadband)
            while hits < count:
                adjacent = by_frame.get(chain[-1].frame + step)
                if adjacent is None or not qualified(sorted([chain[-1], adjacent], key=lambda row: row.frame)):
                    break
                signed = side * distance(adjacent)
                if signed < 0:
                    break
                chain.append(adjacent)
                hits = hits + 1 if signed > t.deadband else 0
            if hits < count:
                return finding("UNCERTAIN", "UNSTABLE_STOP_LINE_TRAJECTORY", "The track does not establish a continuous approach and departure past the stop line.", approach[::-1] + departure)
        support = approach[::-1] + departure
        if self.red.lane:
            polygon = np.asarray(self.red.lane, np.float32)
            if any(cv2.pointPolygonTest(polygon, anchor(row), False) < 0 for row in approach
                   if distance(row) > t.deadband):
                return finding("UNCERTAIN", "APPROACH_REGION_AMBIGUOUS", "The approach samples do not stay within the configured approach region.", support)

        guard = self.policy.signal_guard_frames
        samples = [self.context.frames.get(index) for index in range(before - guard, after + guard + 1)]
        crossing = pair[0].timestamp + fraction * (pair[1].timestamp - pair[0].timestamp)
        details = dict(crossing_timestamp=crossing, signal_frames=samples,
                       stop_line=self.red.line, approach=self.red.approach, signal_roi=self.red.roi)
        if any(not frame or not frame["timing_valid"] or not frame["signal"] for frame in samples):
            return finding("UNCERTAIN", "MISSING_SIGNAL_OBSERVATIONS", "Signal timing or observations are missing around the crossing.", support, **details)
        signals = [frame["signal"] for frame in samples]
        states = {signal["state"] for signal in signals}
        if len(states) != 1 or any(signal["raw"] != signal["state"] or signal["quality"] < self.policy.signal_min_quality
                                  for signal in signals):
            return finding("UNCERTAIN", "SIGNAL_AMBIGUOUS_AT_CROSSING", "Signal samples disagree or have low quality near the crossing.", support, **details)
        state = signals[0]["state"]
        if state in {"GREEN", "YELLOW"}:
            return finding("REJECTED", "NON_RED_AT_CROSSING", "The directed crossing occurs during a stable non-red signal.", support, **details)
        onset = self.context.frames.get(signals[0]["onset"])
        if (state == "RED" and onset and onset["timing_valid"] and onset["timestamp"] < crossing
                and all(signal["observed_onset"] and signal["onset"] == signals[0]["onset"] for signal in signals)):
            return finding("CONFIRMED", "DIRECTED_CROSSING_AFTER_RED_ONSET", "A continuous directed stop-line crossing follows the observed onset of a stable red signal.", support,
                           red_onset_timestamp=onset["timestamp"], **details)
        return finding("UNCERTAIN", "RED_ONSET_UNVERIFIED", "The red onset cannot be established before the crossing.", support, **details)

    def lane_change(self, event):
        if not self.lane:
            return finding("UNCERTAIN", "MISSING_LANE_GEOMETRY", "Lane geometry is unavailable.", [])
        geometry = self.lane
        settings = self.lane_settings
        regions = {geometry.lane_number(region): region for region in range(len(geometry.boundaries) - 1)}
        source, target = (regions.get(event.lanes.get(field)) for field in ("source_lane", "destination_lane"))
        details = dict(event.lanes)
        if source is None or target is None or abs(source - target) != 1:
            return finding("UNCERTAIN", "AMBIGUOUS_LANE_ASSIGNMENT", "The candidate lacks adjacent source and destination lane assignments.", [], **details)

        before = min(event.evidence_frames, default=event.frame)
        end = min(max(event.frame, event.metadata.get("confirmation_frame", event.frame)),
                  event.frame + max(45, 3 * settings.stable_frames))
        rows = self.context.window(event.track_id, before - self.policy.crossing_search_frames, end)
        edge = geometry.boundaries[max(source, target)]
        neighbor = geometry.boundaries[min(source, target) if source < target else max(source, target) + 1]

        def usable(row):
            frame = self.context.frames.get(row.frame)
            return (frame and frame["lane_overlap_tracks"] is not None
                    and event.track_id not in frame["lane_overlap_tracks"] and row.confidence >= settings.confidence
                    and geometry.locate(row.anchor, -1) is not None)

        source_run = []
        stable_source = []
        destination_run = []
        sweep = None
        chain = []
        for row in rows:
            reliable = usable(row)
            if not reliable or (chain and not contiguous([chain[-1], row], self.fps, timed=False)):
                chain, source_run, stable_source, destination_run, sweep = [], [], [], [], None
            if not reliable:
                continue
            region = geometry.locate(row.anchor, settings.margin)
            if region == source:
                sweep = None
                source_run = (source_run + [row])[-settings.stable_frames:]
                if len(source_run) == settings.stable_frames:
                    stable_source = source_run
                destination_run = []
            else:
                source_run = []
                if region == target:
                    destination_run.append(row)
                else:
                    destination_run = []
                    if region is not None:
                        stable_source, sweep = [], None
            if chain and stable_source and row.frame == event.frame:
                previous = chain[-1]
                a, x = geometry.axes(previous.anchor)
                b, y = geometry.axes(row.anchor)
                d0, d1 = x - edge.lateral_at(a), y - edge.lateral_at(b)
                if source > target:
                    d0, d1 = -d0, -d1
                width = abs(neighbor.lateral_at(b) - edge.lateral_at(b))
                if d0 <= 0 <= d1 and d1 > d0 and d1 - d0 <= settings.max_jump * width:
                    fraction = geometry.crossing(previous.anchor, row.anchor, edge)
                    if fraction is not None:
                        sweep = (previous, row)
            chain.append(row)
            if len(destination_run) >= settings.stable_frames and sweep:
                first, last = stable_source[0], destination_run[-1]
                delta = geometry.axes(last.anchor)[0] - geometry.axes(first.anchor)[0]
                direction = "increasing" if delta > 0 else "decreasing" if delta < 0 else None
                details.update(boundary=edge.number, boundary_type=edge.kind, direction=direction,
                               crossing_frames=[sample.frame for sample in sweep],
                               destination_stable_frames=len(destination_run))
                support = [sample for sample in chain if sample.frame >= first.frame]
                if edge.kind == "Permitted":
                    return finding("REJECTED", "PERMITTED_BOUNDARY", "The sustained transition crosses a permitted boundary.", support, **details)
                if not direction:
                    return finding("UNCERTAIN", "UNKNOWN_TRAVEL_DIRECTION", "The transition has no established longitudinal travel direction.", support, **details)
                monitor = self.configuration.get("lane_direction")
                if monitor and direction != monitor:
                    return finding("REJECTED", "OUTSIDE_MONITORED_DIRECTION", "The sustained transition travels outside the selected monitoring direction.", support, **details)
                return finding("CONFIRMED", "SUSTAINED_RESTRICTED_TRANSITION", "The continuous trajectory crosses a restricted boundary and stabilizes in the destination lane.", support, **details)

        observed = [row for row in rows if row.frame >= before - settings.stable_frames + 1]
        if (observed and observed[0].frame <= before and observed[-1].frame >= event.frame + settings.stable_frames
                and contiguous(observed, self.fps, timed=False) and all(usable(row) for row in observed)
                and not any(geometry.locate(row.anchor, 0) == target for row in observed)
                and all(geometry.locate(row.anchor, settings.margin) == source for row in observed[-settings.stable_frames:])):
            return finding("REJECTED", "NO_LANE_TRANSITION", "The observed anchor stays out of the destination lane and stabilizes in the source lane.", observed, **details)
        return finding("UNCERTAIN", "INCOMPLETE_LANE_TRANSITION", "A continuous boundary crossing with stable source and destination assignments could not be established.", rows, **details)
