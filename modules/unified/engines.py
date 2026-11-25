"""Adapters from session observations to the existing violation rules."""

from dataclasses import asdict

from modules.lcvd.evidence_generator import lane_information
from modules.lcvd.lane_change_detector import LaneChangeDetector, Observation
from modules.lcvd.video_processor import draw_geometry as draw_lanes, overlapping_tracks
from modules.rlvd.signal_detector import SignalHistory, classify_signal
from modules.rlvd.video_processor import draw_geometry as draw_signal
from modules.rlvd.violation_engine import CrossingEngine
from modules.speed_detection.speed_session import SpeedSession


class Overspeed:
    kind = "overspeed"

    def __init__(self, options, dataset_config, path, resolution):
        self.motion = SpeedSession(options, dataset_config)
        self.motion.initialize_speed(path, resolution)
        self.calibration = self.motion._calibration
        self.status = self.motion._calibration_status
        self.available = self.motion._calibrated or bool(self.motion._estimated_speed_mode)
        self.speed_label = "" if self.status.startswith("Verified") else "Est. "
        self.previous = {}

    def prepare(self, active, timing_valid):
        speeds = self.motion.update_speeds(active, self.previous, timing_valid)
        self.previous = active
        return speeds

    def observe(self, observation, track, submit):
        def confirm(record, rule, inputs):
            submit(
                observation, self.kind, lambda: "Confirmed violation" if rule() else "Uncertain overspeed",
                speed=record.speed_kmh, speed_limit=record.speed_limit_kmh,
                metadata={**inputs, "calibration": self.status, "excess_kmh": record.excess_kmh},
            )

        self.motion._violation_detector.update(
            track, observation.estimated_speed, observation.frame_number, observation.timestamp,
            self.motion._calibrated, confirm=confirm,
        )


class RedLight:
    kind = "red_light"

    def __init__(self, config, thresholds, fps):
        self.config = config
        self.timeout = max(fps, thresholds.crossing_frames)
        self.history = SignalHistory(thresholds.signal_frames)
        self.engine = CrossingEngine(config, thresholds, self.history)
        self.times = []
        self.sample = None

    def prepare(self, image, timestamp):
        self.sample = self.history.update(*classify_signal(image, self.config.roi, self.config.lamp_layout))
        self.times.append(timestamp)

    def observe(self, observation, submit, snapshot):
        observation.signal_context = asdict(self.sample)

        def decide(event, rule):
            observation.stop_line_distance = self.engine.tracks[observation.track_id].trajectory[-1].distance
            self.decide(observation, event, rule, submit, snapshot)

        self.engine.observe(observation.track_id, observation.frame_number, observation.bbox,
                            observation.vehicle_class, observation.confidence, decide=decide)
        track = self.engine.tracks[observation.track_id]
        observation.stop_line_distance = track.trajectory[-1].distance
        if track.pending:
            snapshot(observation.track_id, self.kind, track.pending.before.frame, track.pending.after.frame)
            if observation.frame_number - track.pending.after.frame > self.timeout:
                self.finish_track(track, observation, submit, snapshot)
        else:
            snapshot(observation.track_id, self.kind)

    def decide(self, observation, event, rule, submit, snapshot, incomplete=False):
        before, after = event.before.frame, event.after.frame
        snapshot(observation.track_id, self.kind, before, after)
        samples = [asdict(sample) for sample in self.history.samples[before:after + 1]]
        timestamp = self.times[before] + event.fraction * (self.times[after] - self.times[before])
        signal = {"samples": samples, "crossing_frame": float(event.frame)}

        def apply_rule():
            status = rule()
            signal.update(state=event.signal, red_onset_frame=event.onset, reason=event.reason)
            return "Confirmed violation" if status == "Violation" else status

        submit(
            observation, self.kind, apply_rule, frame_number=after, timestamp=float(timestamp),
            bbox=event.after.bbox, confidence=event.after.confidence, vehicle_class=event.after.vehicle_class,
            signal_information=signal, evidence_frames=[before, after],
            metadata={"continuous_crossing": event.reliable, "incomplete": incomplete,
                      "before_anchor": event.before.anchor, "after_anchor": event.after.anchor},
        )

    def finish_track(self, track, observation, submit, snapshot):
        self.engine.finish(track, incomplete=True, decide=lambda event, rule: self.decide(
            observation, event, rule, submit, snapshot, incomplete=True,
        ))
        snapshot(observation.track_id, self.kind)

    def expire(self, index, contexts, submit, snapshot, final=False):
        for track_id, track in self.engine.tracks.items():
            if track.pending and (final or index - track.trajectory[-1].frame > 1):
                self.finish_track(track, contexts[track_id].observations[-1], submit, snapshot)

    def draw(self, canvas, state=None):
        draw_signal(canvas, self.config, state or self.sample.state)


class LaneChange:
    kind = "lane_change"

    def __init__(self, config, direction=None):
        if direction not in (None, "increasing", "decreasing"):
            raise ValueError("Select a supported lane monitoring direction.")
        self.direction = direction
        self.fps = config.fps
        self.geometry = config.geometry()
        self.engine = LaneChangeDetector(self.geometry, config.settings)
        self.ambiguous = set()

    def prepare(self, active):
        self.ambiguous = overlapping_tracks(active)

    def observe(self, observation, submit, snapshot):
        tid = observation.track_id
        local = Observation(
            observation.frame_number, observation.frame_number / self.fps, observation.bbox,
            observation.vehicle_class, observation.confidence,
            "Overlapping tracks near the transition" if tid in self.ambiguous else "",
        )
        event = self.engine.observe(tid, local)
        state = self.engine.tracks[tid]
        if state.stable is not None:
            observation.lane_id = self.geometry.lane_number(state.stable)
        sweep = event.sweep if event else (state.pending.sweep if state.pending else state.sweep)
        if sweep:
            snapshot(tid, self.kind, sweep.before.frame, sweep.after.frame)
        if event:
            self.decide(observation, event, submit, snapshot, observation.frame_number)
        if not sweep or event:
            snapshot(tid, self.kind)

    def decide(self, observation, event, submit, snapshot, index):
        crossing = event.observation
        frames = [event.sweep.before.frame, crossing.frame] if event.sweep else [crossing.frame]
        snapshot(observation.track_id, self.kind, *frames)
        direction = None
        if event.sweep:
            before, _ = self.geometry.axes(event.sweep.before.anchor)
            after, _ = self.geometry.axes(event.sweep.after.anchor)
            if after != before:
                direction = "increasing" if after > before else "decreasing"

        def apply_rule():
            if event.status == "Confirmed violation" and self.direction and direction != self.direction:
                return "Outside monitoring direction" if direction else "Uncertain direction"
            return event.status

        submit(
            observation, self.kind, apply_rule,
            frame_number=crossing.frame,
            timestamp=event.sweep.timestamp if event.sweep else crossing.timestamp,
            bbox=crossing.bbox, vehicle_class=crossing.vehicle, confidence=crossing.confidence,
            evidence_frames=frames,
            lane_information={**lane_information(self.geometry, event), "direction": direction},
            metadata={"reason": event.reason, "confirmation_frame": index, "stable_hits": event.hits},
        )

    def expire(self, index, contexts, submit, snapshot, final=False):
        for tid, event in self.engine.expire(index, final=final):
            self.decide(contexts[tid].observations[-1], event, submit, snapshot, index)
            snapshot(tid, self.kind)

    def draw(self, canvas):
        draw_lanes(canvas, self.geometry)
