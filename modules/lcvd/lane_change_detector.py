"""Per-track transition confirmation using stable contact points and observed crossings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    frame: int
    timestamp: float
    bbox: tuple
    vehicle: str
    confidence: float
    ambiguity: str = ""

    @property
    def anchor(self):
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class Sweep:
    before: Observation
    after: Observation
    boundary: object
    fraction: float

    @property
    def timestamp(self):
        return self.before.timestamp + self.fraction * (self.after.timestamp - self.before.timestamp)


@dataclass
class Transition:
    source: int
    target: int
    first: Observation
    sweep: Sweep = None
    reason: str = ""
    hits: int = 1

    @property
    def observation(self):
        return self.sweep.after if self.sweep else self.first

    @property
    def status(self):
        if self.reason or self.sweep is None:
            return "Uncertain"
        return "Confirmed violation" if self.sweep.boundary.kind == "Restricted" else "Permitted"


@dataclass
class Track:
    stable: int = None
    last: Observation = None
    seed: int = None
    seed_hits: int = 0
    pending: Transition = None
    sweep: Sweep = None
    interrupted: str = ""
    latest: Transition = None
    annotated_until: int = -1


class LaneChangeDetector:
    def __init__(self, geometry, settings):
        self.geometry = geometry
        self.settings = settings
        self.tracks = {}

    def observe(self, track_id, observation):
        state = self.tracks.setdefault(track_id, Track())
        previous = state.last
        state.last = observation
        settings = self.settings
        gap = observation.frame - previous.frame - 1 if previous else 0
        qualified = observation.confidence >= settings.confidence and not observation.ambiguity
        region = self.geometry.locate(observation.anchor, settings.margin) if qualified else None
        if gap > settings.max_gap:
            state.stable = state.seed = None
            state.seed_hits = 0
            state.pending = state.sweep = None
            state.interrupted = ""
        if state.stable is None:
            state.seed_hits = state.seed_hits + 1 if region is not None and region == state.seed and not gap else int(region is not None)
            state.seed = region
            if state.seed_hits >= settings.stable_frames:
                state.stable = region
            return None
        if gap or not qualified:
            state.interrupted = ("Tracking gap near the transition" if gap else
                                 observation.ambiguity or "Low detection confidence near the transition")
            state.sweep = None
            if state.pending:
                state.pending.reason = state.interrupted
                state.pending.hits = 0
        if region == state.stable:
            state.pending = state.sweep = None
            state.interrupted = ""
            return None

        if qualified and region is None and self.geometry.locate(observation.anchor, -1) is None:
            state.interrupted = "Track left the monitored lane region"
            state.sweep = None
            if state.pending:
                state.pending.hits = 0
                state.pending.reason = state.interrupted
            return None

        if (previous and not gap and qualified and not previous.ambiguity
                and previous.confidence >= settings.confidence):
            source = state.stable
            for edge in (source, source + 1):
                if edge == 0 or edge == len(self.geometry.boundaries) - 1:
                    continue
                boundary = self.geometry.boundaries[edge]
                a, x = self.geometry.axes(previous.anchor)
                b, y = self.geometry.axes(observation.anchor)
                before_side = x - boundary.lateral_at(a)
                after_side = y - boundary.lateral_at(b)
                forward = before_side <= 0 <= after_side if edge == source + 1 else before_side >= 0 >= after_side
                if not forward:
                    continue
                fraction = self.geometry.crossing(previous.anchor, observation.anchor, boundary)
                if fraction is None:
                    continue
                state.sweep = Sweep(previous, observation, boundary, fraction)
                neighbor = self.geometry.boundaries[source if edge == source + 1 else source + 1]
                width = abs(neighbor.lateral_at(b) - boundary.lateral_at(b))
                if abs(after_side - before_side) > settings.max_jump * width:
                    state.interrupted = "Abrupt lateral movement near the boundary"

        if region is None:
            if state.pending:
                state.pending.hits = 0
            return None

        if state.pending is None or state.pending.target != region:
            neighbor = abs(region - state.stable) == 1
            edge = self.geometry.boundaries[max(region, state.stable)] if neighbor else None
            if not neighbor:
                state.interrupted = "More than one lane was skipped"
                state.sweep = None
            elif state.pending is not None:
                state.interrupted = "Destination lane changed before confirmation"
            sweep = state.sweep if state.sweep and state.sweep.boundary is edge else None
            reason = state.interrupted
            if not sweep and not reason:
                reason = "Boundary crossing was not observed"
            state.pending = Transition(state.stable, region, observation, sweep, reason)
        else:
            state.pending.hits += 1
        if state.pending.hits >= settings.stable_frames:
            event = state.pending
            state.stable = region
            state.pending = state.sweep = None
            state.interrupted = ""
            state.latest = event
            return event
        return None

    def expire(self, frame, final=False):
        for track_id, state in self.tracks.items():
            if state.pending and (final or frame - state.last.frame > self.settings.max_gap
                                  or frame - state.pending.first.frame > max(45, 3*self.settings.stable_frames)):
                event = state.pending
                event.reason = event.reason or "Destination lane did not remain stable before tracking ended"
                state.pending = state.sweep = None
                state.interrupted = event.reason
                yield track_id, event
