"""Image-space lane regions bounded by ordered, piecewise-linear markings."""

from bisect import bisect_right
from dataclasses import dataclass
import numpy as np


@dataclass
class Boundary:
    number: int
    kind: str
    points: np.ndarray

    def lateral_at(self, along):
        return float(np.interp(along, self.points[:, 0], self.points[:, 1]))


class LaneGeometry:
    def __init__(self, width, height, lines, axis="Vertical", reverse=False):
        if axis not in ("Vertical", "Horizontal"):
            raise ValueError("Select a valid road orientation.")
        self.vertical = axis == "Vertical"
        self.reverse = reverse
        self.boundaries = []
        for index, line in enumerate(lines, 1):
            points = np.asarray(line["points"], dtype=float)
            kind = line["kind"]
            if (kind not in ("Restricted", "Permitted") or points.ndim != 2
                    or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all() or (points < 0).any()
                    or (points >= [width, height]).any()):
                raise ValueError(f"Boundary {index} needs at least two valid points inside the frame.")
            points = points[:, ::-1] if self.vertical else points
            if points[0, 0] > points[-1, 0]:
                points = points[::-1]
            if np.any(np.diff(points[:, 0]) <= 0):
                raise ValueError(f"Boundary {index} doubles back along the road. Draw its points in order.")
            self.boundaries.append(Boundary(index, kind, points))
        if len(self.boundaries) < 3:
            raise ValueError("Draw at least three boundaries, including the outside edges of the monitored lanes.")
        self.start = max(line.points[0, 0] for line in self.boundaries)
        self.end = min(line.points[-1, 0] for line in self.boundaries)
        if self.end - self.start < 12:
            raise ValueError("Extend the boundaries to cover a common stretch of road.")
        middle = (self.start + self.end) / 2
        self.boundaries.sort(key=lambda line: line.lateral_at(middle))
        knots = {self.start, self.end}
        for line in self.boundaries:
            knots.update(value for value in line.points[:, 0] if self.start < value < self.end)
        for along in knots:
            if any(right.lateral_at(along) - left.lateral_at(along) < 4
                   for left, right in zip(self.boundaries, self.boundaries[1:])):
                raise ValueError("Boundaries intersect or leave a lane less than four pixels wide.")

    def axes(self, point):
        return (point[1], point[0]) if self.vertical else tuple(point)

    def image_point(self, along, lateral):
        return (lateral, along) if self.vertical else (along, lateral)

    def lane_number(self, region):
        return len(self.boundaries) - 1 - region if self.reverse else region + 1

    def locate(self, point, margin=0):
        along, lateral = self.axes(point)
        if not (self.start <= along <= self.end):
            return None
        edges = [line.lateral_at(along) for line in self.boundaries]
        slot = bisect_right(edges, lateral) - 1
        if slot < 0 or slot >= len(edges) - 1:
            return None
        width = edges[slot + 1] - edges[slot]
        if min(lateral - edges[slot], edges[slot + 1] - lateral) <= min(margin, width * .2):
            return None
        return slot

    def crossing(self, before, after, boundary):
        """Return the fraction along the observed sweep at one unambiguous intersection."""
        start, left = self.axes(before)
        end, right = self.axes(after)
        if not all(self.start <= value <= self.end for value in (start, end)):
            return None
        fractions = [0.0, 1.0]
        if end != start:
            fractions.extend((knot - start) / (end - start) for knot in boundary.points[:, 0]
                             if min(start, end) < knot < max(start, end))
        knots = sorted(fractions)
        distances = [left + t * (right - left) - boundary.lateral_at(start + t * (end-start))
                     for t in knots]
        roots = set()
        for t0, t1, d0, d1 in zip(knots, knots[1:], distances, distances[1:]):
            if d0 == d1 == 0:
                return None
            if d0 * d1 <= 0:
                roots.add(t0 + (t1-t0) * d0 / (d0-d1))
        return roots.pop() if len(roots) == 1 and distances[0] * distances[-1] <= 0 else None

