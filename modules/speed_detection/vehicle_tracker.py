"""
modules/speed_detection/vehicle_tracker.py

Multi-object vehicle tracking using ByteTrack algorithm.
ByteTrack is integrated into ultralytics; this module wraps it cleanly
with trajectory history management.

Tracking reference point: bottom-centre of bounding box
(where the vehicle contacts the ground plane — best for homography projection).
"""

import numpy as np
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .vehicle_detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State maintained for one tracked vehicle."""
    track_id: int
    class_id: int
    class_name: str
    
    # Recent detections (sliding window)
    bboxes: deque = field(default_factory=lambda: deque(maxlen=60))
    bottom_centers_image: deque = field(default_factory=lambda: deque(maxlen=60))  # (u,v) pixels
    bottom_centers_world: deque = field(default_factory=lambda: deque(maxlen=60))  # (X,Y) metres
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))            # seconds
    
    # Speed history
    speed_samples_kmh: deque = field(default_factory=lambda: deque(maxlen=30))
    smoothed_speed_kmh: float = 0.0
    
    # Detection count
    hit_count: int = 0
    miss_count: int = 0
    is_active: bool = True
    
    # Violation tracking
    frames_above_limit: int = 0
    violation_confirmed: bool = False
    violation_recorded: bool = False
    
    @property
    def age_frames(self) -> int:
        return self.hit_count + self.miss_count
    
    @property
    def latest_bbox(self) -> Optional[Tuple]:
        return self.bboxes[-1] if self.bboxes else None
    
    @property
    def latest_bottom_center_image(self) -> Optional[Tuple]:
        return self.bottom_centers_image[-1] if self.bottom_centers_image else None


class VehicleTracker:
    """
    ByteTrack-based multi-object tracker for vehicles.
    
    Uses ultralytics' built-in ByteTrack implementation for robust tracking.
    Maintains per-track trajectory history for speed estimation.
    
    The bottom-centre of each bounding box is used as the road-plane reference
    point, which minimises errors when transforming via homography.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_track_length: int = 5,
        max_miss_frames: int = 30,
        smoothing_window_size: int = None,  # accepted but unused (for compat)
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_track_length = min_track_length
        self.max_miss_frames = max_miss_frames

        self.tracks: Dict[int, TrackState] = {}
        self._frame_count: int = 0
        self._tracker = None
        self._init_bytetrack()

    @property
    def name(self):
        return "ByteTrack" if self._tracker is not None else "IoU fallback"

    def _init_bytetrack(self):
        from types import SimpleNamespace

        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except (ImportError, OSError) as exc:
            logger.warning("ByteTrack unavailable: %s. Using IoU fallback.", exc)
            self._tracker = None
            return
        args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=True,
        )
        class SessionByteTracker(BYTETracker):
            @staticmethod
            def reset_id():
                # Ultralytics shares its ID counter across tracker instances.
                # Starting another video must leave active trackers' IDs intact.
                pass

        self._tracker = SessionByteTracker(args)
        self._local_ids = {}

    def update(self, detections, frame, timestamp_sec, calibration=None):
        self._frame_count += 1
        if self._tracker is None:
            tracks = self._update_iou_tracker(detections, frame, timestamp_sec, calibration)
            return {tid: state for tid, state in tracks.items() if state.miss_count == 0}

        from ultralytics.engine.results import Boxes

        rows = np.asarray(
            [[*det.bbox, det.confidence, det.class_id] for det in detections],
            dtype=np.float32,
        ).reshape(-1, 6)
        tracks = self._tracker.update(Boxes(rows, frame.shape[:2]), frame)
        active = {}
        for row in tracks:
            tracker_id = int(row[4])
            track_id = self._local_ids.setdefault(tracker_id, len(self._local_ids) + 1)
            detection = detections[int(row[7])]
            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id, detection.class_id, detection.class_name)
                self.tracks[track_id] = state
            if state.miss_count:
                state.bottom_centers_world.clear()
                state.timestamps.clear()
            x, y = detection.bottom_center
            state.bboxes.append(detection.bbox)
            state.bottom_centers_image.append((x, y))
            state.timestamps.append(timestamp_sec)
            if calibration is not None and calibration.is_valid():
                state.bottom_centers_world.append(calibration.transform(x, y))
            state.hit_count += 1
            state.miss_count = 0
            state.is_active = True
            active[track_id] = state

        for track_id, state in self.tracks.items():
            if state.is_active and track_id not in active:
                state.miss_count += 1
                if state.miss_count > self.max_miss_frames:
                    state.is_active = False
        return active

    def _update_iou_tracker(
        self, detections, frame, timestamp_sec, calibration
    ):
        """
        Simple IoU-based tracker as fallback when ByteTrack unavailable.
        Assigns IDs by matching bboxes with previous frame bboxes via IoU.
        """
        # Match current detections to existing tracks
        active_tracks = {tid: s for tid, s in self.tracks.items() if s.is_active}
        matched_det_ids = set()
        matched_track_ids = set()
        
        iou_matrix = np.zeros((len(active_tracks), len(detections)))
        track_list = list(active_tracks.items())
        
        for i, (tid, state) in enumerate(track_list):
            if state.latest_bbox is None:
                continue
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(state.latest_bbox, det.bbox)
        
        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < 0.3:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = track_list[i][0]
            det = detections[j]
            matched_track_ids.add(tid)
            matched_det_ids.add(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
            
            state = self.tracks[tid]
            if state.miss_count:
                state.bottom_centers_world.clear()
                state.timestamps.clear()
            bc = det.bottom_center
            state.bboxes.append(det.bbox)
            state.bottom_centers_image.append(bc)
            state.timestamps.append(timestamp_sec)
            state.hit_count += 1
            state.miss_count = 0
            state.is_active = True
            
            if calibration is not None and calibration.is_valid():
                wx, wy = calibration.transform(bc[0], bc[1])
                state.bottom_centers_world.append((wx, wy))
        
        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_det_ids:
                continue
            new_id = max(self.tracks.keys(), default=0) + 1
            state = TrackState(
                track_id=new_id,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            bc = det.bottom_center
            state.bboxes.append(det.bbox)
            state.bottom_centers_image.append(bc)
            state.timestamps.append(timestamp_sec)
            if calibration is not None and calibration.is_valid():
                wx, wy = calibration.transform(bc[0], bc[1])
                state.bottom_centers_world.append((wx, wy))
            state.hit_count = 1
            self.tracks[new_id] = state
        
        # Mark unmatched existing tracks
        for tid, state in self.tracks.items():
            if tid not in matched_track_ids and state.is_active:
                state.miss_count += 1
                if state.miss_count > self.max_miss_frames:
                    state.is_active = False
        
        return {tid: s for tid, s in self.tracks.items() if s.is_active}

    @staticmethod
    def _iou(bbox_a: Tuple, bbox_b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def get_all_tracks(self) -> Dict[int, TrackState]:
        return dict(self.tracks)

    def get_active_tracks(self) -> Dict[int, TrackState]:
        return {tid: s for tid, s in self.tracks.items() if s.is_active}

    def reset(self):
        self.tracks.clear()
        self._frame_count = 0
        self._init_bytetrack()
