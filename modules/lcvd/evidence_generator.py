"""Write LCVD event records and crossing evidence."""

import csv
import json
from math import ceil, floor

import cv2


FIELDS = (
    "event_id", "violation_id", "track_id", "vehicle_class", "source_lane",
    "destination_lane", "boundary", "boundary_type", "timestamp_sec", "frame_number",
    "confirmation_frame", "confidence", "status", "reason", "original_frame_path",
    "evidence_image_path", "vehicle_crop_path",
)


def lane_information(geometry, event):
    adjacent = abs(event.source - event.target) == 1
    boundary = geometry.boundaries[max(event.source, event.target)] if adjacent else None
    return dict(
        source_lane=geometry.lane_number(event.source), destination_lane=geometry.lane_number(event.target),
        boundary=boundary.number if boundary else "", boundary_type=boundary.kind if boundary else "",
    )


def event_row(geometry, event, track_id, index, confirmation_frame):
    sweep = event.sweep
    observation = event.observation
    return dict(
        event_id=f"LC-{index:04}", violation_id=f"LCV-{index:04}" if event.status == "Confirmed violation" else "",
        track_id=track_id, vehicle_class=observation.vehicle,
        **lane_information(geometry, event),
        timestamp_sec=round(sweep.timestamp if sweep else observation.timestamp, 6),
        frame_number=observation.frame, confirmation_frame=confirmation_frame,
        confidence=round(observation.confidence, 4), status=event.status, reason=event.reason,
    )


def save_image(path, frame):
    ok, data = cv2.imencode(".png", frame)
    if not ok:
        raise OSError(f"Could not encode {path.name}.")
    path.write_bytes(data.tobytes())
    return str(path)


def save_evidence(directory, row, event, before, crossing, annotate):
    directory.mkdir(parents=True)
    annotated = crossing.copy()
    annotate(annotated)
    x1, y1, x2, y2 = event.observation.bbox
    crop = crossing[max(0, floor(y1)):ceil(y2), max(0, floor(x1)):ceil(x2)]
    row.update(
        original_frame_path=save_image(directory / "before.png", before),
        evidence_image_path=save_image(directory / "crossing.png", annotated),
        vehicle_crop_path=save_image(directory / "vehicle.png", crop),
    )
    (directory / "event.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
