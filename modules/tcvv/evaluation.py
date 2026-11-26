"""Compare shared candidates and score explicitly annotated physical events."""

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite


TYPES = {"overspeed", "red_light", "lane_change"}


@dataclass(frozen=True)
class Label:
    track_id: int
    kind: str
    timestamp: float
    violation: bool


def parse_labels(content, session_id, video_id):
    reader = csv.DictReader(io.StringIO(content))
    required = {"session_id", "video_id", "track_id", "violation_type", "timestamp", "label"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Ground truth CSV needs session_id, video_id, track_id, violation_type, timestamp, and label.")
    labels = {}
    for row in reader:
        if row["session_id"] != session_id or row["video_id"] != video_id:
            raise ValueError("All ground truth rows must belong to the selected session and video.")
        try:
            track, timestamp = int(row["track_id"]), float(row["timestamp"])
        except (TypeError, ValueError) as error:
            raise ValueError("Ground truth track IDs and timestamps must be numeric.") from error
        if track < 0 or not isfinite(timestamp) or timestamp < 0 or row["violation_type"] not in TYPES or row["label"] not in ("0", "1"):
            raise ValueError("Use nonnegative timestamps, supported violation types, and labels 0 or 1.")
        key = track, row["violation_type"], timestamp
        label = Label(*key, row["label"] == "1")
        if key in labels and labels[key] != label:
            raise ValueError("Ground truth contains conflicting labels for an event.")
        labels[key] = label
    if not labels:
        raise ValueError("Ground truth has no labeled events.")
    return list(labels.values())


def episodes(results, records):
    """Collapse repeated submissions of the same episode before comparison."""
    groups = defaultdict(list)
    for result in results:
        record = records[result["candidate_id"]]
        kind = result["violation_type"]
        key = result["track_id"], kind
        frame = result["event_frame"]
        if kind == "overspeed":
            start = frame - max(1, record["metadata"].get("consecutive_frames", 1)) + 1
        else:
            start = frame
            key += (tuple(record["evidence_frames"]),)
        groups[key].append((start, frame, result))
    for rows in groups.values():
        current = []
        end = -1
        for start, frame, row in sorted(rows, key=lambda item: item[:2]):
            if current and start > end:
                yield current
                current = []
            current.append(row)
            end = max(end, frame)
        if current:
            yield current


def outcomes(episode):
    decisions = {row["decision"] for row in episode}
    a = any(row["original_decision"] == "Confirmed violation" for row in episode)
    b = next(decision for decision in ("CONFIRMED", "UNCERTAIN", "REJECTED") if decision in decisions)
    return a, b


def compare(results, records):
    counts = Counter()
    for event in episodes(results, records):
        a, b = outcomes(event)
        confirmed = b == "CONFIRMED"
        counts["Events compared"] += 1
        counts["Confirmed by System A"] += a
        counts["Confirmed by System B"] += confirmed
        counts["Rejected by System B"] += b == "REJECTED"
        counts["Uncertain in System B"] += b == "UNCERTAIN"
        counts["Agree on confirmation"] += (a and confirmed) or (not a and b == "REJECTED")
        counts["Confirmed only by System A"] += a and not confirmed
        counts["Confirmed only by System B"] += confirmed and not a
        counts["System A confirmations rejected by TCVV"] += a and b == "REJECTED"
        counts["System A confirmations downgraded to uncertain"] += a and b == "UNCERTAIN"
    return {"Total candidates": len(results), **counts}


def score(results, records, labels, tolerance):
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Matching tolerance must be nonnegative.")
    events = list(episodes(results, records))
    grouped = defaultdict(list)
    for index, event in enumerate(events):
        grouped[event[0]["track_id"], event[0]["violation_type"]].append(index)
    options = []
    for label in labels:
        distances = [(min(abs(row["event_timestamp"] - label.timestamp) for row in events[index]), index)
                     for index in grouped[label.track_id, label.kind]]
        options.append([index for distance, index in sorted(distances) if distance <= tolerance])

    # Augment one match at a time to maximize the number of one-to-one matches.
    owners = {}
    for index in sorted(range(len(labels)), key=lambda i: (len(options[i]), labels[i].timestamp, i)):
        queue = [index]
        links = {}
        visited = {index}
        for label_index in queue:
            free = None
            for event_index in options[label_index]:
                if event_index in links:
                    continue
                links[event_index] = label_index
                if event_index not in owners:
                    free = event_index
                    break
                if owners[event_index] not in visited:
                    visited.add(owners[event_index])
                    queue.append(owners[event_index])
            if free is not None:
                predecessor = {owners[event]: event for event in links if event in owners}
                while free is not None:
                    label_index = links[free]
                    owners[free] = label_index
                    free = predecessor.get(label_index)
                break

    matched = {label: event for event, label in owners.items()}
    metrics = {}
    for system in ("System A", "System B"):
        tp = fp = tn = fn = 0
        for index, label in enumerate(labels):
            predicted = False
            if index in matched:
                a, b = outcomes(events[matched[index]])
                predicted = a if system == "System A" else b == "CONFIRMED"
            tp += predicted and label.violation
            fp += predicted and not label.violation
            tn += not predicted and not label.violation
            fn += not predicted and label.violation

        def rate(numerator, denominator):
            return numerator / denominator if denominator else None

        uncertain = (sum(any(str(row["original_decision"]).lower().startswith("uncertain") for row in event)
                         and not outcomes(event)[0] for event in events)
                     if system == "System A" else sum(outcomes(event)[1] == "UNCERTAIN" for event in events))
        metrics[system] = {
            "Precision": rate(tp, tp + fp), "Recall": rate(tp, tp + fn),
            "F1-score": rate(2 * tp, 2 * tp + fp + fn), "False-positive rate": rate(fp, fp + tn),
            "False-negative rate": rate(fn, fn + tp), "Uncertain event rate": rate(uncertain, len(events)),
        }
    return metrics, len(matched), len(events) - len(matched)
