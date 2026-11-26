"""Central storage and reusable report exports for unified sessions."""

import csv
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def save_session(database, session, candidates, events):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY REFERENCES candidates,
                session_id TEXT NOT NULL REFERENCES sessions, payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS candidates_by_session ON candidates(session_id);
            CREATE INDEX IF NOT EXISTS events_by_session ON events(session_id);
        """)
        connection.execute("INSERT INTO sessions VALUES (?, ?)",
                           (session["session_id"], json.dumps(session, allow_nan=False)))
        for table, records in (("candidates", candidates), ("events", events)):
            connection.executemany(
                f"INSERT INTO {table} VALUES (?, ?, ?)",
                [(row["event_id"], row["session_id"], json.dumps(row, allow_nan=False)) for row in records],
            )


def read_records(database, table="events", session_id=None):
    if table not in ("sessions", "candidates", "events"):
        raise ValueError("Unknown report type")
    if not Path(database).is_file():
        return []
    query = f"SELECT payload FROM {table}"
    args = ()
    if session_id is not None:
        query += " WHERE session_id = ?"
        args = (session_id,)
    with closing(sqlite3.connect(database)) as connection:
        return [json.loads(row[0]) for row in connection.execute(query + " ORDER BY rowid DESC", args)]


CANDIDATE_FIELDS = (
    "event_id", "session_id", "video_id", "track_id", "violation_type", "timestamp",
    "frame_number", "vehicle_class", "confidence", "bbox", "trajectory", "speed",
    "speed_limit", "lane_information", "signal_information", "evidence_frames",
    "evidence_paths", "metadata", "rule_status", "verification_status",
)
EVENT_FIELDS = CANDIDATE_FIELDS + (
    "number_plate", "ocr_confidence", "plate_status", "plate_frame_number",
    "plate_bbox", "plate_detection_confidence", "plate_candidate", "anpr_error",
)


def csv_export(records, table="events"):
    buffer = io.StringIO()
    standard = EVENT_FIELDS if table == "events" else CANDIDATE_FIELDS
    fields = list(dict.fromkeys((*standard, *(key for row in records for key in row))))
    writer = csv.DictWriter(buffer, fields)
    writer.writeheader()
    for row in records:
        writer.writerow({
            field: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
            for field, value in row.items()
        })
    return buffer.getvalue()
