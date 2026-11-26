"""Persist independently versioned TCVV runs in the unified SQLite database."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from .context import Candidate, TemporalContext, availability
from .verification import VERSION, Policy, Verifier


SCHEMA = """
CREATE TABLE IF NOT EXISTS tcvv_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tcvv_runs_session ON tcvv_runs(session_id);
CREATE TABLE IF NOT EXISTS tcvv_results (
    run_id TEXT NOT NULL REFERENCES tcvv_runs(run_id),
    candidate_id TEXT NOT NULL REFERENCES candidates(event_id),
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, candidate_id)
);
"""


def read_runs(database, session_id):
    if not database.is_file():
        return []
    with closing(sqlite3.connect(database)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'tcvv_runs'").fetchone():
            return []
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT payload FROM tcvv_runs WHERE session_id = ? ORDER BY rowid DESC", (session_id,),
        )]


def read_results(database, session_id, run_id):
    with closing(sqlite3.connect(database)) as connection:
        return [json.loads(row[0]) for row in connection.execute("""
            SELECT result.payload FROM tcvv_runs AS run
            JOIN tcvv_results AS result USING (run_id)
            WHERE run.session_id = ? AND run.run_id = ? ORDER BY result.rowid
        """, (session_id, run_id))]


def run_verification(database, session_id, *, run_id=None, progress=lambda *_: None):
    """Load the persisted System A population and commit results atomically."""
    started = perf_counter()
    run_id = run_id or uuid4().hex
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        existing = connection.execute("SELECT session_id, payload FROM tcvv_runs WHERE run_id = ?", (run_id,)).fetchone()
        if existing:
            if existing[0] != session_id:
                raise ValueError("Verification run ID belongs to another session.")
            return json.loads(existing[1])
        stored = connection.execute("SELECT payload FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if stored is None:
            raise ValueError("The completed unified session is unavailable.")
        session = json.loads(stored[0])
        records = [json.loads(row[0]) for row in connection.execute(
            "SELECT payload FROM candidates WHERE session_id = ? ORDER BY rowid", (session_id,),
        )]
        if message := availability(session, records):
            raise ValueError(message)
        progress(0, "Loading stored track and scene observations")
        candidates = [Candidate.from_record(row, session) for row in records]
        context = TemporalContext.load(session, candidates)
        policy = Policy()
        verifier = Verifier(session, context, policy)
        results = []
        for count, candidate in enumerate(candidates, 1):
            result = verifier.evaluate(candidate)
            results.append(dict(result, candidate_id=candidate.event_id, track_id=candidate.track_id,
                                violation_type=candidate.kind, event_timestamp=candidate.timestamp,
                                event_frame=candidate.frame, original_decision=candidate.original))
            if count % 25 == 0 or count == len(candidates):
                progress(.2 + .7 * count / len(candidates), f"Evaluated {count:,} of {len(candidates):,} candidates")
        run = dict(run_id=run_id, session_id=session_id, video_id=session["video_id"], version=VERSION,
                   policy=asdict(policy), source_configuration=session["configuration"],
                   speed_calibration=session.get("calibration"),
                   verified_at=datetime.now(timezone.utc).isoformat(), candidate_count=len(candidates),
                   elapsed_seconds=perf_counter() - started)
        inserted = connection.execute("INSERT INTO tcvv_runs VALUES (?, ?, ?) ON CONFLICT(run_id) DO NOTHING",
                                      (run_id, session_id, json.dumps(run, allow_nan=False))).rowcount
        if inserted:
            connection.executemany("INSERT INTO tcvv_results VALUES (?, ?, ?)", [
                (run_id, row["candidate_id"], json.dumps(row, allow_nan=False)) for row in results
            ])
            run["elapsed_seconds"] = perf_counter() - started
            connection.execute("UPDATE tcvv_runs SET payload = ? WHERE run_id = ?",
                               (json.dumps(run, allow_nan=False), run_id))
        else:
            owner, payload = connection.execute(
                "SELECT session_id, payload FROM tcvv_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
            if owner != session_id:
                raise ValueError("Verification run ID belongs to another session.")
            run = json.loads(payload)
    progress(1, "TCVV verification complete")
    return run
