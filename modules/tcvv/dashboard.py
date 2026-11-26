"""TCVV controls and review for the existing unified session dashboard."""

import sqlite3
from collections import Counter
from pathlib import Path
from uuid import uuid4

import streamlit as st

from .context import availability
from .evaluation import compare, parse_labels, score
from .storage import read_results, read_runs, run_verification


def render_tcvv(database, session, candidates, events, key, render_evidence):
    st.divider()
    st.subheader("System B — TCVV Verification")
    st.caption("System A — Existing Verification appears above. TCVV uses the stored candidates and observations from this session.")
    message = availability(session, candidates)
    pending = key + "_tcvv_pending"

    def queue():
        st.session_state.setdefault(pending, uuid4().hex)

    if message:
        st.info(message)
    else:
        st.button("Run TCVV Verification", key=key + "_tcvv_run", on_click=queue)
        if request := st.session_state.get(pending):
            progress = st.progress(0, text="Preparing TCVV")
            try:
                run = run_verification(database, session["session_id"], run_id=request,
                                       progress=lambda fraction, message: progress.progress(float(fraction), text=message))
                st.session_state[key + "_tcvv_selected"] = run["run_id"]
                st.success("TCVV verification complete.")
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                st.error(f"TCVV could not read or verify the stored session. {error}")
            finally:
                st.session_state.pop(pending, None)

    try:
        runs = {run["run_id"]: run for run in read_runs(database, session["session_id"])}
    except (sqlite3.Error, ValueError) as error:
        st.error(f"TCVV run history is unavailable. {error}")
        return
    if not runs:
        return
    run_id = st.selectbox("TCVV run", list(runs), key=key + "_tcvv_selected",
                          format_func=lambda identifier: (
                              f"{runs[identifier]['verified_at']} · v{runs[identifier]['version']} · {identifier[:8]}"
                          ))
    run = runs[run_id]
    try:
        results = read_results(database, session["session_id"], run_id)
    except (sqlite3.Error, ValueError) as error:
        st.error(f"The selected TCVV results are unavailable. {error}")
        return
    records = {row["event_id"]: row for row in candidates}
    confirmed = {row["event_id"]: row for row in events}
    details, comparison = st.tabs(["TCVV results", "System A vs System B"])
    with details:
        distribution = Counter(row["decision"] for row in results)
        columns = st.columns(5)
        for column, (label, value) in zip(columns, [
            ("Candidates evaluated", len(results)), ("Confirmed", distribution["CONFIRMED"]),
            ("Rejected", distribution["REJECTED"]), ("Uncertain", distribution["UNCERTAIN"]),
            ("Verification time", f"{run['elapsed_seconds']:.3f} s"),
        ]):
            column.metric(label, value)

        counts = Counter(row["violation_type"] for row in results)
        st.write(" · ".join(f"{kind.replace('_', ' ').title()} {count}" for kind, count in sorted(counts.items())))
        st.dataframe([{
            "Candidate ID": row["candidate_id"], "Violation type": row["violation_type"],
            "Vehicle track ID": row["track_id"], "Original System A decision": row["original_decision"],
            "TCVV decision": row["decision"], "Event timestamp (s)": row["event_timestamp"],
            "Verification explanation": row["explanation"],
            "Evidence availability": any(path and Path(path).is_file() for field, path in
                                         (confirmed.get(row["candidate_id"]) or records[row["candidate_id"]])["evidence_paths"].items()
                                         if field != "video"),
        } for row in results], hide_index=True)
        if results:
            chosen = st.selectbox("Inspect TCVV candidate", results, key=f"{key}_{run_id}_inspect",
                                  format_func=lambda row: f"Track {row['track_id']} · {row['violation_type']} · {row['event_timestamp']:.2f} s · {row['candidate_id'][:8]}")
            event_id = chosen["candidate_id"]
            st.write(f"System A · {chosen['original_decision']}")
            st.write(f"System B · {chosen['decision']} · {chosen['explanation']}")
            render_evidence(confirmed.get(event_id) or records[event_id], event_id in confirmed)
            with st.expander("Observation timeline and supporting context"):
                st.dataframe(chosen["observations"], hide_index=True)
                st.json({field: chosen[field] for field in ("reason_codes", "context", "limitations")})
    with comparison:
        st.caption("Both systems use this session’s candidate population. Comparison counts collapse repeated submissions of the same episode. Agreement measures confirmation decisions. Uncertain TCVV outcomes are excluded from agreement.")
        st.dataframe([{"Metric": label, "Count": value} for label, value in compare(results, records).items()],
                     hide_index=True)
        elapsed = run["elapsed_seconds"]
        rate = session.get("processing_fps")
        if rate:
            baseline = session["frames"] / rate
            st.write(f"System A processing {baseline:.3f} s · Additional TCVV latency {elapsed:.3f} s ({100 * elapsed / baseline:.2f}% of System A processing time)")
        st.caption("Upload labeled events for accuracy metrics. Use one row per physical event, with 1 for a violation and 0 for a verified non-violation. Required columns are session_id, video_id, track_id, violation_type, timestamp (seconds), and label. Include eligible missed violations to measure recall.")
        st.download_button("Ground truth CSV template", "session_id,video_id,track_id,violation_type,timestamp,label\n",
                           "tcvv-ground-truth.csv", "text/csv", key=key + "_tcvv_template")
        labels = st.file_uploader("Ground truth CSV", type="csv", key=key + "_tcvv_truth")
        if labels is not None:
            tolerance = st.number_input("Event matching tolerance (seconds)", min_value=0.0, value=0.5,
                                        step=0.1, key=key + "_tcvv_tolerance")
            try:
                truth = parse_labels(labels.getvalue().decode("utf-8-sig"), session["session_id"], session["video_id"])
                metrics, matched, unscored = score(results, records, truth, tolerance)
                st.caption(f"Matched {matched} of {len(truth)} annotations. {unscored} unmatched candidate episodes are unscored. Unmatched positive annotations count as misses. Uncertain outcomes count as non-confirmations.")
                st.dataframe([{"System": system, **values} for system, values in metrics.items()], hide_index=True)
            except ValueError as error:
                st.error(str(error))
