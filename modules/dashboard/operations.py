"""Operational dashboards backed by completed unified sessions."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.unified.dashboard import LABELS
from modules.unified.pipeline import DATABASE
from modules.unified.storage import read_records

from .components import empty_state, kpis, navigate, page_header


def session_date(run):
    path = Path(run.get("annotated_video", ""))
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y · %H:%M")
    return "Unavailable"


def session_table(sessions):
    return [
        {"Session": run["session_id"][:8], "Video": run["video_name"],
         "Output saved": session_date(run), "Frames": run["frames"],
         "Vehicle detections": run["detections"], "Tracks": run["unique_tracks"],
         "Confirmed": sum(run.get("confirmed_counts", {}).values())}
        for run in sessions
    ]


def model_table(root, settings):
    anpr = settings.get("anpr", {})
    paths = {
        "YOLOv11 vehicle weights": root / "models" / settings["detection"].get("model", "yolo11n.pt"),
        "Plate detector weights": root / anpr.get("detector_weights", "models/anpr/yolo_best.pt"),
        "EasyOCR weights": root / anpr.get("ocr_model_dir", "models/anpr") / "english_g2.pth",
    }
    return [{"Resource": model, "Availability": "File present" if path.is_file() else "File missing"}
            for model, path in paths.items()]


def render_operations(root, current, settings):
    page_header("Operations overview", "Review completed monitoring sessions and recent confirmed violations.", "OVERVIEW")

    database = root / "outputs" / "unified" / DATABASE
    sessions = read_records(database, "sessions")
    events = read_records(database)
    st.button("Open Unified Monitoring", on_click=navigate, args=("unified",), type="primary")
    if current:
        st.markdown("#### Current browser session")
        st.caption(current["video_name"])
        kpis([("", label, value, "blue") for label, value in (
            ("Frames processed", current["frames"]), ("Vehicle tracks", current["unique_tracks"]),
            ("Confirmed violations", sum(current["confirmed_counts"].values())))])
    st.markdown("#### Unified history")
    st.caption("Detections are counted per frame. Track counts are summed across independent sessions.")
    kpis([("", label, value, "blue") for label, value in (
        ("Total sessions", len(sessions)), ("Vehicle detections", sum(run["detections"] for run in sessions)),
        ("Session track total", sum(run["unique_tracks"] for run in sessions)),
        ("Confirmed violations", len(events)),
    )])
    by_rule = Counter(row["violation_type"] for row in events)
    if sessions:
        kpis([("", label, by_rule[kind], "red") for kind, label in LABELS.items()])
    if sessions:
        st.caption(f"Latest completed session · {sessions[0]['video_name']} · {session_date(sessions[0])}")
        st.markdown("#### Recent sessions")
        st.dataframe(session_table(sessions[:8]), hide_index=True, width="stretch")
    else:
        empty_state("No completed sessions", "Process a video in Unified Monitoring to populate the overview.")
    if events:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Latest confirmed events")
            st.dataframe([
                {"Rule": LABELS.get(event["violation_type"], event["violation_type"]),
                 "Session": event["session_id"][:8], "Track": event["track_id"],
                 "Video time": f"{event['timestamp']:.2f} s", "Registration": event.get("number_plate") or "—"}
                for event in events[:8]
            ], hide_index=True, width="stretch")
        with right:
            st.markdown("#### Confirmed by violation type")
            counts = pd.Series([LABELS.get(row["violation_type"], row["violation_type"]) for row in events]).value_counts()
            st.bar_chart(counts, horizontal=True)
    with st.expander("Local model files"):
        st.dataframe(model_table(root, settings), hide_index=True, width="stretch")
    st.button("Open Violation Records", on_click=navigate, args=("reports",))


def plot(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111d30", plot_bgcolor="#111d30",
                      margin=dict(l=24, r=24, t=52, b=25), height=340)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_analytics(root):
    page_header("Traffic analytics", "Compare event outcomes and processing performance across unified sessions.", "ANALYTICS")
    database = root / "outputs" / "unified" / DATABASE
    sessions = read_records(database, "sessions")
    if not sessions:
        empty_state("No analytics available", "Complete a unified monitoring session to generate analytics.")
        return
    choice = st.selectbox("Scope", [None, *[run["session_id"] for run in sessions]],
                          format_func=lambda value: "All unified sessions" if value is None else
                          next(f"{run['video_name']} · {value[:8]}" for run in sessions if run["session_id"] == value))
    chosen = [run for run in sessions if choice is None or run["session_id"] == choice]
    events = read_records(database, "events", choice)
    candidates = read_records(database, "candidates", choice)
    event_counts = {"Confirmed": len(events), "Uncertain": sum(
        "uncertain" in str(row.get("verification_status", "")).casefold() for row in candidates),
        "Candidate records": len(candidates)}
    kpis([("", name, count, "blue") for name, count in event_counts.items()])
    st.caption("Candidate records include confirmed and uncertain outcomes. Track totals are session-scoped.")
    if events or event_counts["Uncertain"]:
        outcomes = {"Confirmed": len(events), "Uncertain": event_counts["Uncertain"]}
        plot(px.bar(x=list(outcomes), y=list(outcomes.values()), title="Confirmed and uncertain outcomes",
                    labels={"x": "Outcome", "y": "Events"}))
    if events:
        counts = pd.Series([LABELS.get(row["violation_type"], row["violation_type"]) for row in events]).value_counts()
        plot(px.bar(x=counts.values, y=counts.index, orientation="h", title="Confirmed violations by type",
                    labels={"x": "Confirmed events", "y": "Violation type"}))
        speeds = [row["speed"] for row in events if row.get("speed") is not None]
        if speeds:
            plot(px.histogram(x=speeds, title="Vehicle speeds at confirmed events",
                              labels={"x": "Recorded speed (km/h)"}))
    else:
        st.info("This scope has no confirmed events.")
    frame = pd.DataFrame(session_table(chosen))
    st.dataframe(frame, hide_index=True, width="stretch")
    rates = [{"Session": run["session_id"][:8], "Video": run["video_name"], "FPS": run["processing_fps"]}
             for run in chosen if run.get("processing_fps") is not None]
    if rates:
        plot(px.bar(pd.DataFrame(rates), x="Session", y="FPS", hover_data=["Video"], title="Processing throughput by session",
                    labels={"FPS": "Processing FPS"}))


def render_settings(root, settings):
    page_header("System & Settings", "Inspect model files, processing defaults, saved scenes, and storage.", "SYSTEM")
    st.markdown("#### Model availability")
    st.caption("File checks reflect local prerequisites. Model loading is verified during processing.")
    st.dataframe(model_table(root, settings), hide_index=True, width="stretch")
    st.caption("ByteTrack initializes with the vehicle processor for each run.")
    st.markdown("#### Processing defaults")
    defaults = [
        {"Setting": "Vehicle detector confidence", "Value": str(settings["detection"]["confidence_threshold"])},
        {"Setting": "Default speed limit", "Value": f"{settings['speed_estimation']['default_speed_limit_kmh']} km/h"},
        {"Setting": "ANPR enabled", "Value": "Yes" if settings.get("anpr", {}).get("enabled") else "No"},
    ]
    st.dataframe(defaults, hide_index=True, width="stretch")
    st.markdown("#### Saved scene profiles")
    presets = []
    for kind, label in (("rlvd", "Red light"), ("lcvd", "Lane change")):
        for file in sorted((root / "configs" / kind).glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8-sig"))
                presets.append({"Name": data.get("name", file.stem), "Rule": label,
                                "Profile": file.name, "Video hash": str(data.get("video_id", ""))[:12]})
            except (OSError, ValueError, AttributeError):
                continue
    if presets:
        st.dataframe(presets, hide_index=True, width="stretch")
    else:
        st.caption("No saved geometry profiles are available.")
    with st.expander("Detection workflows"):
        st.markdown(
            "- **Overspeed** uses a matching camera profile or configured speed assumptions.\n"
            "- **Red light** combines signal history with directed stop-line crossings.\n"
            "- **Lane change** checks transitions across permitted and restricted boundaries."
        )
    st.markdown("#### Output storage")
    database = root / "outputs" / "unified" / DATABASE
    st.code(str(database), language=None)
    st.caption("Unified sessions store structured records in SQLite and keep videos, CSVs, and evidence on disk.")
    st.button("Review saved records", on_click=navigate, args=("reports",))
