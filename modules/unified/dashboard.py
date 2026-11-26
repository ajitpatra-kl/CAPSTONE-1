"""Streamlit input, geometry and reports for unified sessions."""

import hashlib
import json
import io
import re
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import pandas as pd
import streamlit as st

from modules.dashboard.components import empty_state, kpis, page_header, status
from modules.lcvd.configuration import Configuration as LaneConfig, Settings
from modules.lcvd.dashboard import EDITOR as LANE_EDITOR
from modules.rlvd.configuration import Configuration as RedConfig, DIRECTIONS, LAMP_LAYOUTS, video_metadata
from modules.rlvd.dashboard import EDITOR as RED_EDITOR, preview_image, signal_check
from modules.rlvd.violation_engine import Thresholds
from modules.speed_detection.camera_calibration import CalibrationManager
from modules.speed_detection.speed_session import SpeedSession
from .pipeline import DATABASE, process_video
from .storage import csv_export, read_records


LABELS = {"overspeed": "Overspeed", "red_light": "Red-light violation", "lane_change": "Lane-change violation"}


@st.cache_data(max_entries=8, show_spinner=False)
def scene_metadata(path):
    return video_metadata(path)


def draft_value(key, default):
    return st.session_state.get("unified_setup_draft", {}).get(key, default)


def draft_index(key, options, default):
    value = draft_value(key, default)
    return options.index(value) if value in options else 0


def matching_config(root, kind, video_id):
    matches = {}
    for path in sorted((root / "configs" / kind).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("video_id") == video_id and data.get("name"):
            matches[path.stem] = data
    choices = [None, *matches]
    key = f"unified_{kind}_saved_{video_id}"
    chosen = st.selectbox("Saved geometry", choices, key=key, index=draft_index(key, choices, None),
                          format_func=lambda key: matches[key]["name"] if key else "Draw new geometry")
    return chosen, matches.get(chosen, {})


def configure_red(root, video_id, image, width, height, fps):
    st.subheader("Signal and stop line")
    name, saved = matching_config(root, "rlvd", video_id)
    key = f"unified_red_{video_id}_{name}"
    approach = st.selectbox("Side vehicles approach from", list(DIRECTIONS),
                            index=draft_index(key + "_approach", list(DIRECTIONS), saved.get("approach", "Bottom")), key=key + "_approach",
                            help="A crossing is measured from this side of the line to the opposite side.")
    layout = st.selectbox("Lamp arrangement", LAMP_LAYOUTS,
                          index=draft_index(key + "_lamps", LAMP_LAYOUTS, saved.get("lamp_layout", "Automatic")), key=key + "_lamps")
    initial = draft_value(key, {field: saved.get(field, []) for field in ("roi", "line", "lane")})
    geometry = RED_EDITOR(image=image, width=width, height=height, geometry=initial,
                          revision=key, key=key, default=initial)
    with st.expander("Red-light confirmation settings"):
        thresholds = Thresholds(
            signal_frames=st.number_input("Consecutive lamp frames", 2, 30, draft_value(key + "_signal", 3), key=key + "_signal"),
            approach_frames=st.number_input("Stable approach frames", 2, 30, draft_value(key + "_stable", 3), key=key + "_stable"),
            crossing_frames=st.number_input("Frames beyond the stop line", 2, 30, draft_value(key + "_cross", 2), key=key + "_cross"),
            deadband=st.number_input("Stop-line tolerance (pixels)", .5, 30., draft_value(key + "_band", 3.), key=key + "_band"),
            confidence=st.slider("Minimum track confidence", .15, .95, draft_value(key + "_confidence", .35), key=key + "_confidence"),
            anchor_fraction=st.slider("Anchor height within box", .5, 1., draft_value(key + "_anchor", 1.), key=key + "_anchor"),
        )
    return RedConfig(saved.get("name", "Unified signal geometry"), video_id, width, height, fps,
                     approach=approach, lamp_layout=layout, **geometry), thresholds


def configure_lane(root, video_id, image, width, height, fps):
    st.subheader("Lane boundaries")
    name, saved = matching_config(root, "lcvd", video_id)
    key = f"unified_lane_{video_id}_{name}"
    angles = ["Vertical", "Horizontal"]
    axis = st.selectbox("Road orientation", angles, index=draft_index(key + "_axis", angles, saved.get("axis", "Vertical")),
                        key=key + "_axis")
    directions = ("Left to right", "Right to left") if axis == "Vertical" else ("Top to bottom", "Bottom to top")
    direction = st.selectbox("Lane numbering direction", directions, index=draft_index(key + "_" + axis, directions, directions[int(saved.get("reverse", False))]),
                              key=key + "_" + axis)
    travel = ("Toward the bottom", "Toward the top") if axis == "Vertical" else ("Toward the right", "Toward the left")
    monitoring = st.selectbox("Monitoring direction", ("Either direction", *travel), key=key + "_travel_" + axis,
                              index=draft_index(key + "_travel_" + axis, ("Either direction", *travel), "Either direction"),
                              help="Uses movement along the road at the observed boundary crossing.")
    monitoring_direction = {travel[0]: "increasing", travel[1]: "decreasing"}.get(monitoring)
    initial = draft_value(key, saved.get("lines", []))
    lines = LANE_EDITOR(image=image, width=width, height=height, axis=axis, reverse=direction == directions[1],
                        revision=key, lines=initial, default=initial, key=key)
    settings = Settings(**saved.get("settings", {}))
    with st.expander("Lane-change confirmation settings"):
        settings = Settings(
            stable_frames=st.number_input("Stable observations per lane", 2, 30, draft_value(key + "_stable", settings.stable_frames),
                                          key=key + "_stable"),
            margin=st.number_input("Boundary tolerance (pixels)", 0., 40., draft_value(key + "_margin", float(settings.margin)),
                                   key=key + "_margin"),
            confidence=st.slider("Minimum track confidence", .1, .99, draft_value(key + "_confidence", float(settings.confidence)),
                                 key=key + "_confidence"),
            max_gap=st.number_input("Maximum gap before re-seeding", 1, 60, settings.max_gap, key=key + "_gap"),
            max_jump=st.slider("Maximum one-frame shift (fraction of lane width)", .15, 1., draft_value(key + "_jump", float(settings.max_jump)),
                               key=key + "_jump"),
        )
    return LaneConfig(saved.get("name", "Unified lane geometry"), video_id, width, height, fps,
                      lines, axis, direction == directions[1], settings), monitoring_direction


def configure_speed(settings, mixed, video_id):
    st.subheader("Speed calibration")
    speed = settings["speed_estimation"]
    modes = ("Matching video profile", "Custom road dimensions", "Configured assumptions")
    key = f"unified_calibration_{video_id}"
    mode = st.selectbox("Road calibration", modes, key=key, index=draft_index(key, modes, modes[0]))
    speed_settings = dict(speed)
    if mode != "Configured assumptions":
        speed_settings.update(assumed_road=None, pixel_scale_mpp=None)
    raw_points = None
    if mode == "Custom road dimensions":
        st.caption("Speeds from user-entered road dimensions are shown as estimates.")
        raw_points = st.text_input(
            "Four image points in pixels as JSON",
            value=draft_value(f"unified_speed_points_{video_id}", ""),
            placeholder="[[100,100], [200,100], [300,300], [50,300]]",
            help="Start at the far left road corner, then enter far right, near right, and near left.",
            key=f"unified_speed_points_{video_id}",
        )
        speed_settings["custom_road"] = {
            "width_m": st.number_input("Road width (m)", min_value=.01, value=draft_value(f"unified_speed_width_{video_id}", 11.), key=f"unified_speed_width_{video_id}"),
            "length_m": st.number_input("Road length (m)", min_value=.01, value=draft_value(f"unified_speed_length_{video_id}", 50.), key=f"unified_speed_length_{video_id}"),
        }
    elif mode == "Configured assumptions":
        st.caption("A matching profile takes priority. Speeds based on configured assumptions are estimates.")
    else:
        st.caption("Speed evaluation requires a matching camera profile in this mode.")
    options = {
        "speed_limit": st.number_input("Speed limit (km/h)", min_value=1., value=draft_value(f"unified_speed_limit_{video_id}", float(speed["default_speed_limit_kmh"])), key=f"unified_speed_limit_{video_id}"),
        "speed_estimation": speed_settings,
    }
    with st.expander("Speed confirmation settings"):
        options["violation_duration"] = st.number_input(
            "Minimum time above the limit (seconds)", min_value=.1,
            value=draft_value(f"unified_speed_duration_{video_id}", float(settings["violation"]["min_violation_duration_seconds"])), key=f"unified_speed_duration_{video_id}",
        )
        options["smoothing_window"] = st.number_input("Smoothing window (frames)", 5, 21, draft_value(f"unified_speed_smoothing_{video_id}", speed["smoothing_window"]), key=f"unified_speed_smoothing_{video_id}")
        if mixed:
            st.caption("This session uses the shared signal and lane tracking profile.")
        else:
            options["confidence"] = st.slider("Vehicle detector confidence", .1, .9,
                                              draft_value(f"unified_speed_confidence_{video_id}", float(settings["detection"]["confidence_threshold"])), step=.05, key=f"unified_speed_confidence_{video_id}")
    return options, raw_points


def render_unified(root, settings, dataset_config):
    page_header("AI Traffic Monitoring",
                "Upload a traffic surveillance video, configure the scene, and analyze vehicle violations using AI-powered detection and tracking.",
                "UNIFIED MONITORING")
    output = root / "outputs" / "unified"
    source, rules = st.container(key="monitor_setup").columns([3, 2], border=True)
    with source:
        st.subheader("1. Video input")
        uploaded = st.file_uploader("Traffic video", type=["mp4", "avi", "mov", "mkv"], key="unified_upload",
                                    help="MP4, AVI, MOV or MKV. Maximum upload size: 500 MB.")
    with rules:
        st.subheader("2. Violation types")
        with st.expander("Select multiple rules"):
            defaults = draft_value("unified_selected", ["overspeed"]) if "unified_selected" not in st.session_state else None
            selected = st.multiselect("Violation types", list(LABELS), default=defaults,
                                      format_func=LABELS.get, key="unified_selected", label_visibility="collapsed")
        descriptions = {"overspeed": ("Speed above the configured limit", ":material/speed:"),
                        "red_light": ("Crossing the stop line on red", ":material/traffic:"),
                        "lane_change": ("Crossing a restricted lane boundary", ":material/alt_route:")}

        def toggle_rule(kind):
            current = st.session_state.unified_selected
            st.session_state.unified_selected = [item for item in current if item != kind] if kind in current else [*current, kind]

        for kind, (description, icon) in descriptions.items():
            active = kind in selected
            st.button(f"**{LABELS[kind]}** · {description}", key=f"monitor_rule_{kind}",
                      icon=icon, width="stretch", type="primary" if active else "secondary",
                      help="Click to remove this rule" if active else "Click to add this rule",
                      on_click=toggle_rule, args=(kind,))
    staged = st.session_state.get("unified_staged")
    if uploaded is not None:
        name = uploaded.name
        upload_id = getattr(uploaded, "file_id", None)
        if staged is None or upload_id is None or st.session_state.get("unified_upload_id") != (upload_id, name):
            video_id = hashlib.sha256(uploaded.getbuffer()).hexdigest()
            if staged is None or staged[0] != (video_id, name):
                temporary = TemporaryDirectory(prefix="traffic-unified-")
                path = Path(temporary.name) / ("input" + Path(name).suffix.lower())
                path.write_bytes(uploaded.getbuffer())
                staged = ((video_id, name), temporary, path)
                st.session_state.unified_staged = staged
            st.session_state.unified_upload_id = (upload_id, name)
    elif staged and source.button("Remove staged video"):
        st.session_state.pop("unified_staged", None)
        st.session_state.pop("unified_upload_id", None)
        staged = None

    if staged is None:
        if summary := st.session_state.get("unified_summary"):
            render_summary(output, summary, "unified_last")
        return

    (video_id, name), _, path = staged
    try:
        width, height, fps, count = scene_metadata(str(path))
        index = st.number_input("Scene preview frame", min_value=0, max_value=count - 1,
                                value=draft_value(f"unified_preview_{video_id}", 0), key=f"unified_preview_{video_id}")
        image = preview_image(str(path), index)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    with st.container(key="monitor_source", border=True):
        st.markdown(f"**{name}**")
        st.caption(f"{width} × {height} | {fps:.2f} nominal FPS | {count:,} frames | Approx. {count/fps:.1f} s")
        with st.expander("Source video and scene preview", expanded=True):
            video, still = st.columns(2)
            video.video(str(path))
            still.image(image, caption=f"Scene frame {index}", width="stretch")

    st.subheader("3. Scene configuration")
    options = {}
    checks = [("Video uploaded and metadata validated", "green"),
              ("Violation rules selected" if selected else "Select a violation rule", "green" if selected else "amber")]
    issues = []
    if "overspeed" in selected:
        with st.container(border=True):
            speed, points = configure_speed(settings, len(selected) > 1, video_id)
        options["speed_options"] = speed
        speed_settings = speed["speed_estimation"]
        try:
            road = speed_settings.get("assumed_road")
            if points is not None:
                road = speed_settings["custom_road"]
                road["image_points"] = json.loads(points)
                SpeedSession._assumed_homography(road, (width, height))
                checks.append(("Custom road geometry ready · speeds are estimates", "amber"))
            else:
                manager = CalibrationManager(dataset_config)
                profile = next((key for key, entry in manager.profiles.items()
                                if entry.get("sha256") == video_id and tuple(entry.get("resolution", ())) == (width, height)), None)
                if profile and manager.get_calibration(profile) is not None:
                    checks.append((f"Matching calibration profile · {manager.profiles[profile]['calibration_status']}", "green"))
                elif (road and road.get("image_points") is not None) or speed_settings.get("pixel_scale_mpp") is not None:
                    if road and road.get("image_points") is not None:
                        SpeedSession._assumed_homography(road, (width, height))
                    checks.append(("Configured assumptions · speeds are estimates", "amber"))
                else:
                    checks.append(("Speed evaluation unavailable without calibration; tracking can still run", "amber"))
        except (ValueError, TypeError, RuntimeError, cv2.error) as error:
            issues.append(f"Speed calibration: {error}")
    if "red_light" in selected:
        with st.container(border=True):
            scene, thresholds = configure_red(root, video_id, image, width, height, fps)
            options.update(red_config=scene, red_thresholds=thresholds)
            try:
                scene.validate()
                checks.append(("Signal ROI and stop line ready", "green"))
                signature = (video_id, scene.roi, scene.lamp_layout, thresholds.signal_frames, index)
                if st.button("Verify signal near preview frame", key="unified_red_verify"):
                    st.session_state.unified_signal_result = (signature, signal_check(
                        path, scene, thresholds.signal_frames, count, index)[0])
                previous = st.session_state.get("unified_signal_result")
                if previous and previous[0] == signature:
                    with st.expander("Signal verification history", expanded=True):
                        st.dataframe(previous[1], hide_index=True, width="stretch")
            except ValueError as error:
                issues.append(f"Red light: {error}")
    if "lane_change" in selected:
        with st.container(border=True):
            scene, direction = configure_lane(root, video_id, image, width, height, fps)
            options.update(lane_config=scene, lane_direction=direction)
            try:
                scene.geometry()
                checks.append(("Lane boundaries ready", "green"))
            except ValueError as error:
                issues.append(f"Lane change: {error}")

    with st.container(border=True):
        st.subheader("4. Processing readiness")
        for label, tone in checks:
            status(label, tone)
        for issue in issues:
            status(f"Required · {issue}", "amber")
        if st.button("Start processing", type="primary", disabled=not selected or bool(issues)):
            try:
                started = perf_counter()
                progress = st.progress(0, text="Preparing video")
                timing = st.empty()

                def update_progress(fraction, message):
                    progress.progress(fraction, text=message)
                    elapsed = perf_counter() - started
                    match = re.fullmatch(r"Processed ([\d,]+) frames", message)
                    if match:
                        frames = int(match[1].replace(",", ""))
                        timing.caption(f"{frames:,} of {count:,} frames · {elapsed:.1f} s elapsed")
                    else:
                        timing.caption(f"{elapsed:.1f} s elapsed")

                st.session_state.unified_summary = process_video(
                    path, root, settings, dataset_config, video_name=name, progress=update_progress, **options,
                )
            except (ValueError, TypeError, RuntimeError, OSError, cv2.error) as error:
                st.error(str(error))
    summary = st.session_state.get("unified_summary")
    if summary and summary["video_id"] == video_id:
        render_summary(output, summary, "unified_live")


def event_status(row):
    return row.get("verification_status") or "Candidate"


def evidence_zip(record, output):
    root = output.resolve()
    data = io.BytesIO()
    with ZipFile(data, "w", ZIP_DEFLATED) as archive:
        for field, value in record["evidence_paths"].items():
            if field == "video" or not value:
                continue
            path = Path(value).resolve()
            if path.is_relative_to(root) and path.is_file():
                archive.write(path, arcname=path.name)
    return data.getvalue()


def review_events(records, output, key, confirmed, can_seek=False):
    if not records:
        empty_state("No records in this category", "Select another event category or processing session.")
        return
    with st.expander("Search and filters"):
        left, right = st.columns(2)
        rules = left.multiselect("Violation type", sorted({row["violation_type"] for row in records}),
                                 format_func=lambda value: LABELS.get(value, value), key=key + "_rule")
        statuses = right.multiselect("Event status", sorted({event_status(row) for row in records}),
                                     key=key + "_status")
        track = left.text_input("Track ID", key=key + "_track").strip().lstrip("#")
        plate = right.text_input("Registration contains", key=key + "_registration").strip().casefold()
        query = st.text_input("Search events", key=key + "_search").strip().casefold()
        low, high = min(row["timestamp"] for row in records), max(row["timestamp"] for row in records)
        if low < high:
            low, high = st.slider("Event time (seconds)", float(low), float(high), (float(low), float(high)),
                                  key=key + "_range")
    filtered = [row for row in records if
                (not rules or row["violation_type"] in rules) and (not statuses or event_status(row) in statuses)
                and (not track or str(row["track_id"]) == track)
                and plate in (row.get("number_plate") or "").casefold()
                and (not query or query in " ".join(str(value) for value in (
                    row["event_id"], row.get("vehicle_class"), LABELS.get(row["violation_type"]),
                    event_status(row), row.get("number_plate"), row["track_id"])).casefold())
                and low <= row["timestamp"] <= high]
    if not filtered:
        st.info("No events match these filters.")
        return
    table = [{
        "Event ID": row["event_id"][:8],
        "Violation type": LABELS.get(row["violation_type"], row["violation_type"]),
        "Status": event_status(row), "Track ID": row["track_id"],
        "Vehicle": row.get("vehicle_class"), "Registration": row.get("number_plate"),
        "Video time": row["timestamp"], "Speed": row.get("speed"), "Speed limit": row.get("speed_limit"),
        "Confidence": row.get("confidence"), "Plate status": row.get("plate_status"),
        "Evidence": "Available" if any(value and field != "video" and Path(value).is_file()
                                       for field, value in row.get("evidence_paths", {}).items()) else "Unavailable",
    } for row in filtered]
    table = pd.DataFrame(table).dropna(axis=1, how="all")
    view_id = hashlib.sha256("|".join(row["event_id"] for row in filtered).encode()).hexdigest()[:10]
    selected_table = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                                  selection_mode="single-row", key=key + "_rows_" + view_id,
                                  column_config={
                                      "Video time": st.column_config.NumberColumn(format="%.2f s"),
                                      "Speed": st.column_config.NumberColumn(format="%.1f km/h"),
                                      "Speed limit": st.column_config.NumberColumn(format="%.1f km/h"),
                                      "Confidence": st.column_config.NumberColumn(format="percent"),
                                  })
    st.caption(f"{len(filtered):,} of {len(records):,} events")
    choices = {row["event_id"]: row for row in filtered}
    active_row = selected_table.selection.rows
    if active_row and active_row[0] < len(filtered):
        picked = filtered[active_row[0]]["event_id"]
        if st.session_state.get(key + "_table_pick") != picked:
            st.session_state[key + "_evidence"] = picked
            st.session_state[key + "_table_pick"] = picked
    record_id = st.selectbox("Evidence record", list(choices), key=key + "_evidence",
                             format_func=lambda event: f"{LABELS.get(choices[event]['violation_type'])} | "
                             f"Track #{choices[event]['track_id']} | {choices[event]['timestamp']:.2f} s")
    record = choices[record_id]
    if can_seek:
        st.button("Cue video at event", key=key + "_cue",
                  on_click=st.session_state.update,
                  kwargs={key + "_play_from": max(0, int(record["timestamp"]))})
    with st.container(key="monitor_evidence_" + key):
        render_evidence(record, confirmed)
    if any(value and field != "video" and Path(value).is_file()
           for field, value in record["evidence_paths"].items()):
        st.download_button("Download event evidence ZIP", lambda: evidence_zip(record, output),
                           file_name=f"{record_id[:8]}-evidence.zip", mime="application/zip",
                           key=key + "_zip", on_click="ignore")


def run_metadata(summary):
    from modules.dashboard.operations import session_date

    config = summary.get("configuration", {})
    rules = [LABELS[kind] for kind, field in (("overspeed", "speed"), ("red_light", "red_light"),
                                             ("lane_change", "lane_change")) if config.get(field) is not None]
    st.caption(f"Session {summary['session_id'][:8]} | Output saved {session_date(summary)}")
    if rules:
        st.caption("Rules · " + ", ".join(rules))


def render_summary(output, summary, key):
    key = f"{key}_{summary['session_id']}"
    st.divider()
    st.subheader(summary["video_name"])
    run_metadata(summary)
    database = output / DATABASE
    events = read_records(database, "events", summary["session_id"])
    candidates = read_records(database, "candidates", summary["session_id"])
    video = Path(summary["annotated_video"])
    can_seek = video.is_file()
    with st.container(border=True):
        st.markdown("#### Annotated output video")
        if can_seek:
            start = st.session_state.get(key + "_confirmed_play_from", 0)
            st.video(str(video), start_time=start)
            if key + "_confirmed_play_from" in st.session_state:
                st.caption(f"Playback start set to {start} s. The evidence frame below marks the exact event.")
        else:
            st.caption("The stored video file is unavailable.")
    st.markdown("#### Confirmed violations")
    review_events(events, output, key + "_confirmed", confirmed=True, can_seek=can_seek)
    st.markdown("#### Tracking and processing")
    latency = summary.get("average_frame_latency_ms")
    rate = summary.get("processing_fps")
    kpis([("", label, value, "blue") for label, value in (
        ("Frames processed", summary.get("frames")), ("Vehicle detections", summary.get("detections")),
        ("Unique tracked vehicles", summary.get("unique_tracks")), ("Confirmed violations", len(events)),
        ("Processing FPS", f"{rate:.1f}" if rate is not None else None),
        ("Mean frame latency", f"{latency:.1f} ms" if latency is not None else None),
    ) if value is not None])
    if summary.get("calibration"):
        st.caption(summary["calibration"])
    track_path = Path(summary["tracks_csv"])
    if track_path.is_file() and st.toggle("Inspect vehicle tracks", key=key + "_show_tracks"):
        try:
            fields = ["track_id", "vehicle_class", "frame_number", "timestamp"]
            observations = pd.read_csv(track_path, usecols=fields)
            tracks = observations.groupby("track_id", sort=True).agg(
                Vehicle=("vehicle_class", "first"),
                Observations=("frame_number", "count"),
                First_seen=("timestamp", "min"), Last_seen=("timestamp", "max"),
            ).reset_index().rename(columns={"track_id": "Track ID", "First_seen": "First seen",
                                            "Last_seen": "Last seen"})
            st.dataframe(tracks, hide_index=True, width="stretch", column_config={
                field: st.column_config.NumberColumn(format="%.2f s") for field in ("First seen", "Last seen")
            })
        except (OSError, ValueError, pd.errors.ParserError):
            st.info("Track details are unavailable for this session. The original export is still available.")
    with st.expander(f"Candidate and uncertain records · {len(candidates)}"):
        st.caption("The candidate log includes confirmed, uncertain, permitted, and other rule outcomes.")
        review_events(candidates, output, key + "_candidates", confirmed=False)
    with st.expander("Export session files", expanded=True):
        left, right = st.columns(2)
        left.download_button("Confirmed events CSV", csv_export(events, "events"), "events.csv", "text/csv",
                             key=key + "_events", on_click="ignore")
        right.download_button("Candidate CSV", csv_export(candidates, "candidates"), "candidates.csv", "text/csv",
                              key=key + "_candidates_csv", on_click="ignore")
        if track_path.is_file():
            left.download_button("Track observations CSV", track_path.read_bytes, "tracks.csv", "text/csv",
                                 key=key + "_tracks", on_click="ignore")
        right.download_button("Experiment results JSON", json.dumps(summary, indent=2), "experiment.json",
                              "application/json", key=key + "_experiment", on_click="ignore")


    from modules.tcvv.dashboard import render_tcvv

    render_tcvv(database, summary, candidates, events, key, render_evidence)


def render_evidence(record, confirmed):
    paths = record["evidence_paths"]
    frame = record["frame_number"]
    st.caption(f"Track #{record['track_id']} | {LABELS.get(record['violation_type'], record['violation_type'])} | "
               f"Evidence at {record['timestamp']:.2f} s | Frame {frame}")
    reason = (record.get("metadata") or {}).get("reason") or (record.get("signal_information") or {}).get("reason")
    if reason:
        st.write(reason)
    st.subheader("Violation evidence" if confirmed else "Candidate evidence")
    def show_image(target, field, caption=None):
        path = paths.get(field)
        if path and Path(path).is_file():
            target.image(path, caption=caption, width="stretch")
        else:
            target.caption("Image unavailable.")

    original, annotated = st.columns(2, gap="medium")
    with original.container(border=True):
        st.markdown("**Original frame**")
        show_image(st, f"frame_{frame}", f"Frame {frame}")
    with annotated.container(border=True):
        st.markdown("**Annotated frame**")
        show_image(st, "annotated", f"Frame {frame}")

    vehicle, plate = st.columns(2, gap="medium")
    with vehicle.container(border=True):
        st.markdown("**Vehicle crop**")
        show_image(st, "vehicle")
    if confirmed:
        with plate.container(border=True):
            st.markdown("**Number plate**")
            raw, enhanced = st.tabs(["Original", "Enhanced"])
            show_image(raw, "plate")
            show_image(enhanced, "plate_enhanced")
            if record.get("number_plate"):
                st.markdown(f"**Registration:** {record['number_plate']}")
            if record.get("plate_bbox"):
                st.caption("Plate bounds on source frame · " + ", ".join(str(point) for point in record["plate_bbox"]))
            if record.get("plate_frame_number") is not None:
                st.caption(f"Source frame {record['plate_frame_number']}")
            if record.get("plate_status"):
                result = record["plate_status"]
                label = "Not Readable" if result in ("Text Unreadable", "Not Readable", "Low Confidence") else result
                st.caption(f"Plate recognition · {label}")
            if record.get("plate_candidate") and not record.get("number_plate"):
                st.caption(f"Unverified OCR candidate · {record['plate_candidate']}")

    extra = {field: path for field, path in paths.items() if path and Path(path).is_file() and (
        (field.startswith("frame_") and field != f"frame_{frame}") or field == "plate_source_vehicle"
    )}
    if extra:
        with st.expander("Additional evidence frames"):
            for field, path in extra.items():
                label = "Vehicle crop from plate source frame" if field == "plate_source_vehicle" else field.replace("_", " ").capitalize()
                st.image(path, caption=label)


def render_unified_reports(output):
    sessions = read_records(output / DATABASE, "sessions")
    if not sessions:
        st.info("Complete a unified detection run to generate a report.")
        return
    index = st.selectbox("Processing session", range(len(sessions)),
                         format_func=lambda i: f"{sessions[i]['video_name']} · {sessions[i]['session_id'][:8]}")
    render_summary(output, sessions[index], "unified_report")


def render_unified_anpr(output):
    database = output / DATABASE
    sessions = read_records(database, "sessions")
    if not sessions:
        return
    st.subheader("Unified ANPR records")
    selected = st.selectbox("Unified processing session", sessions,
                            format_func=lambda run: f"{run['video_name']} | {run['session_id'][:8]}",
                            key="unified_anpr_session")
    session_id = selected["session_id"]
    events = [row for row in read_records(database, "events", session_id) if row.get("plate_status")]
    review_events(events, output, f"unified_anpr_{session_id}", confirmed=True)
