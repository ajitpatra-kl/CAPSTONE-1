"""Independent Streamlit workflow for RLVD."""

import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Lock
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.dashboard.components import download_csv, empty_state, kpis, page_header, status
from modules.speed_detection.vehicle_detector import VehicleDetector

from .configuration import Configuration, DIRECTIONS, LAMP_LAYOUTS, read_frame, save_configuration, video_metadata
from .signal_detector import COLORS, SignalHistory, classify_signal
from .video_processor import process_video
from .violation_engine import Thresholds


EDITOR = components.declare_component("rlvd_geometry", path=str(Path(__file__).with_name("editor")))
EMPTY_GEOMETRY = {"roi": [], "line": [], "lane": []}


@st.cache_resource
def vehicle_detector():
    return VehicleDetector(confidence_threshold=.15), Lock()


@st.cache_data(max_entries=4, show_spinner=False)
def preview_image(path, index):
    frame = read_frame(path, index)
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Could not render the selected video frame.")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


def reset_editor(geometry=None):
    st.session_state.rlvd_revision = uuid4().hex
    st.session_state.rlvd_geometry = dict(geometry or EMPTY_GEOMETRY)


def signal_check(path, config, frames, frame_count, preview_index):
    capture = cv2.VideoCapture(str(path))
    # A bounded sequential window preserves temporal meaning and keeps the check responsive.
    begin = min(max(0, preview_index - 45), max(0, frame_count - 180))
    capture.set(cv2.CAP_PROP_POS_FRAMES, begin)
    history = SignalHistory(frames)
    rows, examples = [], {}
    try:
        for offset in range(min(180, frame_count - begin)):
            ok, frame = capture.read()
            if not ok:
                break
            reading = history.update(*classify_signal(frame, config.roi, config.lamp_layout))
            index = begin + offset
            rows.append({"Frame": index, "Seconds": round(index / config.fps, 3),
                         "Lamp candidate": reading.raw, "Confirmed": reading.state,
                         "Quality": round(reading.quality, 2)})
            x1, y1, x2, y2 = config.roi
            if reading.state not in examples or reading.quality > examples[reading.state][0]:
                cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), COLORS[reading.state], 2)
                examples[reading.state] = (reading.quality, index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return rows, examples


def render_results(result):
    st.divider()
    st.subheader("Detection results")
    for warning in result["warnings"]:
        st.warning(warning)
    summary, events, tracks, video, evidence, reports = st.tabs(
        ["Summary", "Violations & uncertain events", "Tracking", "Annotated video", "Evidence", "CSV reports"],
    )
    confirmed, uncertain = result["confirmed"], result["uncertain"]
    metrics = result["metrics"]
    with summary:
        kpis([(icon, label, metrics.get(label), tone) for icon, label, tone in (
            ("◉", "Confirmed violations", "red"), ("✓", "Legal crossings", "green"),
            ("?", "Uncertain events", "amber"), ("◎", "Unknown signal frames", "neutral"),
        )])
    with events:
        st.subheader("Confirmed violations")
        if confirmed:
            status("Confirmed crossings on red", "red")
            st.dataframe(pd.DataFrame(confirmed)[[
                "violation_id", "track_id", "vehicle_class", "crossing_timestamp",
                "signal_state", "vehicle_confidence", "plate_number",
            ]], hide_index=True, column_config={
                "violation_id": "Violation ID", "track_id": "Track ID", "vehicle_class": "Vehicle",
                "crossing_timestamp": st.column_config.NumberColumn("Crossing time", format="%.3f s"),
                "signal_state": "Signal", "vehicle_confidence": st.column_config.NumberColumn("Vehicle confidence", format="percent"),
                "plate_number": "Plate outcome",
            })
        else:
            empty_state("No confirmed violations", "This run did not confirm a crossing on red.")
        st.subheader("Uncertain events")
        if uncertain:
            status("Requires review", "amber")
            st.dataframe(pd.DataFrame(uncertain)[[
                "track_id", "crossing_timestamp", "signal_state", "reason",
            ]], hide_index=True, column_config={
                "track_id": "Track ID", "crossing_timestamp": st.column_config.NumberColumn("Crossing time", format="%.3f s"),
                "signal_state": "Signal", "reason": "Reason",
            })
        else:
            st.caption("No uncertain crossing events in this run.")
    with tracks:
        st.subheader("Vehicle tracking results")
        kpis([(icon, label, metrics.get(label), "blue") for icon, label in (
            ("◎", "Vehicle detections"), ("⌖", "Unique vehicle tracks"),
            ("↗", "Vehicles approaching"), ("⇥", "Vehicles crossing"),
        )])
        st.caption("Detections are counted per frame. Track IDs and trajectories are visible in the annotated video.")
    with video:
        st.subheader("Annotated video")
        if Path(result["video"]).is_file():
            st.video(str(result["video"]))
        else:
            empty_state("Annotated video unavailable", "Check the output folder for this run.")
    with evidence:
        st.subheader("Evidence gallery")
        if not confirmed:
            empty_state("No evidence to review", "Evidence is saved for confirmed violations.")
        for row in confirmed:
            with st.expander(f"{row['violation_id']} · Track #{row['track_id']} · {row['crossing_timestamp']:.2f} s"):
                st.caption(f"{row['signal_state']} signal · Vehicle confidence {row['vehicle_confidence']:.1%} · Plate {row['plate_number']}")
                if row.get("evidence_image_path") and Path(row["evidence_image_path"]).is_file():
                    st.image(row["evidence_image_path"], caption="Annotated crossing evidence")
                for column, key, caption in zip(st.columns(3),
                                                ("original_image_path", "vehicle_crop_path", "plate_crop_path"),
                                                ("Original crossing frame", "Vehicle crop", "Plate crop")):
                    if row.get(key) and Path(row[key]).is_file():
                        column.image(row[key], caption=caption)
    with reports:
        st.subheader("CSV reports")
        for key, label in (("report", "Download confirmed CSV"), ("uncertain_report", "Download uncertain CSV")):
            path = Path(result[key])
            if path.is_file():
                download_csv(path, label, f"rlvd_{key}_download")


def queue_rlvd():
    st.session_state.rlvd_queued = True


def render_rlvd(root, settings):
    state = st.session_state
    page_header("Red-Light Violation Detection",
                "Configure the traffic signal and stop line, then verify crossings against signal history.", "DETECTION")
    st.subheader("Video input")
    uploaded = st.file_uploader("Upload an RLVD video", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
                                key="rlvd_upload", help="MP4, MOV, AVI, MKV, WEBM, or M4V.")
    removed = state.get("rlvd_uploader_populated", False) and uploaded is None
    state.rlvd_uploader_populated = uploaded is not None
    if uploaded is None:
        if not removed and state.get("rlvd_path"):
            removed = st.button("Remove staged video", key="rlvd_remove_staged")
        if removed:
            state.pop("rlvd_temp", None)
            state.pop("rlvd_path", None)
            state.pop("rlvd_upload_id", None)
            state.rlvd_result = None
        if not state.get("rlvd_path"):
            empty_state("No red-light video selected", "Upload a video to configure signal and stop-line geometry.")
            return
    try:
        if uploaded is not None and state.get("rlvd_upload_id") != uploaded.file_id:
            content = uploaded.getbuffer()
            video_id = hashlib.sha256(content).hexdigest()
            temp = tempfile.TemporaryDirectory(prefix="traffic-rlvd-")
            path = Path(temp.name) / ("source" + Path(uploaded.name).suffix.lower())
            path.write_bytes(content)
            metadata = video_metadata(path)
            state.rlvd_temp = temp
            state.rlvd_path = str(path)
            state.rlvd_metadata = metadata
            state.rlvd_video_id = video_id
            state.rlvd_upload_id = uploaded.file_id
            state.rlvd_video_name = uploaded.name
            state.rlvd_config_id = None
            state.rlvd_name = Path(state.rlvd_video_name).stem
            state.rlvd_approach = "Bottom"
            state.rlvd_lamp_layout = "Automatic"
            state.rlvd_result = None
            state.rlvd_created = None
            reset_editor()
        width, height, fps, count = state.rlvd_metadata
        path = state.rlvd_path
        st.markdown(f"**{state.rlvd_video_name}**")
        st.caption(f"{width} × {height} · {fps:.3f} FPS · {count:,} frames · {count/fps:.2f} seconds")
        with st.expander("Video preview"):
            if Path(path).suffix in (".mp4", ".webm", ".m4v"):
                st.video(path, format="video/webm" if Path(path).suffix == ".webm" else "video/mp4")
            else:
                st.caption("Use the frame selector below to preview this format.")
        st.subheader("Saved configuration")
        folder = root / "configs" / "rlvd"
        saved = {}
        for file in sorted(folder.glob("*.json")):
            try:
                saved[file.stem] = Configuration(**json.loads(file.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
        selected = st.selectbox("Saved configurations", [None, *saved], key="rlvd_saved",
                                format_func=lambda key: saved[key].name if key in saved else "Choose a configuration")
        left, right = st.columns(2)
        selected_config = saved.get(selected)
        compatible = selected_config and (selected_config.width, selected_config.height) == (width, height)
        if selected_config and not compatible:
            st.warning("This configuration uses a different resolution. Redraw the geometry for this video.")
        elif selected_config and selected_config.video_id != state.rlvd_video_id:
            st.warning("This configuration belongs to another video. Verify the overlay before saving a new copy.")
        if left.button("Open selected configuration", disabled=not compatible):
            geometry = {key: getattr(selected_config, key) for key in EMPTY_GEOMETRY}
            reset_editor(geometry)
            same_video = selected_config.video_id == state.rlvd_video_id
            state.rlvd_config_id = selected if same_video else None
            state.rlvd_created = selected_config.created if same_video else None
            state.rlvd_approach = selected_config.approach
            state.rlvd_lamp_layout = selected_config.lamp_layout
            state.rlvd_name = selected_config.name
        if right.button("Reset configuration"):
            reset_editor()
            state.rlvd_config_id = None
            state.rlvd_created = None
            state.rlvd_approach = "Bottom"
            state.rlvd_lamp_layout = "Automatic"
            state.rlvd_name = Path(state.rlvd_video_name).stem
        configuration_status = st.empty()
        st.subheader("Geometry editor")
        st.caption("The signal rectangle isolates the lamps. The stop line defines the crossing boundary. "
                   "Add a lane polygon only when you need to limit the monitored approach.")
        frame_column, approach_column = st.columns(2)
        index = frame_column.number_input("Preview frame (zero-based)", 0, count-1,
                                          key=f"rlvd_frame_{state.rlvd_upload_id}")
        st.caption("Frame time · %.3f s" % (index/fps))
        name = st.text_input("Configuration name", key="rlvd_name")
        approach = approach_column.selectbox("Side vehicles approach from", list(DIRECTIONS), key="rlvd_approach",
                                              help="The side of the image vehicles start from before they cross.")
        approach_column.caption(f"Vehicles enter from the {approach.lower()} side of the image.")
        with st.expander("Signal layout"):
            layout = st.selectbox("Lamp arrangement", LAMP_LAYOUTS, key="rlvd_lamp_layout")
        geometry = EDITOR(
            image=preview_image(path, index), width=width, height=height,
            geometry=state.rlvd_geometry, revision=state.rlvd_revision,
            key=f"rlvd_editor_{state.rlvd_revision}", default=state.rlvd_geometry,
        )
        state.rlvd_geometry = geometry
        for column, label, ready in zip(st.columns(3),
                                        ("Signal rectangle", "Stop line", "Lane polygon"),
                                        (len(geometry["roi"]) == 4, len(geometry["line"]) == 2,
                                         len(geometry["lane"]) >= 3)):
            with column:
                status(f"{label} · {'drawn' if ready else ('optional' if label == 'Lane polygon' else 'required')}",
                       "green" if ready else "neutral")
        st.caption("Include the signal lamps and their housing. Place the stop line on the road marking before the crossing.")
        config = Configuration(name, state.rlvd_video_id, width, height, fps, approach=approach,
                               lamp_layout=layout, **geometry)
        if state.rlvd_created:
            config.created = state.rlvd_created
        try:
            config.validate()
            valid = True
        except ValueError as error:
            st.caption(str(error))
            valid = False
        if st.button("Save configuration", disabled=not valid):
            config_id = state.rlvd_config_id or uuid4().hex
            save_configuration(folder, asdict(config), config_id)
            state.rlvd_config_id, state.rlvd_created = config_id, config.created
            saved[config_id] = config
            st.success(f"Saved {name}.")
        active_config = saved.get(state.get("rlvd_config_id"))
        with configuration_status.container():
            if active_config is None:
                status("Unsaved geometry")
            elif active_config == config:
                status(f"Saved configuration · {active_config.name}", "green")
            else:
                status(f"{active_config.name} · unsaved changes", "amber")
        with st.expander("Advanced detection settings"):
            thresholds = Thresholds(
                signal_frames=st.number_input("Consecutive lamp frames", 2, 30, 3),
                approach_frames=st.number_input("Stable approach frames", 2, 30, 3),
                crossing_frames=st.number_input("Frames beyond the stop line", 2, 30, 2),
                deadband=st.number_input("Stop-line tolerance (pixels)", .5, 30.0, 3.0),
                confidence=st.slider("Minimum track confidence", .15, .95, .35),
                anchor_fraction=st.slider("Anchor height within box (top to bottom)", .5, 1., 1.),
            )
            run_anpr = st.checkbox("Read plates for confirmed violations", value=True)
        st.subheader("Signal verification")
        st.caption(f"Analyze the frames around preview frame {index:,}.")
        signature = (state.rlvd_video_id, geometry, thresholds.signal_frames, layout, index)
        if st.button("Verify signal around the selected frame", disabled=not valid):
            state.rlvd_signal_check = (signature, signal_check(path, config, thresholds.signal_frames, count, index))
        verification = state.get("rlvd_signal_check")
        if verification and verification[0] == signature:
            rows, examples = verification[1]
            selected_reading = next((row for row in rows if row["Frame"] == index), None)
            tones = {"RED": "red", "YELLOW": "amber", "GREEN": "green", "UNKNOWN": "neutral"}
            if selected_reading:
                color = selected_reading["Confirmed"]
                status(f"Frame {index:,} · {color} · Quality {selected_reading['Quality']:.2f}", tones[color])
            for column, color in zip(st.columns(4), tones):
                with column, st.container(border=True):
                    status(color, tones[color])
                    if color in examples:
                        quality, sample_index, image = examples[color]
                        st.image(image, caption=f"Frame {sample_index} · Quality {quality:.2f}")
                    else:
                        st.caption("No sample in this window")
            with st.expander("Signal history"):
                st.dataframe(rows, height=220, hide_index=True)
        st.subheader("Run detection")
        placeholder = st.empty()
        started = placeholder.button("Start Red-Light Violation Detection", type="primary", disabled=not valid,
                                     key="start_rlvd")
        if valid and (state.pop("rlvd_queued", False) or started):
            placeholder.button("Processing video", type="primary", disabled=True, key="rlvd_running")
            state.rlvd_result = None
            bar = st.progress(0, text="Preparing RLVD")
            detector, lock = vehicle_detector()
            with lock:
                state.rlvd_result = process_video(
                    path, config, thresholds, root, settings.get("anpr", {}) if run_anpr else None,
                    lambda fraction, message: bar.progress(fraction, text=message), detector,
                )
            placeholder.button("Start Red-Light Violation Detection", type="primary",
                               key="restart_rlvd", on_click=queue_rlvd)
            st.success("Processing complete. Review the results below.")
        if state.rlvd_result:
            render_results(state.rlvd_result)
    except Exception as error:
        st.error(f"RLVD could not continue. {error}")
