"""Streamlit workspace for configuring and running lane-change detection."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.dashboard.components import download_csv, empty_state, kpis, page_header, status
from modules.rlvd.configuration import video_metadata
from modules.rlvd.dashboard import preview_image, vehicle_detector

from .configuration import Configuration, Settings
from .video_processor import process_video


EDITOR = components.declare_component("lcvd_boundaries", path=str(Path(__file__).with_name("editor")))


def reset_input():
    state = st.session_state.lcvd_workspace
    state.clear()
    state["upload_key"] = uuid4().hex


def set_draft(state, config, config_id=None):
    state.update(config=config, config_id=config_id, revision=uuid4().hex)


def queue_run():
    st.session_state.lcvd_workspace["queued"] = True


def event_table(rows):
    st.dataframe(pd.DataFrame(rows)[[
        "event_id", "violation_id", "track_id", "vehicle_class", "source_lane", "destination_lane",
        "timestamp_sec", "boundary", "boundary_type", "confidence", "status", "reason",
    ]], hide_index=True, column_config={
        "event_id": "Event ID", "violation_id": "Violation ID", "track_id": "Track ID",
        "vehicle_class": "Vehicle", "source_lane": "From lane", "destination_lane": "To lane",
        "timestamp_sec": st.column_config.NumberColumn("Crossing time", format="%.3f s"),
        "boundary": "Boundary", "boundary_type": "Type",
        "confidence": st.column_config.NumberColumn("Confidence", format="percent"),
        "status": "Status", "reason": "Reason",
    })


def evidence_package(directory):
    package = BytesIO()
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        for filename in ("event.json", "before.png", "crossing.png", "vehicle.png"):
            archive.write(directory / filename, filename)
    return package.getvalue()


def render_results(result):
    st.divider()
    st.subheader("Detection results")
    tones = ("blue", "blue", "blue", "red", "green", "amber")
    icons = ("◎", "⌖", "⇄", "!", "✓", "?")
    kpis([(icon, label, value, tone) for (label, value), icon, tone
          in zip(result["metrics"].items(), icons, tones)])
    st.caption("Vehicle detections are counted per frame. Confirmed events include permitted changes and restricted crossings.")
    events, violations, uncertain, video, reports = st.tabs(
        ["Lane change events", "Confirmed violations", "Uncertain events", "Annotated video", "Downloads"],
    )
    for tab, key, title in (
        (events, "events", "No confirmed lane changes"),
        (violations, "violations", "No restricted crossings confirmed"),
        (uncertain, "uncertain", "No uncertain events"),
    ):
        with tab:
            if result[key]:
                event_table(result[key])
            else:
                empty_state(title, "No records in this category for the completed run.")
    if result["violations"]:
        with violations:
            rows = result["violations"]
            selection = st.selectbox("Evidence record", range(len(rows)), format_func=lambda i: (
                f"{rows[i]['violation_id']} · Track #{rows[i]['track_id']} · {rows[i]['timestamp_sec']:.2f} s"
            ))
            row = rows[selection]
            st.caption(f"Lane {row['source_lane']} → {row['destination_lane']} · Restricted boundary B{row['boundary']}")
            st.image(row["evidence_image_path"], caption="Annotated crossing")
            left, right = st.columns(2)
            left.image(row["original_frame_path"], caption="Original frame before crossing")
            right.image(row["vehicle_crop_path"], caption="Vehicle crop")
            directory = Path(row["evidence_image_path"]).parent
            st.download_button("Download evidence package", lambda: evidence_package(directory),
                               f"{row['violation_id']}.zip", "application/zip",
                               key=f"lcvd_evidence_{row['event_id']}", on_click="ignore")
    with video:
        st.video(str(result["video"]))
    with reports:
        for key, title in (("events", "Lane-change events CSV"), ("violations", "Confirmed violations CSV"),
                           ("uncertain", "Uncertain events CSV")):
            download_csv(result["reports"][key], f"Download {title}", f"lcvd_{key}_csv")
        st.download_button("Download annotated MP4", result["video"].read_bytes, result["video"].name,
                           "video/mp4", key="lcvd_video_download", on_click="ignore")


def render_lcvd(root):
    page_header("Lane Change Detection", "Draw lane boundaries and review permitted and restricted crossings.", "DETECTION")
    state = st.session_state.setdefault("lcvd_workspace", {"upload_key": uuid4().hex})
    st.subheader("Video input")
    upload = st.file_uploader("Upload an LCVD video", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
                              key=f"lcvd_upload_{state['upload_key']}",
                              help="Use constant-frame-rate footage from a fixed camera.")
    if state.get("path") or upload:
        st.button("Remove LCVD video", on_click=reset_input)
    try:
        if upload is not None and upload.file_id != state.get("file_id"):
            temporary = tempfile.TemporaryDirectory(prefix="traffic-lcvd-")
            path = Path(temporary.name) / ("source" + Path(upload.name).suffix.lower())
            data = upload.getbuffer()
            path.write_bytes(data)
            width, height, fps, count = video_metadata(path)
            config = Configuration(Path(upload.name).stem, sha256(data).hexdigest(), width, height, fps, [])
            state.update(temp=temporary, path=path, file_id=upload.file_id, filename=upload.name,
                         frame_count=count, result=None, preview_frame=0)
            set_draft(state, config)
        if not state.get("path"):
            empty_state("No lane-change video selected", "Upload a highway video to start drawing lane boundaries.")
            return

        config = state["config"]
        path = state["path"]
        st.write(state["filename"])
        st.caption(f"{config.width} × {config.height} · {config.fps:.3f} FPS · "
                   f"{state['frame_count']:,} frames · {state['frame_count']/config.fps:.2f} s")
        with st.expander("Video preview"):
            if path.suffix in (".mp4", ".webm", ".m4v"):
                st.video(str(path), format="video/webm" if path.suffix == ".webm" else "video/mp4")
            else:
                st.caption("Use the frame selector below to preview this format.")

        st.subheader("Saved configuration")
        folder = root / "configs" / "lcvd"
        saved = {}
        for file in sorted(folder.glob("*.json")):
            try:
                saved[file.stem] = Configuration.load(file)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        selected = st.selectbox("Saved lane configurations", [None, *saved],
                                format_func=lambda key: saved[key].name if key in saved else "Choose a configuration")
        found = saved.get(selected)
        compatible = found and (found.width, found.height) == (config.width, config.height)
        if found and not compatible:
            st.warning("This configuration uses a different resolution. Draw new geometry for this video.")
        elif found and found.video_id != config.video_id:
            st.warning("This configuration belongs to another video. Verify its overlay before saving a copy.")
        if st.button("Open selected configuration", disabled=not compatible):
            same_video = found.video_id == config.video_id
            found.video_id, found.fps = config.video_id, config.fps
            set_draft(state, found, selected if same_video else None)
            config = found
        revision = state["revision"]
        name = st.text_input("Configuration name", config.name, key=f"lcvd_name_{revision}")
        st.subheader("Lane geometry")
        st.caption("Draw the outside edges and each divider over a shared stretch of road. "
                   "The shaded regions between adjacent lines are the monitored lanes.")
        frame_col, axis_col, numbering_col = st.columns(3)
        index = frame_col.number_input("Preview frame", 0, state["frame_count"]-1, state["preview_frame"],
                                      key=f"lcvd_frame_{state['file_id']}")
        state["preview_frame"] = index
        axis = axis_col.selectbox("Road orientation", ["Vertical", "Horizontal"],
                                 index=int(config.axis == "Horizontal"), key=f"lcvd_axis_{revision}",
                                 help="Choose the dominant direction in which the road extends across the frame.")
        directions = ("Left to right", "Right to left") if axis == "Vertical" else ("Top to bottom", "Bottom to top")
        numbering = numbering_col.selectbox("Lane numbering", directions, index=int(config.reverse),
                                            key=f"lcvd_numbering_{revision}_{axis}")
        lines = EDITOR(image=preview_image(str(path), index), width=config.width, height=config.height,
                       axis=axis, reverse=numbering == directions[1], revision=revision, lines=config.lines,
                       default=config.lines, key=f"lcvd_editor_{revision}")
        with st.expander("Confirmation settings"):
            values = config.settings
            left, right = st.columns(2)
            stable = left.number_input("Stable observations per lane", 2, 30, values.stable_frames,
                                       key=f"lcvd_stable_{revision}")
            margin = left.number_input("Boundary tolerance (pixels)", 0., 40., float(values.margin),
                                       key=f"lcvd_margin_{revision}")
            confidence = left.slider("Minimum track confidence", .10, .99, float(values.confidence),
                                     key=f"lcvd_confidence_{revision}")
            gap = right.number_input("Maximum gap before re-seeding (frames)", 1, 60, values.max_gap,
                                     key=f"lcvd_gap_{revision}")
            jump = right.slider("Maximum one-frame shift (fraction of lane width)", .15, 1., float(values.max_jump),
                                key=f"lcvd_jump_{revision}")
        config.name, config.lines, config.axis, config.reverse = name, lines, axis, numbering == directions[1]
        config.settings = Settings(stable, margin, confidence, gap, jump)
        try:
            geometry = config.geometry()
        except ValueError as error:
            geometry = None
            st.warning(str(error))
        if geometry:
            status(f"{len(geometry.boundaries)-1} monitored lanes · {len(lines)} boundaries", "green")
        if st.button("Save lane configuration", disabled=geometry is None):
            config_id = state["config_id"] or uuid4().hex
            config.save(folder, config_id)
            state["config_id"] = config_id
            saved[config_id] = Configuration.load(folder / f"{config_id}.json")
            st.success(f"Saved {config.name}.")
        saved_config = saved.get(state["config_id"])
        if saved_config:
            label = "Saved configuration" if config == saved_config else "Unsaved changes"
            st.caption(f"{label} · {saved_config.name}")

        st.subheader("Run detection")
        action = st.empty()
        clicked = action.button("Start Lane Change Detection", type="primary", disabled=geometry is None)
        if geometry is not None and (state.pop("queued", False) or clicked):
            action.button("Processing lane changes", type="primary", disabled=True)
            state["result"] = None
            bar = st.progress(0, text="Preparing vehicle tracking")
            try:
                detector, lock = vehicle_detector()
                with lock:
                    state["result"] = process_video(path, config, root,
                                                    lambda f, msg: bar.progress(f, text=msg), detector)
            finally:
                action.button("Start Lane Change Detection", type="primary", key="lcvd_restart", on_click=queue_run)
            st.success("Processing complete. Video dimensions, frame rate, and frame count verified.")
        if state.get("result"):
            render_results(state["result"])
    except Exception as error:
        st.error(f"Lane Change Detection could not continue. {error}")
