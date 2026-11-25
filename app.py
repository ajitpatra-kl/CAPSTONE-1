"""
AI Traffic Violation Detection System
Main Streamlit Application

Unified dashboard supporting:
- Vehicle Overspeed Detection (ACTIVE)
- Red-Light Violation Detection (ACTIVE)
- Lane Change Detection (ACTIVE)
"""

import streamlit as st
import json
import os
import sys
import tempfile
import hashlib
from contextlib import closing
from pathlib import Path
from math import ceil, floor

from imageio_ffmpeg import read_frames
import pandas as pd
import plotly.graph_objects as go
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.dashboard.components import apply_theme, empty_state, navigate, navigation, page_header
from modules.dashboard.pages import read_report, render_reports
from modules.dashboard.operations import render_analytics, render_operations, render_settings
from modules.speed_detection.video_processor import VideoProcessor
from modules.speed_detection.violation_detector import ViolationDetector
from modules.speed_detection.report_generator import ReportGenerator
from modules.speed_detection.camera_calibration import CalibrationManager

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Load Config ────────────────────────────
def load_config():
    cfg_path = PROJECT_ROOT / "config" / "settings.json"
    with cfg_path.open(encoding="utf-8-sig") as f:
        return json.load(f)

@st.cache_resource
def load_dataset_config():
    cfg_path = PROJECT_ROOT / "config" / "dataset_config.json"
    with open(cfg_path) as f:
        return json.load(f)

config = load_config()
dataset_config = load_dataset_config()

# ─────────────────────────── Session State ──────────────────────────
def init_session():
    defaults = {
        "current_module": None,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "processing_complete": False,
        "processing_results": None,
        "violations": [],
        "analytics": {},
        "speed_limit": config["speed_estimation"]["default_speed_limit_kmh"],
        "confidence": config["detection"]["confidence_threshold"],
        "smoothing_window": config["speed_estimation"]["smoothing_window"],
        "violation_duration": config["violation"]["min_violation_duration_seconds"],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def render_speed_configuration():
    st.subheader("Processing configuration")
    with st.container(border=True):
        st.session_state.speed_limit = st.slider(
            "Speed Limit (km/h)", 20, 130, int(st.session_state.speed_limit), step=5,
            help="Tracks must exceed this limit long enough to confirm an overspeed event.",
        )
        with st.expander("Advanced detection settings"):
            left, right = st.columns(2)
            with left:
                st.session_state.confidence = st.slider(
                    "Detection Confidence", .1, .9, st.session_state.confidence, step=.05,
                )
                st.session_state.smoothing_window = st.slider(
                    "Smoothing Window (frames)", 5, 21, st.session_state.smoothing_window, step=2,
                )
            with right:
                st.session_state.violation_duration = st.slider(
                    "Min Duration Above Limit (sec)", .5, 5., st.session_state.violation_duration, step=.5,
                )
                st.caption("The application selects a calibration profile using the video checksum and resolution.")


# ─────────────────────────── Video Upload ────────────────────────────
def render_video_upload():
    st.subheader("Video input")
    uploaded = st.file_uploader(
        "Upload a traffic video",
        key="speed_upload",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV. Max size: 500MB",
    )
    removed = st.session_state.get("speed_uploader_populated", False) and uploaded is None
    st.session_state.speed_uploader_populated = uploaded is not None
    if not uploaded and not removed and st.session_state.uploaded_video_path:
        if st.button("Remove staged video", key="speed_remove_staged"):
            removed = True
    if uploaded is not None:
        upload_key = (uploaded.name, hashlib.sha256(uploaded.getbuffer()).hexdigest())
        video_path = st.session_state.uploaded_video_path
        if st.session_state.get("upload_key") != upload_key or not video_path or not Path(video_path).is_file():
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.getbuffer())
            if video_path:
                Path(video_path).unlink(missing_ok=True)
            video_path = tmp.name
            st.session_state.upload_key = upload_key
            st.session_state.uploaded_video_path = video_path
            st.session_state.uploaded_video_name = uploaded.name
            st.session_state.processing_complete = False
            st.session_state.processing_results = None
        video_name = uploaded.name
        preview = uploaded
    elif not removed and st.session_state.uploaded_video_path and Path(st.session_state.uploaded_video_path).is_file():
        video_path = st.session_state.uploaded_video_path
        video_name = st.session_state.uploaded_video_name
        preview = video_path
    else:
        previous = st.session_state.uploaded_video_path
        if previous:
            Path(previous).unlink(missing_ok=True)
            st.session_state.uploaded_video_path = None
            st.session_state.uploaded_video_name = None
            st.session_state.processing_complete = False
            st.session_state.processing_results = None
        demo = dataset_config.get("roboflow", {}).get("videos", {}).get("Roboflow_Vehicles", {})
        video_path = str(PROJECT_ROOT / "dataset" / demo["path"]) if demo else None
        if not video_path or not Path(video_path).is_file():
            st.info("The Roboflow demo is unavailable. Upload a video or place vehicles.mp4 in dataset/roboflow.")
            st.info("The ForeSpeed T1P1 benchmark can also be uploaded from the "
                    "[ForeSpeed repository](https://github.com/ampedsoftware/forespeed). "
                    "It has a reference speed of 46.67 km/h (29 mph) and no verified camera calibration.")
            return None
        video_name = Path(video_path).name
        preview = video_path
        st.caption("Using the Roboflow demonstration video.")

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            st.error(f"Could not open {video_name}.")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    duration = None
    try:
        with closing(read_frames(video_path)) as video:
            duration = next(video).get("duration") or None
    except (OSError, RuntimeError, StopIteration):
        pass

    st.markdown(f"**{video_name}**")
    duration_label = f"{duration:.1f} s" if duration is not None else "Duration unavailable"
    st.caption(f"{width} × {height} · {duration_label} · {fps:.2f} nominal FPS · {frame_count:,} frames")

    manager = CalibrationManager(dataset_config)
    profile = manager.detect_profile(video_path, (width, height))
    if profile and manager.get_calibration(profile):
        camera = manager.profiles[profile]["camera"]
        if manager.profiles[profile]["calibration_status"] == "reference":
            st.info(f"Reference calibration: {camera}. Speeds are estimates within the calibrated road area.")
        else:
            st.success(f"Verified calibration: {camera}")
    elif ((config["speed_estimation"].get("assumed_road") or {}).get("image_points") is not None
          or config["speed_estimation"].get("pixel_scale_mpp") is not None):
        st.warning("Approximate speeds and overspeed alerts will use the configured assumptions.")
    elif profile:
        st.warning(f"Calibration unavailable for {manager.profiles[profile]['camera']}. "
                   "Processing will show vehicle tracks and IDs.")
    else:
        st.warning("Uncalibrated video. Speed and overspeed records are unavailable.")

    with st.expander("Video preview", expanded=False):
        st.video(preview)
    return video_path, video_name


# ─────────────────────────── Processing ──────────────────────────────
def queue_overspeed():
    st.session_state.overspeed_queued = True


def render_processing(video_path: str, video_name: str):
    st.subheader("Run detection")
    ready = st.empty()
    start = ready.button("Start Overspeed Detection", type="primary", key="start_overspeed")

    if st.session_state.pop("overspeed_queued", False) or start:
        ready.button("Processing video", type="primary", key="overspeed_running", disabled=True)
        st.session_state.processing_complete = False
        st.session_state.processing_results = None
        
        progress_bar = st.progress(0, text="Initializing...")
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        
        try:
            # Build processor config
            settings = load_config()
            proc_config = {
                "speed_limit": st.session_state.speed_limit,
                "speed_estimation": settings["speed_estimation"],
                "anpr": settings.get("anpr", {}),
                "confidence": st.session_state.confidence,
                "smoothing_window": st.session_state.smoothing_window,
                "violation_duration": st.session_state.violation_duration,
                "model_path": str(PROJECT_ROOT / "models" / "yolo11n.pt"),
                "output_dir": str(PROJECT_ROOT / "outputs"),
                "video_name": video_name,
            }
            
            processor = VideoProcessor(proc_config, dataset_config)
            
            # Progress callback
            def update_progress(pct, msg, log=""):
                progress_bar.progress(min(pct, 1.0), text=msg)
                if log:
                    log_placeholder.markdown(f"```\n{log}\n```")
            
            status_placeholder.info("Analyzing vehicle tracks and speeds.")
            
            results = processor.process(video_path, progress_callback=update_progress)
            results["run_speed_limit"] = proc_config["speed_limit"]
            
            st.session_state.processing_complete = True
            st.session_state.processing_results = results
            st.session_state.violations = results.get("violations", [])
            st.session_state.analytics = results.get("analytics", {})
            
            progress_bar.progress(1.0, text="✅ Processing complete!")
            status_placeholder.success("Processing complete. Review the results below.")
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        finally:
            ready.button("Start Overspeed Detection", type="primary", key="restart_overspeed", on_click=queue_overspeed)

# ─────────────────────────── Results Dashboard ───────────────────────
def render_results_dashboard(results: dict):
    analytics = results.get("analytics", {})
    violations = results.get("violations", [])
    speed_limit = results.get("run_speed_limit", st.session_state.speed_limit)
    
    st.subheader("Processing summary")
    
    # Key metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("🚗 Vehicles Detected", analytics.get("total_detections", 0))
    with c2:
        st.metric("🔢 Unique Tracks", analytics.get("unique_tracks", 0))
    with c3:
        v_count = len(violations)
        evaluation_enabled = results["calibrated"] or results.get("estimated_speed_mode")
        st.metric("🚨 Violations", v_count if evaluation_enabled else "Not evaluated",
                  delta=f"+{v_count}" if v_count > 0 else None,
                  delta_color="inverse")
    with c4:
        avg_spd = analytics.get("avg_speed_kmh", 0)
        st.metric("📈 Avg Speed", f"{avg_spd:.1f} km/h" if avg_spd is not None else "Unavailable")
    with c5:
        max_spd = analytics.get("max_speed_kmh", 0)
        st.metric("⚡ Max Speed", f"{max_spd:.1f} km/h" if max_spd is not None else "Unavailable")
    with c6:
        fps = analytics.get("processing_fps", 0)
        st.metric("🎬 Processing FPS", f"{fps:.1f}")
    
    # Speed distribution
    if analytics.get("speed_history"):
        st.markdown("### 📉 Speed Distribution")
        speeds = analytics["speed_history"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=speeds, 
            nbinsx=30,
            name="Vehicle Speeds",
            marker_color="#4a9eff",
            opacity=0.8
        ))
        fig.add_vline(
            x=speed_limit,
            line_dash="dash", line_color="red",
            annotation_text=f"Speed Limit: {speed_limit} km/h"
        )
        fig.update_layout(
            template="plotly_dark",
            title="Vehicle Speed Distribution",
            xaxis_title="Speed (km/h)",
            yaxis_title="Count",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Speed over time per track
    if analytics.get("track_speeds"):
        st.markdown("### 📈 Speed Profiles per Vehicle")
        track_speeds = analytics["track_speeds"]
        fig2 = go.Figure()
        for track_id, speeds_data in list(track_speeds.items())[:10]:  # top 10 tracks
            if len(speeds_data["times"]) > 1:
                fig2.add_trace(go.Scatter(
                    x=speeds_data["times"],
                    y=speeds_data["speeds"],
                    mode="lines",
                    name=f"Track #{track_id}",
                    line=dict(width=2)
                ))
        fig2.add_hline(
            y=speed_limit,
            line_dash="dash", line_color="red",
            annotation_text=f"Limit: {speed_limit} km/h"
        )
        fig2.update_layout(
            template="plotly_dark",
            title="Speed vs Time per Track",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (km/h)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Benchmark validation
    if results.get("benchmark"):
        bm = results["benchmark"]
        st.markdown("### 🏆 Benchmark Validation (ForeSpeed T1P1)")
        st.metric("Pass reference speed", f"{bm['reference_speed_kmh']:.2f} km/h")
        if bm["estimated_speed_kmh"] is None:
            st.warning(bm["diagnostics"])
        else:
            est, error, pct = st.columns(3)
            est.metric("Estimated speed", f"{bm['estimated_speed_kmh']:.2f} km/h")
            error.metric("Absolute error", f"{bm['abs_error_kmh']:.2f} km/h")
            pct.metric("Percentage error", f"{bm['pct_error']:.1f}%")

# ─────────────────────────── Violations Table ────────────────────────
def render_violations_table(violations: list):
    if not violations:
        st.info("✅ No overspeed violations detected in this video.")
        return
    
    st.subheader("Confirmed overspeed violations")
    
    st.markdown(f"""
    <div class="violation-banner">
        🚨 {len(violations)} overspeed violation(s) detected and recorded
    </div>
    """, unsafe_allow_html=True)
    
    df_data = []
    for v in violations:
        df_data.append({
            "Track ID": f"#{v['track_id']}",
            "Vehicle": v["class_name"].capitalize(),
            "Estimated Speed": f"{v['speed_kmh']:.1f} km/h",
            "Speed Limit": f"{v['speed_limit_kmh']:.1f} km/h",
            "Excess Speed": f"{v['excess_kmh']:.1f} km/h",
            "Timestamp": f"{v['timestamp_sec']:.2f} s",
            "Number Plate Status": violation_plate_outcome(v),
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────── Evidence Viewer ─────────────────────────
def plate_status_label(status):
    return {
        "Not Detected": "Plate Not Detected",
        "Not Readable": "Plate Detected — Text Unreadable",
        "Text Unreadable": "Plate Detected — Text Unreadable",
        "Low Confidence": "Plate Detected — Text Unreadable",
        "Recognized": "Number Recognized",
    }.get(status, status)


def render_engine_reads(engines):
    for name, label in (("easyocr", "EasyOCR"), ("ppocrv4", "PP-OCRv4")):
        report = engines.get(name)
        if report is None:
            continue
        best = report.get("best")
        if report.get("error"):
            st.text(f"{label} error — {report['error']}")
        elif best:
            st.text(f"{label} — {best['raw_text']} ({best['confidence']:.1%})")
        else:
            st.text(f"{label} — no text")


def violation_plate_outcome(violation):
    status = violation.get("anpr_status")
    if status == "Processing Failed":
        return status
    if violation.get("vehicle_number"):
        return violation["vehicle_number"]
    if violation.get("plate_candidate"):
        return f"{violation['plate_candidate']} (Unverified)"
    if violation.get("plate_bbox") or violation.get("plate_crop_path"):
        return "Plate Detected — Text Unreadable"
    if status in ("Not Detected", "Plate Not Detected"):
        return "Plate Not Detected"
    if status in ("Text Unreadable", "Not Readable", "Low Confidence"):
        return "Plate Detected — Text Unreadable"
    return plate_status_label(status) if status else "ANPR unavailable"


def evidence_image_path(violation, field):
    path = violation.get(field)
    return Path(path) if path and Path(path).is_file() else None


def violation_vehicle_image(violation):
    saved = evidence_image_path(violation, "vehicle_crop_path")
    if saved is not None:
        return str(saved), violation.get("plate_frame_number")

    original = evidence_image_path(violation, "original_frame_path")
    bbox = violation.get("bbox")
    if original is None or bbox is None:
        return None, None
    try:
        frame = cv2.imdecode(np.frombuffer(original.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, None
        left, top, right, bottom = bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, floor(left)), max(0, floor(top))
        x2, y2 = min(width, ceil(right)), min(height, ceil(bottom))
    except (OSError, ValueError, TypeError, OverflowError, cv2.error):
        return None, None
    if x1 >= x2 or y1 >= y2:
        return None, None
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), violation.get("frame_number")


def image_frame_caption(label, frame):
    return f"{label} · frame {frame}" if frame is not None else f"{label} · source frame unavailable"


def render_plate_pair(original, enhanced, frame):
    before, after = st.columns(2)
    with before:
        st.markdown("**Original Number Plate**")
        st.image(str(original), caption=image_frame_caption("Native crop", frame))
    with after:
        st.markdown("**Enhanced Number Plate**")
        if enhanced is not None:
            st.image(str(enhanced), caption=image_frame_caption("Enhanced copy", frame))
        else:
            st.info("Enhanced plate image unavailable.")


def render_evidence_viewer(violations: list):
    if not violations:
        return
    st.subheader("Violation evidence")
    for v in violations:
        outcome = violation_plate_outcome(v)
        with st.expander(f"Track #{v['track_id']} · {v['class_name'].capitalize()} · {v['speed_kmh']:.1f} km/h"):
            st.write(f"Speed limit {v['speed_limit_kmh']:.1f} km/h · "
                     f"Excess {v['excess_kmh']:.1f} km/h · "
                     f"Timestamp {v['timestamp_sec']:.2f} s · Violation frame {v['frame_number']}")
            vehicle, context = st.columns(2)
            with vehicle:
                st.markdown("**Vehicle**")
                image, frame = violation_vehicle_image(v)
                if image is not None:
                    st.image(image, caption=image_frame_caption("Vehicle crop", frame))
                else:
                    st.info("Vehicle crop unavailable.")
            with context:
                annotated = evidence_image_path(v, "evidence_path")
                if annotated is not None:
                    st.image(str(annotated), caption=image_frame_caption("Vehicle bounding box", v.get("frame_number")))
            st.markdown("**Number plate**")
            st.write(outcome)
            path = evidence_image_path(v, "plate_crop_path")
            if path is not None:
                render_plate_pair(path, evidence_image_path(v, "plate_enhanced_path"), v.get("plate_frame_number"))
            elif v.get("plate_crop_path") or v.get("plate_bbox") or outcome == "Plate Detected — Text Unreadable":
                st.info("Plate crop image unavailable.")
            plate_evidence = evidence_image_path(v, "plate_evidence_path")
            if plate_evidence is not None and (annotated is None or v.get("plate_frame_number") != v.get("frame_number")):
                with st.popover("Plate bounding box"):
                    st.image(str(plate_evidence), caption=image_frame_caption("Plate bounding box", v.get("plate_frame_number")))
            score = v.get("plate_detection_confidence")
            if score is not None:
                st.caption(f"Plate detector confidence {score:.1%}")


def render_evidence_details(violations):
    if not violations:
        return
    index = st.selectbox(
        "Violation details", range(len(violations)),
        format_func=lambda i: f"Track #{violations[i]['track_id']} · frame {violations[i]['frame_number']}",
    )
    v = violations[index]
    st.write(f"Plate box {v.get('plate_bbox') or '—'} · Plate source frame "
             f"{v['plate_frame_number'] if v.get('plate_frame_number') is not None else '—'}")
    if v.get("plate_raw_text"):
        st.text(f"Raw OCR text — {v['plate_raw_text']}")
    if v.get("plate_confidence") is not None:
        st.caption(f"OCR confidence {v['plate_confidence']:.1%}")
    if v.get("plate_candidate"):
        st.caption(f"Supporting frames {v.get('plate_candidate_frames', ())} · "
                   f"Indian format match {'yes' if v.get('plate_candidate_format') else 'no'}")
    render_engine_reads(v.get("plate_engine_results", {}))
    for field, label in (
        ("evidence_path", image_frame_caption("Annotated violation evidence", v.get("frame_number"))),
        ("original_frame_path", image_frame_caption("Original violation frame", v.get("frame_number"))),
        ("plate_evidence_path", image_frame_caption("Annotated plate evidence", v.get("plate_frame_number"))),
    ):
        path = evidence_image_path(v, field)
        if path is not None:
            st.image(str(path), caption=label)
        elif v.get(field):
            st.caption(f"{label} · file unavailable")


def render_plate_scan(scan):
    if not scan:
        return
    st.subheader("Number plate observations")
    if scan.get("error"):
        st.warning(scan["error"])
    summary = scan["summary"]
    st.write(f"{summary['detected_plates']} detections · "
             f"{summary['tracks_with_detections']} tracks with plates · "
             f"{summary.get('consistent_tracks', 0)} consistent unverified candidates · "
             f"{summary['recognized_tracks']} verified registrations")
    tracks = scan.get("tracks", {})
    if not tracks:
        return
    track = st.selectbox("Vehicle tracking ID", sorted(tracks), format_func=lambda tid: f"#{tid}")
    st.caption(plate_status_label(tracks[track]))
    candidate = scan.get("candidates", {}).get(str(track))
    if candidate and candidate.get("text"):
        st.caption(f"OCR candidate (unverified) · {candidate['text']} · "
                   f"Registration format {'matched' if candidate.get('format_valid') else 'failed validation'}")
    plates = [plate for plate in scan["plates"] if str(plate["track_id"]) == str(track)]
    for plate in plates:
        with st.expander(f"Frame {plate['frame']} · {plate_status_label(plate['status'])}"):
            crop = evidence_image_path(plate, "crop_path")
            if crop is not None:
                enhanced = evidence_image_path(plate, "enhanced_crop_path")
                if enhanced is None:
                    enhanced = evidence_image_path(plate.get("ocr_inputs", {}), "sharpened")
                render_plate_pair(crop, enhanced, plate["frame"])
            else:
                st.info("Plate crop image unavailable.")
            width, height = plate["size"]
            st.write(f"Track #{track} · box {tuple(plate['bbox'])} · "
                     f"{width}×{height} native pixels · detector {plate['detection_confidence']:.1%}")
            if plate["raw_text"]:
                st.text(f"OCR text {plate['raw_text']} · confidence {plate['ocr_confidence']:.1%}")
            render_engine_reads(plate.get("engines", {}))
            if plate["text_agreement"] and not plate["registration"]:
                st.caption("Text agreed across frames. The registration remains unverified.")
            if plate["error"]:
                st.text(plate["error"])


# ─────────────────────────── Download Reports ────────────────────────
def render_downloads(results: dict):
    st.subheader("Download results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_path = results.get("csv_report_path")
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    "📄 Download CSV Report",
                    f.read(),
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col1:
        tracks_csv = results.get("tracks_csv_path")
        if tracks_csv and os.path.exists(tracks_csv):
            st.download_button("Download vehicle CSV", Path(tracks_csv).read_bytes(),
                               file_name=Path(tracks_csv).name, mime="text/csv")

    with col2:
        vid_path = results.get("output_video_path")
        if vid_path and os.path.exists(vid_path):
            with open(vid_path, "rb") as f:
                st.download_button(
                    "🎬 Download Annotated Video",
                    f.read(),
                    file_name=os.path.basename(vid_path),
                    mime="video/mp4",
                    use_container_width=True
                )
    
    with col3:
        violations = results.get("violations", [])
        available = [v for v in violations if evidence_image_path(v, "evidence_path") is not None]
        if available:
            selected = st.selectbox(
                "Violation evidence to download", range(len(available)),
                format_func=lambda i: f"Track #{available[i]['track_id']} · frame {available[i]['frame_number']}",
            )
            path = evidence_image_path(available[selected], "evidence_path")
            if path is not None:
                st.download_button("Download violation evidence", path.read_bytes(), file_name=path.name)
        out_dir = str(PROJECT_ROOT / "outputs")
        st.caption(f"Outputs saved to {out_dir}")

# ─────────────────────────── Speed Module Main ───────────────────────
def render_speed_module():
    page_header("Overspeed Detection", "Measure vehicle speeds and review confirmed overspeed events.", "DETECTION")
    selected_video = render_video_upload()
    render_speed_configuration()
    if not selected_video:
        return
    render_processing(*selected_video)

    results = st.session_state.processing_results
    if not st.session_state.processing_complete or not results:
        return

    st.divider()
    summary, vehicles, violations, evidence, video, downloads = st.tabs(
        ["Summary", "Vehicles", "Violations", "Evidence", "Annotated video", "Downloads"],
    )
    with summary:
        st.info(results["calibration_status"])
        analytics = results["analytics"]
        st.caption(f"{analytics['tracker']} · {analytics['processed_frames']:,} frames processed")
        if analytics["invalid_timestamps"]:
            st.warning(f"Invalid frame timestamps: {analytics['invalid_timestamps']}")
        render_results_dashboard(results)
    with vehicles:
        st.subheader("Detected vehicle tracks")
        path = results.get("tracks_csv_path")
        data = read_report(Path(path)) if path else pd.DataFrame()
        if data.empty:
            empty_state("No vehicle tracks available", "Tracks will appear here when the detector finds vehicles.")
        else:
            def track_highlight(row):
                confirmed = str(row.get("Violation Confirmed", "")).casefold() in ("yes", "true")
                return ["background-color: #402332" if confirmed else "" for _ in row]

            st.dataframe(data.style.apply(track_highlight, axis=1), hide_index=True)
    with violations:
        if results["calibrated"] or results.get("estimated_speed_mode"):
            render_violations_table(results.get("violations", []))
        else:
            st.info("Overspeed evaluation requires calibration or configured speed assumptions.")
    with evidence:
        if not results.get("violations"):
            empty_state("No violation evidence", "Evidence appears when a run confirms an overspeed event.")
        render_evidence_viewer(results.get("violations", []))
        if results.get("violations") or results.get("plate_scan"):
            with st.expander("Technical details and plate observations"):
                render_evidence_details(results.get("violations", []))
                render_plate_scan(results.get("plate_scan"))
    with video:
        out = results.get("output_video_path")
        if out and Path(out).is_file():
            st.video(out)
        else:
            empty_state("Annotated video unavailable", "Check the processing outputs for this run.")
    with downloads:
        render_downloads(results)


def render_anpr():
    page_header("ANPR / Number Plate Recognition",
                "Inspect plate detections, OCR reads, and registration validation from processed videos.", "IDENTIFICATION")
    from modules.unified.dashboard import render_unified_anpr

    render_unified_anpr(PROJECT_ROOT / "outputs" / "unified")
    st.divider()
    st.subheader("Standalone diagnostics")
    speed = st.session_state.processing_results or {}
    red_light = st.session_state.get("rlvd_result") or {}
    scan, violations = speed.get("plate_scan"), speed.get("violations", [])
    red_records = red_light.get("confirmed", [])
    manifests = sorted((PROJECT_ROOT / "outputs" / "evidence").glob("*/plates.json"),
                       key=lambda item: item.stat().st_mtime, reverse=True)
    if not scan and not violations and not red_records and not manifests:
        empty_state("No plate observations in this session",
                    "Run Overspeed Detection or enable plate reading for a red-light run. Saved outcomes are available in Violation Reports.")
    left, right = st.columns(2)
    left.button("Open Overspeed workflow", on_click=navigate, args=("speed",))
    right.button("Open Red-Light workflow", on_click=navigate, args=("rlvd",))
    if manifests:
        selected = st.selectbox("Plate observation source", [None, *manifests], key="anpr_source",
                                format_func=lambda path: f"Saved · {path.parent.name}" if path else "Current session")
        if selected is not None:
            try:
                saved_scan = json.loads(selected.read_text(encoding="utf-8"))
                render_plate_scan(saved_scan)
            except (OSError, ValueError, KeyError, TypeError):
                st.warning("This saved plate observation file could not be displayed.")
            return
        if not scan and not violations and not red_records:
            empty_state("No plate observations in this session", "Select a saved run above or start a detection workflow.")
    if scan or violations:
        st.subheader("Overspeed run")
        render_plate_scan(scan)
        if violations:
            st.dataframe([{"Track ID": row["track_id"], "Plate outcome": violation_plate_outcome(row)}
                          for row in violations], hide_index=True)
            render_evidence_viewer(violations)
            with st.expander("OCR and validation details"):
                render_evidence_details(violations)
    if red_records:
        st.subheader("Red-light run")
        for row in red_records:
            with st.expander(f"{row['violation_id']} · Track #{row['track_id']}"):
                st.write(row.get("plate_number") or "Not Readable")
                crop = evidence_image_path(row, "plate_crop_path")
                if crop is not None:
                    st.image(str(crop), caption="Original plate crop")


def main():
    apply_theme()
    state = st.session_state
    previous = state.current_module
    page = navigation()
    if page != previous:
        # File widgets are remounted on navigation. The staged paths remain usable.
        state.speed_uploader_populated = False
        state.rlvd_uploader_populated = False
        if previous == "rlvd":
            # Keep the form values when Streamlit removes widgets from the inactive page.
            for key in ("rlvd_saved", "rlvd_name", "rlvd_approach", "rlvd_lamp_layout",
                        f"rlvd_frame_{state.get('rlvd_upload_id')}"):
                if key in state:
                    state[key] = state[key]
    if previous == "unified" and page != previous:
        prefixes = ("unified_preview_", "unified_rlvd_", "unified_lcvd_", "unified_red_",
                    "unified_lane_", "unified_calibration_", "unified_speed_")
        state.unified_setup_draft = {key: state[key] for key in list(state)
                                     if key == "unified_selected" or key.startswith(prefixes)}
    state.current_module = page
    if page == "overview":
        render_operations(PROJECT_ROOT, state.get("unified_summary"), config)
    elif page == "analytics":
        render_analytics(PROJECT_ROOT)
    elif page == "unified":
        from modules.unified.dashboard import render_unified

        render_unified(PROJECT_ROOT, config, dataset_config)
    elif page == "speed":
        render_speed_module()
    elif page == "rlvd":
        from modules.rlvd.dashboard import render_rlvd

        render_rlvd(PROJECT_ROOT, config)
    elif page == "lcvd":
        from modules.lcvd.dashboard import render_lcvd

        render_lcvd(PROJECT_ROOT)
    elif page == "anpr":
        render_anpr()
    elif page == "reports":
        render_reports(PROJECT_ROOT / "outputs")
    else:
        render_settings(PROJECT_ROOT, config)


if __name__ == "__main__":
    main()
