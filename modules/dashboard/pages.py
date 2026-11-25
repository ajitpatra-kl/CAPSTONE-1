"""Read existing reports and present session and historical traffic activity."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from .components import download_csv, empty_state, page_header


REPORT_TYPES = {
    "Overspeed violations": "reports/report_*.csv",
    "Red-light violations": "rlvd/*/rlvd_report.csv",
    "Uncertain red-light events": "rlvd/*/rlvd_uncertain_events.csv",
    "Lane-change violations": "lcvd/*/lane_change_violations.csv",
    "Vehicle tracks": "reports/all_tracks_*.csv",
}


def reports_for(outputs, kind):
    return sorted(outputs.glob(REPORT_TYPES[kind]), key=lambda path: path.stat().st_mtime, reverse=True)


def read_report(path):
    if not path:
        return pd.DataFrame()
    try:
        # Overspeed reports have a metadata preamble before the CSV header.
        return pd.read_csv(path, comment="#", encoding="utf-8-sig", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def evidence_path(outputs, report, value):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    candidate = path if path.is_absolute() else (
        outputs / "evidence" / path.name if report.parent.name == "reports" else report.parent / path)
    try:
        candidate = candidate.resolve()
        if candidate.is_relative_to(outputs.resolve()) and candidate.is_file() and candidate.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp",
        }:
            return candidate
    except (OSError, ValueError):
        pass
    return None


def render_reports(outputs):
    page_header("Violation Records", "Review saved records, inspect evidence, and download the original CSVs.", "RECORDS")
    kind = st.selectbox("Report category", ["Unified sessions", *REPORT_TYPES], key="traffic_report_kind")
    if kind == "Unified sessions":
        from modules.unified.dashboard import render_unified_reports

        render_unified_reports(outputs / "unified")
        return
    reports = reports_for(outputs, kind)
    st.caption("Saved reports across sessions. Each CSV represents its own processing run.")
    if not reports:
        empty_state("No saved reports in this category", "Complete a detection run to generate a report.")
        return
    path = st.selectbox("Saved report", reports, key=f"traffic_report_file_{kind}",
                        format_func=lambda item: (
                            f"{item.parent.name} · {datetime.fromtimestamp(item.stat().st_mtime):%d %b %Y, %H:%M}"
                            if item.parent.parent.name == "rlvd" else item.stem))
    data = read_report(path)
    download_csv(path, "Download original CSV", "traffic_report_download")
    annotated = path.parent / "annotated.mp4"
    if path.parent.parent.name == "rlvd" and annotated.is_file():
        with st.expander("Annotated video from this run"):
            st.video(str(annotated))
    if data.empty:
        empty_state("No records in this report", "The CSV is available to download.")
        return
    if "Registration Number" in data or "plate_number" in data:
        st.caption("Plate and OCR outcomes are included where available. Unverified candidates are marked in the source report.")
    query = st.text_input("Search report", placeholder="Track ID, vehicle class, registration, or status")
    if query.strip():
        mask = data.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
        data = data[mask]
    st.caption(f"{len(data):,} matching records")
    paths = [column for column in data if column.endswith("_path")
             or column in ("Evidence Image", "Original Frame", "Plate Evidence")]
    st.dataframe(data.drop(columns=paths), hide_index=True, width="stretch")
    pictures = "Evidence Image" if "Evidence Image" in data else "evidence_image_path"
    if pictures not in data:
        return
    available = {i: file for i, value in data[pictures].items()
                 if (file := evidence_path(outputs, path, value)) is not None}
    if available:
        track = "Track ID" if "Track ID" in data else "track_id"
        selected = st.selectbox("Evidence record", list(available),
                                format_func=lambda i: f"Track #{data.at[i, track]} · record {i+1}")
        st.image(str(available[selected]), caption=f"Evidence for track #{data.at[selected, track]}")
