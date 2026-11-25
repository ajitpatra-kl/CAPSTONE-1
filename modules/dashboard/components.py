"""Shared Streamlit presentation components and navigation."""

from html import escape
from pathlib import Path

import streamlit as st


PAGES = {
    "unified": "Unified Monitoring",
    "overview": "Overview",
    "reports": "Violation Records",
    "analytics": "Analytics",
    "system": "System & Settings",
}
TOOLS = {
    "speed": "Overspeed Detection",
    "rlvd": "Red-Light Violation Detection",
    "lcvd": "Lane Change Detection",
    "anpr": "ANPR / Number Plate Recognition",
}


def apply_theme():
    css = Path(__file__).with_name("theme.css").read_text(encoding="utf-8-sig")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def navigation():
    with st.sidebar:
        st.markdown(
            '<div class="ti-brand"><span class="ti-brand-icon" aria-hidden="true">TI</span>'
            '<div><strong>Traffic Intelligence</strong><small>Monitoring command centre</small></div></div>',
            unsafe_allow_html=True,
        )
        primary = st.radio("Workspace", list(PAGES), key="traffic_navigation",
                           format_func=PAGES.get, on_change=clear_tool)
        active = st.session_state.get("traffic_tool")
        if active:
            st.caption(f"Standalone · {TOOLS[active]}")
            st.button("Return to workspace", on_click=clear_tool, width="stretch")
        with st.expander("Standalone tools", expanded=active is not None):
            for key, label in TOOLS.items():
                st.button(label, key=f"nav_tool_{key}", on_click=navigate, args=(key,),
                          width="stretch", type="primary" if key == active else "secondary")
        st.markdown('<div class="ti-sidebar-foot">Recorded video analysis</div>', unsafe_allow_html=True)
    return active or primary


def clear_tool():
    st.session_state.pop("traffic_tool", None)


def navigate(page):
    if page in TOOLS:
        st.session_state.traffic_tool = page
    else:
        clear_tool()
        st.session_state.traffic_navigation = page


def page_header(title, description, section="WORKSPACE"):
    st.markdown(
        f'<header class="ti-page"><div class="ti-eyebrow">{escape(section)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></header>',
        unsafe_allow_html=True,
    )


def status(label, tone="neutral"):
    st.markdown(f'<span class="ti-status ti-{tone}"><i></i>{escape(str(label))}</span>',
                unsafe_allow_html=True)


def kpis(items):
    cards = []
    for icon, label, value, tone in items:
        if isinstance(value, int):
            value = f"{value:,}"
        display = "—" if value is None else str(value)
        cards.append(
            f'<div class="ti-kpi"><span class="ti-kpi-icon ti-{tone}" aria-hidden="true">{escape(icon)}</span>'
            f'<strong>{escape(display)}</strong><small>{escape(label)}</small></div>'
        )
    st.markdown(f'<div class="ti-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def empty_state(title, description):
    st.markdown(f'<div class="ti-empty"><strong>{escape(title)}</strong>'
                f'<p>{escape(description)}</p></div>', unsafe_allow_html=True)


def download_csv(path, label, key):
    st.download_button(label, Path(path).read_bytes(), Path(path).name, "text/csv",
                       key=key, on_click="ignore")
