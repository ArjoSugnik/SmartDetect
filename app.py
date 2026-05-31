"""
AI-Based Image & Video Anomaly Detection System
Main Streamlit application entry point
"""

import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import numpy as np

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SmartDetect · Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from anomaly import detect_image_anomaly, compute_risk_score, STRUCTURAL_ANOMALY_TYPES
from video_processing import (
    process_video_frames, 
    process_camera_frame, 
    get_live_webcam_frame,
    create_background_subtractor,
)
from geo_analysis import compare_geo_images, build_change_table, draw_geo_annotations
from gemini_helper import generate_explanation, chat_with_assistant, analyze_image_structural, check_gemini_status
import time
import cv2
from utils import (
    save_history_entry,
    load_history,
    generate_text_report,
    pil_to_cv2,
    draw_anomaly_overlay,
    draw_ai_visuals,
    generate_ai_heatmap,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;600;700;900&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #060b14;
    --bg-card:      #0d1625;
    --bg-card2:     #111d2e;
    --accent:       #00e5ff;
    --accent2:      #7b2fff;
    --success:      #00e676;
    --warning:      #ffab00;
    --danger:       #ff1744;
    --text-primary: #e8f4fd;
    --text-muted:   #6b8cad;
    --border:       rgba(0,229,255,0.15);
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* Ensure sidebar toggle button is always visible and looks premium */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
    transition: all 0.2s ease-in-out;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(0, 229, 255, 0.1) !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.2) !important;
}

/* ── Animated hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #060b14 0%, #0a1628 40%, #060f20 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 40px,
        rgba(0,229,255,0.02) 40px, rgba(0,229,255,0.02) 41px
    ),
    repeating-linear-gradient(
        90deg, transparent, transparent 40px,
        rgba(0,229,255,0.02) 40px, rgba(0,229,255,0.02) 41px
    );
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin: 0;
    text-shadow: 0 0 30px rgba(0,229,255,0.4);
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
    letter-spacing: 1px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    line-height: 1.2;
    margin-top: 4px;
}

/* ── Section card ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Risk badge ── */
.risk-low    { color: var(--success); background: rgba(0,230,118,0.1);  border: 1px solid rgba(0,230,118,0.3);  }
.risk-medium { color: var(--warning); background: rgba(255,171,0,0.1);  border: 1px solid rgba(255,171,0,0.3);  }
.risk-high   { color: var(--danger);  background: rgba(255,23,68,0.1);   border: 1px solid rgba(255,23,68,0.3);  }
.risk-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ── Progress bar ── */
.score-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* ── Chat bubbles ── */
.chat-user {
    background: rgba(123,47,255,0.15);
    border: 1px solid rgba(123,47,255,0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    text-align: right;
    font-size: 0.9rem;
}
.chat-ai {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: var(--text-primary);
}
.chat-label {
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 3px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(123,47,255,0.1)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.2) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.2) !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
    border-radius: 7px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.12) !important;
    color: var(--accent) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
    border-radius: 8px !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stAlert {
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Sidebar nav pills ── */
.nav-pill {
    display: block;
    padding: 8px 14px;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    text-decoration: none;
    margin-bottom: 4px;
    border: 1px solid transparent;
}
.nav-pill.active {
    background: rgba(0,229,255,0.1);
    border-color: rgba(0,229,255,0.25);
    color: var(--accent);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.25); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "total_scans" not in st.session_state:
    history = load_history()
    st.session_state.total_scans = len(history)


def cached_load_history() -> list:
    """Load history once per Streamlit run and cache in session_state."""
    if "_history_cache" not in st.session_state:
        st.session_state._history_cache = load_history()
    return st.session_state._history_cache


def invalidate_history_cache():
    """Force history to re-read from disk on next access."""
    if "_history_cache" in st.session_state:
        del st.session_state["_history_cache"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-family: Space Mono, monospace; font-size:1.3rem;
                    color: #00e5ff; font-weight:700; letter-spacing:-0.5px;
                    text-shadow: 0 0 20px rgba(0,229,255,0.5)'>⬡ SmartDetect</div>
        <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:2px;
                    text-transform:uppercase; margin-top:4px'>Detection System v2.0</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "🖼️  Image Detection",
            "📷  Camera Detection",
            "🎬  Video Analysis",
            "🌍  Geo Change",
            "🤖  AI Chat",
            "📋  History",
            "ℹ️  About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Live system stats
    history_data = cached_load_history()
    total = len(history_data)
    highrisk = sum(1 for h in history_data if h.get("risk_level") == "HIGH")
    
    gemini_health = check_gemini_status()
    if gemini_health["online"]:
        gemini_status_html = "<span style='color:#00e676; font-family:Space Mono,monospace'>● ONLINE</span>"
    else:
        gemini_status_html = "<span style='color:#ff1744; font-family:Space Mono,monospace'>● OFFLINE</span>"

    st.markdown(
        f"""
    <div style='font-family: Space Mono, monospace; font-size:0.7rem;
                color:#6b8cad; text-transform:uppercase; letter-spacing:1.5px;
                margin-bottom:0.8rem'>System Status</div>
    <div style='display:flex; flex-direction:column; gap:6px'>
        <div style='display:flex; justify-content:space-between; font-size:0.8rem'>
            <span style='color:#6b8cad'>Total Scans</span>
            <span style='color:#00e5ff; font-family:Space Mono,monospace'>{total}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.8rem'>
            <span style='color:#6b8cad'>High Risk</span>
            <span style='color:#ff1744; font-family:Space Mono,monospace'>{highrisk}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.8rem'>
            <span style='color:#6b8cad'>Gemini</span>
            {gemini_status_html}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.68rem; color:#3a5270; text-align:center;"
        "font-family:Space Mono,monospace'>Built with Python · Streamlit<br/>OpenCV · Gemini LLM</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: render risk score widget
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_widget(score: int, risk_level: str):
    color_map = {"LOW": "#00e676", "MEDIUM": "#ffab00", "HIGH": "#ff1744"}
    color = color_map.get(risk_level, "#00e5ff")
    pct = score
    st.markdown(
        f"""
    <div class='section-card'>
        <div class='section-title'>⚡ Risk Assessment</div>
        <div style='display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap'>
            <div style='text-align:center'>
                <div style='font-family:Space Mono,monospace; font-size:2.8rem;
                            font-weight:700; color:{color};
                            text-shadow: 0 0 20px {color}88'>{score}</div>
                <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px'>RISK SCORE</div>
            </div>
            <div style='flex:1'>
                <div class='score-bar-wrap'>
                    <div class='score-bar-fill'
                         style='width:{pct}%; background: linear-gradient(90deg, #00e5ff, {color})'></div>
                </div>
                <div style='display:flex; justify-content:space-between;
                            font-size:0.68rem; color:#3a5270; margin-top:4px'>
                    <span>0 — SAFE</span><span>100 — CRITICAL</span>
                </div>
                <div style='margin-top:10px'>
                    <span class='risk-badge risk-{risk_level.lower()}'>{risk_level}</span>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Dashboard":
    st.markdown(
        """
    <div class='hero-banner'>
        <div>
            <span class='hero-badge'>AI-POWERED</span>
            <span class='hero-badge'>REAL-TIME</span>
            <span class='hero-badge'>LOCAL LLM</span>
        </div>
        <div class='hero-title' style='margin-top:0.8rem'>
            Anomaly Detection System
        </div>
        <div class='hero-sub'>
            Computer vision · Geo change detection · LLM-powered explanations · Risk scoring
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    history_data = cached_load_history()
    total = len(history_data)
    highrisk = sum(1 for h in history_data if h.get("risk_level") == "HIGH")
    mediumrisk = sum(1 for h in history_data if h.get("risk_level") == "MEDIUM")
    lowrisk = sum(1 for h in history_data if h.get("risk_level") == "LOW")

    st.markdown(
        f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-label'>Total Scans</div>
            <div class='metric-value'>{total:03d}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>High Risk</div>
            <div class='metric-value' style='color:#ff1744'>{highrisk:03d}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Medium Risk</div>
            <div class='metric-value' style='color:#ffab00'>{mediumrisk:03d}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Low Risk</div>
            <div class='metric-value' style='color:#00e676'>{lowrisk:03d}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
        <div class='section-card'>
            <div class='section-title'>⬡ System Capabilities</div>
        """,
            unsafe_allow_html=True,
        )
        features = [
            ("🖼️", "Image Anomaly Detection", "OpenCV analysis + Structural AI detection"),
            ("📷", "Camera Real-Time Detection", "Live frame processing with AI snapshots"),
            ("🛠️", "Structural Defect Detection", "Local AI for Scratches, Cracks, Dents, etc."),
            ("🎬", "Video Frame Analysis", "1 FPS extraction with timestamps"),
            ("🌍", "Geo Change Detection", "SSIM + absdiff region classification"),
            ("🤖", "Gemini Vision Support", "Cloud-based Gemini model for structural analysis"),
            ("📊", "Risk Score System", "0–100 scoring with LOW/MED/HIGH"),
            ("📋", "Detection History", "Persistent JSON log with replay"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
            <div style='display:flex; gap:12px; align-items:flex-start;
                        padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.06)'>
                <div style='font-size:1.3rem; min-width:30px'>{icon}</div>
                <div>
                    <div style='font-weight:600; font-size:0.9rem'>{title}</div>
                    <div style='font-size:0.78rem; color:#6b8cad; margin-top:1px'>{desc}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
        <div class='section-card'>
            <div class='section-title'>🚀 Quick Start</div>
        """,
            unsafe_allow_html=True,
        )
        steps = [
            ("01", "Navigate to any module via the sidebar"),
            ("02", "Upload image/video or start camera"),
            ("03", "View detection results & risk score"),
            ("04", "Get AI explanation from Gemini"),
            ("05", "Download report or check history"),
        ]
        for num, text in steps:
            st.markdown(
                f"""
            <div style='display:flex; gap:12px; align-items:flex-start; margin-bottom:12px'>
                <div style='font-family:Space Mono,monospace; font-size:0.7rem;
                            color:#00e5ff; background:rgba(0,229,255,0.08);
                            border:1px solid rgba(0,229,255,0.2); border-radius:6px;
                            padding:2px 8px; white-space:nowrap; min-width:34px;
                            text-align:center'>{num}</div>
                <div style='font-size:0.85rem; color:#a8c4e0; padding-top:1px'>{text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div style='margin-top:1rem; padding:12px; background:rgba(123,47,255,0.08);
                    border:1px solid rgba(123,47,255,0.2); border-radius:8px'>
            <div style='font-size:0.72rem; font-family:Space Mono,monospace;
                        color:#7b2fff; letter-spacing:1px; margin-bottom:6px'>OLLAMA SETUP</div>
            <code style='font-size:0.72rem; color:#a8c4e0; display:block; line-height:1.8'>
            $ gemini pull mistral<br/>
            $ gemini serve
            </code>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if history_data:
            st.markdown(
                """
            <div class='section-card' style='margin-top:0'>
                <div class='section-title'>🕐 Recent Activity</div>
            """,
                unsafe_allow_html=True,
            )
            for entry in reversed(history_data[-4:]):
                risk = entry.get("risk_level", "LOW")
                color = {"LOW": "#00e676", "MEDIUM": "#ffab00", "HIGH": "#ff1744"}.get(
                    risk, "#00e5ff"
                )
                st.markdown(
                    f"""
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:7px 0; border-bottom:1px solid rgba(0,229,255,0.06);
                            font-size:0.8rem'>
                    <span style='color:#a8c4e0'>{entry.get("source","Unknown")[:22]}</span>
                    <span style='color:{color}; font-family:Space Mono,monospace;
                                font-size:0.7rem'>{risk}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🖼️  Image Detection":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>🖼️ Image Anomaly Detection</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload an image for analysis",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP",
    )

    # Detection Mode Selection
    mode_options = ["General Anomaly (OpenCV)"] + STRUCTURAL_ANOMALY_TYPES
    detect_mode = st.selectbox("🎯 Target Anomaly Type", mode_options, index=0)
    
    st.markdown("<br/>", unsafe_allow_html=True)

    if uploaded:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            cv_img = pil_to_cv2(pil_img)
        except Exception as e:
            st.error(f"❌ Could not open image: {e}. Please upload a valid image file.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<div class='section-title'>📥 Original Image</div>",
                unsafe_allow_html=True,
            )
            st.image(pil_img, use_container_width=True)

        with st.spinner(f"🔍 Analyzing image for {detect_mode}..."):
            if detect_mode == "General Anomaly (OpenCV)":
                result = detect_image_anomaly(cv_img)
                score, risk_level = compute_risk_score(result)
                annotated = draw_anomaly_overlay(cv_img, result)
            else:
                # Structural AI Detection
                ai_result = analyze_image_structural(cv_img, detect_mode)
                
                if "error" in ai_result or not ai_result.get("bboxes"):
                    # Fallback to OpenCV if Gemini is not running OR failed to provide boxes
                    cv_result = detect_image_anomaly(cv_img)
                    if "error" in ai_result:
                        score, _ = compute_risk_score(cv_result)
                    else:
                        score = ai_result.get("risk_score", 0)
                    
                    bboxes = []
                    img_h, img_w = cv_img.shape[:2]
                    for c in cv_result.get("contours", []):
                        x, y, w, h = cv2.boundingRect(c)
                        ymin = int((y / img_h) * 1000)
                        xmin = int((x / img_w) * 1000)
                        ymax = int(((y+h) / img_h) * 1000)
                        xmax = int(((x+w) / img_w) * 1000)
                        bboxes.append([ymin, xmin, ymax, xmax])
                        
                    if not bboxes:
                        # Edge-based fallback for thin structures like cracks
                        edges = cv2.Canny(cv_img, 50, 150)
                        
                        # Ignore top 33% (usually sky/distant trees) to prevent massive blobs
                        edges[0:int(img_h * 0.33), :] = 0
                        
                        # Smaller dilation kernel so cracks form distinct boxes
                        edges_dilated = cv2.dilate(edges, np.ones((11, 11), np.uint8), iterations=1)
                        cnts, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        max_area = (img_w * img_h) * 0.8
                        valid_cnts = [c for c in cnts if 200 < cv2.contourArea(c) < max_area]
                        
                        for c in sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:5]:
                            x, y, w, h = cv2.boundingRect(c)
                            ymin = int((y / img_h) * 1000)
                            xmin = int((x / img_w) * 1000)
                            ymax = int(((y+h) / img_h) * 1000)
                            xmax = int(((x+w) / img_w) * 1000)
                            bboxes.append([ymin, xmin, ymax, xmax])
                        
                    if "error" in ai_result:
                        ai_result = {
                            "detected": len(bboxes) > 0,
                            "risk_score": score,
                            "description": f"🔌 [Offline Fallback] OpenCV detected structural irregularities matching {detect_mode}.",
                            "recommendation": "Manual review required. Start Gemini for AI analysis.",
                            "bboxes": bboxes
                        }
                    else:
                        # Append the CV bboxes to the AI result for visual rendering
                        ai_result["bboxes"] = bboxes

                score = ai_result.get("risk_score", 0)
                risk_level = "HIGH" if score > 70 else ("MEDIUM" if score > 30 else "LOW")
                
                # Mock result dict for compatibility with existing UI
                result = {
                    "anomaly_type": detect_mode if ai_result.get("detected") else "None Detected",
                    "contour_count": "AI Managed" if "error" not in ai_result else len(ai_result.get("bboxes", [])),
                    "anomaly_area_pct": score,
                    "description": ai_result.get("description", ""),
                    "recommendation": ai_result.get("recommendation", "")
                }
                annotated = cv_img.copy() # No contours for AI yet

        with col2:
            if detect_mode == "General Anomaly (OpenCV)":
                st.markdown(
                    "<div class='section-title'>🎯 Anomaly Detection</div>",
                    unsafe_allow_html=True,
                )
                st.image(annotated, channels="BGR", use_container_width=True)
            else:
                # Show AI Visual Results in Tabs
                v_tab1, v_tab2 = st.tabs(["🔳 Bounding Boxes", "🔥 Heatmap"])
                with v_tab1:
                    boxed_img = draw_ai_visuals(cv_img, ai_result.get("bboxes", []))
                    st.image(boxed_img, channels="BGR", use_container_width=True)
                with v_tab2:
                    heatmap_img = generate_ai_heatmap(cv_img, ai_result.get("bboxes", []))
                    st.image(heatmap_img, channels="BGR", use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        render_risk_widget(score, risk_level)

        # Detection details
        if detect_mode == "General Anomaly (OpenCV)":
            st.markdown(
                f"""
            <div class='section-card'>
                <div class='section-title'>📊 Detection Details</div>
                <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; flex-wrap:wrap'>
                    <div>
                        <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px;
                                    text-transform:uppercase; font-family:Space Mono,monospace'>Type</div>
                        <div style='font-size:1rem; font-weight:600; margin-top:4px; color:#00e5ff'>
                            {result['anomaly_type']}</div>
                    </div>
                    <div>
                        <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px;
                                    text-transform:uppercase; font-family:Space Mono,monospace'>Contours Found</div>
                        <div style='font-size:1rem; font-weight:600; margin-top:4px; color:#e8f4fd'>
                            {result['contour_count']}</div>
                    </div>
                    <div>
                        <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px;
                                    text-transform:uppercase; font-family:Space Mono,monospace'>Anomaly Area %</div>
                        <div style='font-size:1rem; font-weight:600; margin-top:4px; color:#e8f4fd'>
                            {result['anomaly_area_pct']:.2f}%</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class='section-card'>
                <div class='section-title'>📊 AI Structural Analysis</div>
                <div style='margin-bottom:1rem'>
                    <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px;
                                text-transform:uppercase; font-family:Space Mono,monospace'>Findings</div>
                    <div style='font-size:0.95rem; line-height:1.6; margin-top:6px; color:#e8f4fd'>
                        {result['description']}</div>
                </div>
                <div>
                    <div style='font-size:0.7rem; color:#6b8cad; letter-spacing:1px;
                                text-transform:uppercase; font-family:Space Mono,monospace'>Recommendation</div>
                    <div style='font-size:0.9rem; font-weight:600; margin-top:4px; color:#00e676'>
                        {result['recommendation']}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # AI Explanation
        with st.expander("🤖 Generate AI Explanation (Gemini)", expanded=False):
            if st.button("✨ Get AI Explanation", key="img_explain"):
                with st.spinner("Connecting to Gemini..."):
                    explanation = generate_explanation(result, score, risk_level)
                st.markdown(
                    f"""
                <div class='chat-ai'>
                    <div class='chat-label' style='color:#00e5ff'>SmartDetect · Gemini</div>
                    {explanation}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Save & Download
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 Save to History"):
                save_history_entry(
                    source=uploaded.name,
                    anomaly_type=result["anomaly_type"],
                    score=score,
                    risk_level=risk_level,
                    details=result,
                )
                invalidate_history_cache()
                st.success("✅ Saved to detection history!")

        with col_b:
            report_text = generate_text_report(
                uploaded.name, result, score, risk_level
            )
            st.download_button(
                "📥 Download Report",
                data=report_text,
                file_name=f"anomaly_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
                mime="text/plain",
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CAMERA DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📷  Camera Detection":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>📷 Real-Time Camera Detection</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>ℹ️ How It Works</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0'>
        Capture a photo using your device camera. Each frame is processed through
        the anomaly detection pipeline — contour analysis, thresholding, and risk scoring.
        Results are displayed instantly alongside AI explanation support.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Detection Mode Selection (mirrors Image Detection page)
    cam_mode_options = ["General Anomaly (OpenCV)"] + STRUCTURAL_ANOMALY_TYPES
    cam_detect_mode = st.selectbox("🎯 Target Anomaly Type", cam_mode_options, index=0, key="cam_mode")

    camera_img = st.camera_input("📸 Capture Frame")

    if camera_img:
        try:
            pil_img = Image.open(camera_img).convert("RGB")
            cv_img = pil_to_cv2(pil_img)
        except Exception as e:
            st.error(f"❌ Could not process camera frame: {e}")
            st.stop()

        with st.spinner(f"⚡ Processing frame for {cam_detect_mode}..."):
            if cam_detect_mode == "General Anomaly (OpenCV)":
                result = process_camera_frame(cv_img)
                score, risk_level = compute_risk_score(result)
                annotated = draw_anomaly_overlay(cv_img, result)
                ai_result = None
            else:
                # Structural AI Detection
                ai_result = analyze_image_structural(cv_img, cam_detect_mode)
                
                if "error" in ai_result or not ai_result.get("bboxes"):
                    cv_result = process_camera_frame(cv_img)
                    if "error" in ai_result:
                        score, _ = compute_risk_score(cv_result)
                    else:
                        score = ai_result.get("risk_score", 0)

                    bboxes = []
                    img_h, img_w = cv_img.shape[:2]
                    for c in cv_result.get("contours", []):
                        x, y, w, h = cv2.boundingRect(c)
                        ymin = int((y / img_h) * 1000)
                        xmin = int((x / img_w) * 1000)
                        ymax = int(((y+h) / img_h) * 1000)
                        xmax = int(((x+w) / img_w) * 1000)
                        bboxes.append([ymin, xmin, ymax, xmax])
                        
                    if not bboxes:
                        edges = cv2.Canny(cv_img, 50, 150)
                        edges[0:int(img_h * 0.33), :] = 0
                        edges_dilated = cv2.dilate(edges, np.ones((11, 11), np.uint8), iterations=1)
                        cnts, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        max_area = (img_w * img_h) * 0.8
                        valid_cnts = [c for c in cnts if 200 < cv2.contourArea(c) < max_area]
                        
                        for c in sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:5]:
                            x, y, w, h = cv2.boundingRect(c)
                            ymin = int((y / img_h) * 1000)
                            xmin = int((x / img_w) * 1000)
                            ymax = int(((y+h) / img_h) * 1000)
                            xmax = int(((x+w) / img_w) * 1000)
                            bboxes.append([ymin, xmin, ymax, xmax])
                    if "error" in ai_result:
                        ai_result = {
                            "detected": len(bboxes) > 0,
                            "risk_score": score,
                            "description": f"🔌 [Offline Fallback] OpenCV detected regions matching {cam_detect_mode}.",
                            "recommendation": "Manual review required. Start Gemini for AI analysis.",
                            "bboxes": bboxes
                        }
                    else:
                        ai_result["bboxes"] = bboxes

                score = ai_result.get("risk_score", 0)
                risk_level = "HIGH" if score > 70 else ("MEDIUM" if score > 30 else "LOW")
                result = {
                    "anomaly_type": cam_detect_mode if ai_result.get("detected") else "None Detected",
                    "contour_count": "AI Managed" if "error" not in ai_result else len(ai_result.get("bboxes", [])),
                    "anomaly_area_pct": score,
                    "description": ai_result.get("description", ""),
                    "recommendation": ai_result.get("recommendation", ""),
                }
                annotated = cv_img.copy()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<div class='section-title'>🎯 Processed Frame</div>",
                unsafe_allow_html=True,
            )
            if cam_detect_mode == "General Anomaly (OpenCV)":
                st.image(annotated, channels="BGR", use_container_width=True)
            else:
                # Camera AI Visuals
                c_tab1, c_tab2 = st.tabs(["🔳 Bounding Boxes", "🔥 Heatmap"])
                with c_tab1:
                    boxed_cam = draw_ai_visuals(cv_img, ai_result.get("bboxes", []))
                    st.image(boxed_cam, channels="BGR", use_container_width=True)
                with c_tab2:
                    heatmap_cam = generate_ai_heatmap(cv_img, ai_result.get("bboxes", []))
                    st.image(heatmap_cam, channels="BGR", use_container_width=True)

        with col2:
            render_risk_widget(score, risk_level)
            if cam_detect_mode == "General Anomaly (OpenCV)":
                st.markdown(
                    f"""
                <div class='section-card'>
                    <div class='section-title'>📊 Frame Analysis</div>
                    <div style='font-size:0.85rem; color:#a8c4e0; line-height:2'>
                        <b style='color:#00e5ff'>Type:</b> {result['anomaly_type']}<br/>
                        <b style='color:#00e5ff'>Contours:</b> {result['contour_count']}<br/>
                        <b style='color:#00e5ff'>Area:</b> {result['anomaly_area_pct']:.2f}%<br/>
                        <b style='color:#00e5ff'>Timestamp:</b> {datetime.now():%H:%M:%S}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class='section-card'>
                    <div class='section-title'>📊 AI Structural Analysis</div>
                    <div style='margin-bottom:0.8rem'>
                        <div style='font-size:0.7rem; color:#6b8cad; text-transform:uppercase'>Findings</div>
                        <div style='font-size:0.88rem; color:#e8f4fd; margin-top:4px'>{result['description']}</div>
                    </div>
                    <div>
                        <div style='font-size:0.7rem; color:#6b8cad; text-transform:uppercase'>Recommendation</div>
                        <div style='font-size:0.88rem; font-weight:600; color:#00e676; margin-top:4px'>{result['recommendation']}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        if st.button("💾 Save Frame to History"):
            save_history_entry(
                source="Camera Capture",
                anomaly_type=result["anomaly_type"],
                score=score,
                risk_level=risk_level,
                details=result,
            )
            invalidate_history_cache()
            st.success("✅ Saved!")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VIDEO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎬  Video Analysis":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>🎬 Video Anomaly Analysis</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📤 Upload Video", "🔴 Live Simulation"])

    with tab1:
        video_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Video will be analyzed at 1 frame per second",
        )

        if video_file:
            st.markdown(
                f"""
            <div class='section-card'>
                <div style='font-size:0.85rem; color:#a8c4e0'>
                    📁 <b style='color:#00e5ff'>{video_file.name}</b> · 
                    {video_file.size / 1024:.1f} KB
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            max_frames = st.slider(
                "Max frames to analyze", min_value=5, max_value=60, value=20
            )
            fps_rate = st.slider(
                "Analysis rate (Frames Per Second)", min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                help="Higher FPS means more detailed analysis but takes longer."
            )

            if st.button("▶️ Analyze Video"):
                with st.spinner("🎬 Extracting and analyzing frames..."):
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix="." + video_file.name.split(".")[-1]
                    ) as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    frame_results = process_video_frames(tmp_path, max_frames=max_frames, fps_rate=fps_rate)
                    os.unlink(tmp_path)

                if not frame_results:
                    st.error("❌ Could not extract frames. Check video format.")
                else:
                    st.markdown(
                        f"<div style='color:#00e676; margin:0.5rem 0'>✅ Analyzed {len(frame_results)} frames</div>",
                        unsafe_allow_html=True,
                    )

                    # Summary stats
                    scores = [r["score"] for r in frame_results]
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    if max_score < 30:
                        overall_risk = "LOW"
                    elif max_score < 65:
                        overall_risk = "MEDIUM"
                    else:
                        overall_risk = "HIGH"

                    st.markdown(
                        f"""
                    <div class='metric-row'>
                        <div class='metric-card'>
                            <div class='metric-label'>Frames</div>
                            <div class='metric-value'>{len(frame_results)}</div>
                        </div>
                        <div class='metric-card'>
                            <div class='metric-label'>Avg Score</div>
                            <div class='metric-value'>{avg_score:.0f}</div>
                        </div>
                        <div class='metric-card'>
                            <div class='metric-label'>Peak Score</div>
                            <div class='metric-value' style='color:#ff1744'>{max_score}</div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Frame table
                    st.markdown(
                        "<div class='section-title'>📋 Frame-by-Frame Results</div>",
                        unsafe_allow_html=True,
                    )
                    import pandas as pd

                    df = pd.DataFrame(frame_results)
                    df.columns = [c.replace("_", " ").title() for c in df.columns]
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Visual timeline
                    st.markdown(
                        "<div class='section-title' style='margin-top:1rem'>📈 Risk Timeline</div>",
                        unsafe_allow_html=True,
                    )
                    chart_data = pd.DataFrame(
                        {"Frame": [r["frame"] for r in frame_results], "Risk Score": scores}
                    ).set_index("Frame")
                    st.line_chart(chart_data, color="#00e5ff")

                    save_history_entry(
                        source=video_file.name,
                        anomaly_type="Video Analysis",
                        score=int(max_score),
                        risk_level=overall_risk,
                        details={"frames_analyzed": len(frame_results), "avg_score": avg_score},
                    )
                    invalidate_history_cache()

    with tab2:
        st.markdown(
            """
        <div class='section-card'>
            <div class='section-title'>🔴 Live CCTV Monitoring (SmartDetect)</div>
            <p style='font-size:0.85rem; color:#a8c4e0; margin:0'>
            Connects to your local camera for real-time <b>Motion Detection</b> and <b>Human Tracking</b>. 
            The system uses background subtraction to flag movement and Haar Cascades to identify people.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_ctrl, _ = st.columns([1, 3])
        with col_ctrl:
            sim_frames = st.slider("Monitoring duration (ticks)", 10, 1000, 100)
            live_fps = st.slider("Simulation FPS", 1, 60, 15)
            start_btn = st.button("▶️ Start Live Feed")
            stop_btn = st.button("⏹️ Stop Feed")

        # Handle button clicks (these trigger a rerun)
        if start_btn:
            st.session_state.run_sim = True
            st.session_state._sim_tick = 0
            # Create fresh background subtractor for this session
            st.session_state._back_sub = create_background_subtractor()
        if stop_btn:
            st.session_state.run_sim = False

        if st.session_state.get("run_sim", False):
            frame_placeholder = st.empty()
            prog_placeholder = st.empty()
            
            tick = st.session_state.get("_sim_tick", 0)
            back_sub = st.session_state.get("_back_sub")

            while tick < sim_frames and st.session_state.get("run_sim", False):
                prog_placeholder.progress((tick + 1) / sim_frames)

                frame_img, result = get_live_webcam_frame(tick, back_sub=back_sub)
                if frame_img is None:
                    st.error("⚠️ No webcam detected. Please connect a camera.")
                    st.session_state.run_sim = False
                    break

                score, risk_level = compute_risk_score(result)

                with frame_placeholder.container():
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.image(frame_img, caption=f"SmartDetect Live CCTV Feed — Tick {tick+1}/{sim_frames}", use_container_width=True)
                    with c2:
                        risk_color = "#ff1744" if (result.get("human_detected") or risk_level == "HIGH") else \
                                     ("#ffab00" if risk_level == "MEDIUM" else "#00e676")

                        st.markdown(
                            f"""
                        <div style='text-align:center; padding:1rem; background:var(--bg-card);
                                    border:1px solid var(--border); border-radius:12px'>
                            <div style='font-size:0.7rem; color:#6b8cad; font-family:Space Mono,monospace'>LIVE STREAM</div>
                            <div style='font-size:2.5rem; font-weight:700; color:{risk_color};
                                        font-family:Space Mono,monospace; margin:8px 0'>{score}</div>
                            <div style='color:{risk_color}; font-weight:600'>{risk_level} RISK</div>
                            <div style='font-size:0.78rem; color:#6b8cad; margin-top:6px; font-weight:700'>
                                {result['anomaly_type'].upper()}
                            </div>
                            <div style='font-size:0.65rem; color:#6b8cad; margin-top:8px'>
                                Humans: {result.get('face_count', 0) + result.get('body_count', 0)}<br/>
                                Motion: {result.get('anomaly_area_pct', 0):.1f}%<br/>
                                FPS: {live_fps}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                tick += 1
                time.sleep(1.0 / live_fps)
                
            st.session_state._sim_tick = tick
            if tick >= sim_frames:
                st.session_state.run_sim = False
                st.success("✅ Monitoring session completed.")




# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GEO CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍  Geo Change":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>🌍 Geo Change Detection</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>ℹ️ Satellite / Aerial Image Comparison</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0; line-height:1.7'>
        Upload two images of the same geographic region taken at different times.
        The system aligns images, generates a change heatmap, extracts candidate regions
        via OpenCV, and validates each region using <b style='color:#00e5ff'>Gemini vision AI</b>
        (or CV fallback). Use the confidence slider to filter noise.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        old_img_file = st.file_uploader(
            "📅 OLD Image (Before)", type=["jpg", "jpeg", "png"]
        )
    with col2:
        new_img_file = st.file_uploader(
            "📅 NEW Image (After)", type=["jpg", "jpeg", "png"]
        )

    if old_img_file and new_img_file:
        try:
            old_pil = Image.open(old_img_file).convert("RGB")
            new_pil = Image.open(new_img_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not open image(s): {e}. Please upload valid image files.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<div class='section-title'>🕰️ Before</div>", unsafe_allow_html=True
            )
            st.image(old_pil, use_container_width=True)
        with col2:
            st.markdown(
                "<div class='section-title'>📅 After</div>", unsafe_allow_html=True
            )
            st.image(new_pil, use_container_width=True)

        # Cache geo result so slider changes don't re-run the expensive pipeline
        geo_cache_key = f"{old_img_file.name}_{old_img_file.size}_{new_img_file.name}_{new_img_file.size}"
        if st.session_state.get("_geo_cache_key") != geo_cache_key:
            with st.spinner("🔍 Comparing images — aligning, heatmapping, classifying..."):
                geo_result = compare_geo_images(pil_to_cv2(old_pil), pil_to_cv2(new_pil))
            st.session_state._geo_cache_key = geo_cache_key
            st.session_state._geo_result = geo_result
        else:
            geo_result = st.session_state._geo_result

        if not geo_result["similar"]:
            st.error(
                f"❌ Images are too dissimilar (SSIM: {geo_result['ssim']:.3f} < 0.20). "
                "Please upload images of the same region."
            )
        else:
            # Detection source badge
            source_label = "🤖 Gemini Vision AI" if geo_result.get("gemini_used") else "🔬 CV Analysis (Offline)"
            source_color = "#00e5ff" if geo_result.get("gemini_used") else "#ffab00"
            st.markdown(
                f"""
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:1rem; flex-wrap:wrap'>
                <div style='color:#00e676; font-weight:600; font-size:0.9rem'>
                    ✅ Images matched (SSIM: {geo_result['ssim']:.3f})
                </div>
                <div style='background:rgba(0,229,255,0.08); border:1px solid {source_color}40;
                            color:{source_color}; padding:3px 12px; border-radius:20px;
                            font-family:Space Mono,monospace; font-size:0.72rem;
                            letter-spacing:0.5px'>
                    {source_label}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Confidence filter slider
            min_conf = st.slider(
                "🎚️ Confidence Filter — only show regions above this threshold",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                help="Slide right to hide low-confidence detections. 0 = show all.",
            )

            # Re-annotate with filter applied
            filtered_annotated = draw_geo_annotations(
                geo_result["new_resized"],
                geo_result["regions"],
                min_confidence=min_conf,
            )

            # Always use heatmap as base for the result display
            base_img = geo_result.get("heatmap_img")
            if base_img is None:
                base_img = geo_result["new_resized"]

            final_display = draw_geo_annotations(
                base_img,
                geo_result["regions"],
                min_confidence=min_conf,
            )

            st.markdown(
                "<div class='section-title' style='margin-top:1rem'>🔥 Change Intensity Heatmap</div>",
                unsafe_allow_html=True,
            )
            st.image(
                final_display,
                channels="BGR",
                use_container_width=True,
                caption=f"Heatmap Overlay (Confidence ≥ {int(min_conf*100)}%)"
            )

            # Classification table (filtered)
            changes = build_change_table(geo_result["regions"], min_confidence=min_conf)
            st.markdown(
                "<div class='section-title' style='margin-top:1rem'>📋 Change Classification</div>",
                unsafe_allow_html=True,
            )
            import pandas as pd

            df_changes = pd.DataFrame(changes)
            st.dataframe(df_changes, use_container_width=True, hide_index=True)

            # Metrics
            real_regions = [r for r in geo_result["regions"]
                           if r["category"] != "No Significant Change" and r["confidence"] >= min_conf]
            st.markdown(
                f"""
            <div class='metric-row' style='margin-top:1rem'>
                <div class='metric-card'>
                    <div class='metric-label'>SSIM Score</div>
                    <div class='metric-value'>{geo_result['ssim']:.2f}</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-label'>Change %</div>
                    <div class='metric-value'>{geo_result['change_pct']:.1f}%</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-label'>Total Regions</div>
                    <div class='metric-value'>{geo_result['region_count']}</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-label'>Real Changes</div>
                    <div class='metric-value' style='color:#ff1744'>{len(real_regions)}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Risk level based on real changes
            if len(real_regions) == 0:
                geo_risk = "LOW"
            elif geo_result["change_pct"] > 30 or len(real_regions) >= 4:
                geo_risk = "HIGH"
            elif geo_result["change_pct"] > 10 or len(real_regions) >= 2:
                geo_risk = "MEDIUM"
            else:
                geo_risk = "LOW"

            # Guard against duplicate saves on slider reruns
            geo_save_key = f"{old_img_file.name}_vs_{new_img_file.name}_{geo_result['ssim']}"
            if st.session_state.get("_geo_saved_key") != geo_save_key:
                save_history_entry(
                    source=f"{old_img_file.name} vs {new_img_file.name}",
                    anomaly_type="Geo Change",
                    score=int(geo_result["change_pct"]),
                    risk_level=geo_risk,
                    details={
                        "ssim": geo_result["ssim"],
                        "change_pct": geo_result["change_pct"],
                        "region_count": geo_result["region_count"],
                        "real_changes": len(real_regions),
                        "gemini_used": geo_result.get("gemini_used", False),
                    },
                )
                invalidate_history_cache()
                st.session_state["_geo_saved_key"] = geo_save_key

            # AI Explanation
            with st.expander("🤖 Generate AI Explanation (Gemini)", expanded=False):
                if st.button("✨ Get AI Explanation", key="geo_explain"):
                    with st.spinner("Connecting to Gemini..."):
                        mock_result = {
                            "anomaly_type": "Geographic Change Detection",
                            "contour_count": len(real_regions),
                            "anomaly_area_pct": geo_result["change_pct"],
                            "edge_density": 0.0,
                            "brightness_std": 0.0,
                        }
                        explanation = generate_explanation(mock_result, int(geo_result["change_pct"]), geo_risk)
                    st.markdown(
                        f"""
                    <div class='chat-ai'>
                        <div class='chat-label' style='color:#00e5ff'>SmartDetect · Gemini</div>
                        {explanation}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI CHAT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  AI Chat":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>🤖 AI Chat Assistant</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>💬 Powered by Gemini (Local LLM)</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0'>
        Ask questions about anomaly detection, risk levels, or get AI-powered explanations.
        Connects to your local Gemini instance running Gemini 2.5 Flash or Gemini 2.5 Flash.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick prompts
    st.markdown(
        "<div style='font-size:0.75rem; color:#6b8cad; font-family:Space Mono,monospace;"
        "letter-spacing:1px; text-transform:uppercase; margin-bottom:8px'>Quick Prompts</div>",
        unsafe_allow_html=True,
    )
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompts = {
        "q1": "What is anomaly detection?",
        "q2": "Explain risk scoring",
        "q3": "What is SSIM?",
        "q4": "Tips for geo change detection",
    }
    for (key, prompt), col in zip(quick_prompts.items(), [q_col1, q_col2, q_col3, q_col4]):
        with col:
            if st.button(prompt, key=key):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking..."):
                    reply = chat_with_assistant(prompt)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    # Chat history display
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown(
                """
            <div style='text-align:center; padding:3rem; color:#3a5270;
                        font-family:Space Mono,monospace; font-size:0.8rem'>
                ⬡ No messages yet. Ask something above or use a quick prompt.
            </div>
            """,
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                <div class='chat-user'>
                    <div class='chat-label' style='color:#7b2fff'>You</div>
                    {msg['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class='chat-ai'>
                    <div class='chat-label' style='color:#00e5ff'>SmartDetect · Gemini</div>
                    {msg['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Input
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    user_input = st.text_input(
        "Message",
        placeholder="Ask: 'What does high contour count mean?' or 'Explain SSIM score'...",
        label_visibility="collapsed",
        key="chat_input",
    )

    col_send, col_clear, _ = st.columns([1, 1, 4])
    with col_send:
        if st.button("📤 Send", key="send_chat"):
            if user_input.strip():
                st.session_state.chat_messages.append(
                    {"role": "user", "content": user_input}
                )
                with st.spinner("🤖 Gemini is thinking..."):
                    reply = chat_with_assistant(user_input)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": reply}
                )
                st.rerun()

    with col_clear:
        if st.button("🗑️ Clear", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋  History":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>📋 Detection History</div>",
        unsafe_allow_html=True,
    )

    history_data = cached_load_history()

    if not history_data:
        st.markdown(
            """
        <div style='text-align:center; padding:4rem; color:#3a5270;
                    font-family:Space Mono,monospace; font-size:0.85rem'>
            ⬡ No detection history yet.<br/>
            <span style='font-size:0.75rem'>Run an analysis to start logging results.</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        # Summary
        total = len(history_data)
        highrisk = sum(1 for h in history_data if h.get("risk_level") == "HIGH")
        mediumrisk = sum(1 for h in history_data if h.get("risk_level") == "MEDIUM")

        st.markdown(
            f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-label'>Total Entries</div>
                <div class='metric-value'>{total}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>High Risk</div>
                <div class='metric-value' style='color:#ff1744'>{highrisk}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Medium Risk</div>
                <div class='metric-value' style='color:#ffab00'>{mediumrisk}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Table
        import pandas as pd

        df = pd.DataFrame(history_data)[
            ["timestamp", "source", "anomaly_type", "score", "risk_level"]
        ]
        df.columns = ["Timestamp", "Source", "Anomaly Type", "Risk Score", "Risk Level"]
        df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-entry expandable details
        st.markdown(
            "<div class='section-title' style='margin-top:1rem'>🔍 Entry Details</div>",
            unsafe_allow_html=True,
        )
        for i, entry in enumerate(reversed(history_data[-10:])):
            risk = entry.get("risk_level", "LOW")
            color = {"LOW": "#00e676", "MEDIUM": "#ffab00", "HIGH": "#ff1744"}.get(
                risk, "#00e5ff"
            )
            with st.expander(
                f"[{entry.get('timestamp','?')[:16]}] {entry.get('source','Unknown')} — "
                f"Score: {entry.get('score','?')}",
                expanded=False,
            ):
                st.json(entry)

        # Export & clear
        col_exp, col_clr, _ = st.columns([1, 1, 4])
        with col_exp:
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(history_data, indent=2),
                file_name=f"anomaly_history_{datetime.now():%Y%m%d}.json",
                mime="application/json",
            )
        with col_clr:
            if st.button("🗑️ Clear History"):
                from utils import HISTORY_FILE
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                invalidate_history_cache()
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT PROJECT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️  About Project":
    st.markdown(
        "<div class='hero-title' style='margin-bottom:1.5rem'>ℹ️ About SmartDetect</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class='section-card'>
            <div class='section-title'>🚀 What is SmartDetect?</div>
            <p style='font-size:0.95rem; line-height:1.6; color:#e8f4fd; margin-top:10px;'>
                <b>SmartDetect</b> is a state-of-the-art AI-powered anomaly detection and computer vision analysis platform. 
                Built natively on Python and Streamlit, it leverages the cutting-edge <b>Google Gemini API</b> to perform 
                complex spatial, geographic, and real-time visual analyses with blazing speed.
            </p>
            <p style='font-size:0.95rem; line-height:1.6; color:#e8f4fd; margin-top:10px;'>
                Our mission is to provide an accessible, ultra-fast, and highly accurate tool for identifying structural defects, 
                geographic changes, and real-time video anomalies. By combining traditional OpenCV algorithms with advanced AI reasoning, 
                SmartDetect bridges the gap between raw pixel data and actionable intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='section-card'>
            <div class='section-title'>🛠️ Core Technologies</div>
            <ul style='font-size:0.95rem; line-height:1.8; color:#e8f4fd; margin-top:10px;'>
                <li><b>OpenCV</b>: High-performance computer vision for contour mapping, edge detection, and real-time video processing.</li>
                <li><b>Google Gemini 2.5 Flash</b>: Cloud-hosted LLM and VLM for intelligent structural analysis and deep explanatory text.</li>
                <li><b>Streamlit</b>: The robust Python framework powering this dynamic, responsive, and data-driven user interface.</li>
                <li><b>SSIM & CLAHE</b>: Advanced geographic image alignment, histogram matching, and structural similarity calculations.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='section-card'>
            <div class='section-title'>🎯 Key Features</div>
            <div style='display:flex; flex-direction:column; gap:12px; margin-top:10px;'>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='font-size:1.3rem; min-width:30px'>🖼️</div>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem'>Image & Structural Defect Analysis</div>
                        <div style='font-size:0.85rem; color:#6b8cad; margin-top:2px'>Upload photos of materials, roads, or infrastructure to detect cracks, dents, and contamination using AI bounding boxes.</div>
                    </div>
                </div>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='font-size:1.3rem; min-width:30px'>🌍</div>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem'>Geographic Change Detection</div>
                        <div style='font-size:0.85rem; color:#6b8cad; margin-top:2px'>Compare "Before" and "After" satellite imagery to automatically highlight and categorize new buildings, roads, or environmental shifts.</div>
                    </div>
                </div>
                <div style='display:flex; gap:12px; align-items:flex-start;'>
                    <div style='font-size:1.3rem; min-width:30px'>🔴</div>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem'>Live CCTV Simulation</div>
                        <div style='font-size:0.85rem; color:#6b8cad; margin-top:2px'>Connects directly to your webcam for real-time motion detection, human tracking, and frame extraction up to 60 FPS.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='section-card'>
            <div class='section-title'>👨‍💻 Built By</div>
            <div style='display:flex; flex-direction:column; gap:8px; margin-top:10px;'>
                <div style='font-size:0.95rem; color:#e8f4fd; display:flex; align-items:center; gap:8px;'>
                    <span style='color:#00e5ff'>⬡</span> Sugnik Tarafder
                </div>
                <div style='font-size:0.95rem; color:#e8f4fd; display:flex; align-items:center; gap:8px;'>
                    <span style='color:#00e5ff'>⬡</span> Arifur Rahman
                </div>
                <div style='font-size:0.95rem; color:#e8f4fd; display:flex; align-items:center; gap:8px;'>
                    <span style='color:#00e5ff'>⬡</span> Sk Shonju Ali
                </div>
                <div style='font-size:0.95rem; color:#e8f4fd; display:flex; align-items:center; gap:8px;'>
                    <span style='color:#00e5ff'>⬡</span> Trishan Nayek
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
