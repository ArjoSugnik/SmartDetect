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
    page_title="SmartDetect · AI Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
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
from groq_helper import generate_explanation, chat_with_assistant, analyze_image_structural, check_groq_status
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

# ── Custom CSS — Modern Glassmorphism + Animations ────────────────────────────
st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   SMARTDETECT · PREMIUM CYBER FRONTEND v3
   JetBrains Mono · True-black OLED animated orb BG · Deep glassmorphism
   Ambient-light gradient heading · Glitch-free uploader · Max animation
   ═══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600&display=swap');

:root {
    --bg-black:      #000000;
    --bg-near:       #050505;
    --glass-bg:      rgba(10, 10, 15, 0.45);
    --glass-bg-soft: rgba(12, 12, 20, 0.38);
    --glass-bg-strong: rgba(8, 8, 14, 0.62);
    --glass-border:  rgba(255, 255, 255, 0.07);
    --glass-border-hover: rgba(0, 234, 255, 0.45);
    --accent:        #00eaff;
    --accent-cyan:   #00e5ff;
    --accent2:       #a855ff;
    --accent-violet: #7b2fff;
    --accent-magenta:#ff2fd0;
    --accent-blue:   #2f6bff;
    --success:       #00ffa3;
    --warning:       #ffc400;
    --danger:        #ff2d6b;
    --text-primary:  #eef6ff;
    --text-muted:    #7a90b2;
    --text-dim:      #465a7a;
    --glass-blur:    24px;
    --font-mono:     'JetBrains Mono', ui-monospace, 'SF Mono', 'Courier New', monospace;
    --radius:        16px;
}

/* ═════════════════════════════════════════ KEYFRAMES ═══════════════════════════════════════ */
@keyframes fadeInUp   { from { opacity:0; transform:translateY(32px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn     { from { opacity:0; } to { opacity:1; } }
@keyframes slideInDown{ from { opacity:0; transform:translateY(-30px); } to { opacity:1; transform:translateY(0); } }
@keyframes slideInLeft{ from { opacity:0; transform:translateX(-28px); } to { opacity:1; transform:translateX(0); } }
@keyframes pulseGlow  { 0%,100% { box-shadow:0 0 10px rgba(0,234,255,0.18); } 50% { box-shadow:0 0 30px rgba(0,234,255,0.48); } }
@keyframes ambientPulse { 0%,100% { opacity:0.4; transform:scale(1); } 50% { opacity:0.95; transform:scale(1.18); } }
@keyframes shimmer    { 0% { background-position:-200% 0; } 100% { background-position:200% 0; } }
@keyframes gradientShift { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
@keyframes borderPulse{ 0%,100% { border-color:rgba(255,255,255,0.07); } 50% { border-color:rgba(0,234,255,0.34); } }
@keyframes neonFlicker{ 0%,92%,94%,97%,100% { opacity:1; } 93% { opacity:0.45; } 96% { opacity:0.72; } }
@keyframes dotBlink   { 0%,100% { box-shadow:0 0 6px var(--accent),0 0 12px var(--accent); opacity:1; } 50% { box-shadow:0 0 18px var(--accent),0 0 34px var(--accent); opacity:0.6; } }
@keyframes barSweep   { from { width:0; } }
@keyframes headingGlow{ 0%,100% { filter:drop-shadow(0 0 14px rgba(0,234,255,0.55)) drop-shadow(0 0 34px rgba(168,85,255,0.4)); } 50% { filter:drop-shadow(0 0 22px rgba(0,234,255,0.75)) drop-shadow(0 0 54px rgba(168,85,255,0.55)); } }
/* Ambient orb motion (huge, slow, blurred light orbs) */
@keyframes orbA { 0% { transform:translate(0,0) scale(1); } 33% { transform:translate(14vw,10vh) scale(1.18); } 66% { transform:translate(-8vw,16vh) scale(0.9); } 100% { transform:translate(0,0) scale(1); } }
@keyframes orbB { 0% { transform:translate(0,0) scale(1.1); } 33% { transform:translate(-16vw,-8vh) scale(0.95); } 66% { transform:translate(10vw,-14vh) scale(1.2); } 100% { transform:translate(0,0) scale(1.1); } }
@keyframes orbC { 0% { transform:translate(0,0) scale(1); } 50% { transform:translate(12vw,-12vh) scale(1.25); } 100% { transform:translate(0,0) scale(1); } }

/* ═══════════════════════ GLOBAL FONT ENFORCEMENT ══════════════════════ */
html, body, [class*="css"], [class*="st-"],
[data-testid="stAppViewContainer"] *,
input, textarea, select, button, code, kbd, pre,
h1, h2, h3, h4, h5, h6, p, span, div, label, a, td, th, li {
    font-family: var(--font-mono) !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
/* Restore Streamlit internal icons to prevent "arrow_downward" text overlap */
.stIconMaterial, span.material-icons, [data-testid="stIconMaterial"], span.icon, svg, .st-icon, .streamlit-expanderHeader svg {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
}
html, body { background: var(--bg-black) !important; color: var(--text-primary); }

/* ══════════════════════ TRUE-BLACK OLED ANIMATED ORB BACKGROUND ═══════════════════ */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 50% 0%, #050505 0%, #000000 70%) !important;
    position: relative; isolation: isolate; overflow-x: hidden;
}
/* Orb layer 1 — deep cyan */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; top: -25%; left: -20%;
    width: 70vw; height: 70vw; z-index: -3;
    background: radial-gradient(circle, rgba(0,234,255,0.22) 0%, rgba(0,234,255,0.08) 35%, transparent 66%);
    filter: blur(90px);
    animation: orbA 26s ease-in-out infinite;
    pointer-events: none;
}
/* Orb layer 2 — magenta */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; bottom: -30%; right: -20%;
    width: 66vw; height: 66vw; z-index: -3;
    background: radial-gradient(circle, rgba(255,47,208,0.18) 0%, rgba(168,85,255,0.09) 38%, transparent 66%);
    filter: blur(100px);
    animation: orbB 32s ease-in-out infinite;
    pointer-events: none;
}
/* Orb layer 3 — electric blue (attached to the scroll container) */
.stApp::before {
    content: '';
    position: fixed; top: 30%; left: 45%;
    width: 52vw; height: 52vw; z-index: -3;
    background: radial-gradient(circle, rgba(47,107,255,0.16) 0%, rgba(0,229,255,0.06) 40%, transparent 66%);
    filter: blur(110px);
    animation: orbC 38s ease-in-out infinite;
    pointer-events: none;
}
/* Subtle vignette so edges stay true-black */
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: -2; pointer-events: none;
    background: radial-gradient(ellipse 90% 80% at 50% 40%, transparent 55%, rgba(0,0,0,0.65) 100%);
}

/* ── Strip Streamlit chrome ── */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], section[data-testid="stSidebar"] {
    display:none !important; width:0 !important; min-width:0 !important; max-width:0 !important; visibility:hidden !important;
}
#MainMenu, footer { visibility:hidden !important; display:none !important; }
header[data-testid="stHeader"] { background:transparent !important; height:0 !important; display:none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container {
    padding-top: 1.1rem !important; padding-bottom: 2.6rem !important;
    max-width: 1320px !important; position: relative; z-index: 1;
}
.stMarkdown, .stMarkdown p { line-height: 1.6; letter-spacing: 0.2px; }

/* ══════════════════════════════════════ TOP NAVBAR ══════════════════════════════════════ */
.top-navbar {
    background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border); border-radius: var(--radius);
    padding: 0.7rem 1.5rem; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 1.2rem; position: relative; overflow: hidden;
    animation: slideInDown 0.7s cubic-bezier(0.16,1,0.3,1) both;
    box-shadow: 0 10px 44px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
}
.top-navbar::after {
    content:''; position:absolute; top:0; left:-60%; width:60%; height:100%;
    background: linear-gradient(90deg, transparent, rgba(0,234,255,0.12), transparent);
    animation: shimmer 6s linear infinite;
}
.navbar-brand {
    font-size: 1.25rem; font-weight: 800; color: var(--accent); letter-spacing: 1.5px;
    text-shadow: 0 0 24px rgba(0,234,255,0.6); display: flex; align-items: center; gap: 11px;
    white-space: nowrap; animation: neonFlicker 8s linear infinite;
}
.navbar-brand .brand-dot { width:10px; height:10px; background:var(--accent); border-radius:50%; animation: dotBlink 1.8s ease-in-out infinite; }
.navbar-divider { width:1px; height:30px; background:linear-gradient(180deg, transparent, rgba(0,234,255,0.4), transparent); }

.accent-line {
    height:2px; width:100%; margin:0.2rem 0 1.5rem 0; border-radius:2px;
    background: linear-gradient(90deg, transparent 0%, var(--accent) 20%, var(--accent-violet) 50%, var(--accent) 80%, transparent 100%);
    background-size:200% 100%; animation: shimmer 3.5s linear infinite, fadeIn 0.8s ease-out;
    box-shadow: 0 0 20px rgba(0,234,255,0.45);
}

/* Radio → neon nav tabs */
div[data-testid="stRadio"] > label { display:none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display:flex !important; gap:6px !important; flex-wrap:wrap !important; justify-content:center !important;
    animation: slideInDown 0.7s cubic-bezier(0.16,1,0.3,1) 0.1s both;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label {
    background: var(--glass-bg-soft) !important; border: 1px solid var(--glass-border) !important; border-radius: 12px !important;
    padding: 8px 17px !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.6px !important;
    color: var(--text-muted) !important; cursor: pointer !important;
    transition: all 0.28s cubic-bezier(0.34,1.56,0.64,1) !important; white-space: nowrap !important; margin: 0 !important;
    display: flex !important; align-items: center !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
    background: rgba(0,234,255,0.10) !important; border-color: var(--glass-border-hover) !important;
    color: var(--accent) !important; transform: translateY(-2px) scale(1.05) !important; box-shadow: 0 6px 22px rgba(0,234,255,0.3) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"],
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(0,234,255,0.2), rgba(168,85,255,0.16)) !important;
    border-color: rgba(0,234,255,0.6) !important; color: var(--accent) !important;
    box-shadow: 0 0 22px rgba(0,234,255,0.34), inset 0 0 12px rgba(0,234,255,0.1) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label span[data-testid="stMarkdownContainer"] p { font-size:0.72rem !important; font-weight:600 !important; margin:0 !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child { display:none !important; }

/* ══════════════════════════════════════ HERO + AMBIENT-LIGHT HEADING ═══════════════════════════ */
.hero-banner {
    background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border); border-radius: 24px;
    padding: 3.2rem 2.8rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; text-align: center;
    animation: fadeInUp 0.7s cubic-bezier(0.16,1,0.3,1) both;
    box-shadow: 0 28px 80px rgba(0,0,0,0.55);
}
/* Ambient light pool behind the heading */
.hero-banner::before {
    content:''; position:absolute; top:8%; left:50%; transform:translateX(-50%);
    width: 70%; height: 70%;
    background: radial-gradient(ellipse at center, rgba(0,234,255,0.18) 0%, rgba(168,85,255,0.12) 40%, transparent 70%);
    filter: blur(50px); animation: ambientPulse 6s ease-in-out infinite; pointer-events:none; z-index:0;
}
.hero-badge {
    display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:2px;
    color:var(--accent); background:rgba(0,234,255,0.08); border:1px solid rgba(0,234,255,0.3);
    border-radius:6px; padding:4px 11px; margin:0 4px; position:relative; z-index:1; animation: fadeIn 1s ease-out both;
}
.hero-title {
    font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; line-height: 1.06;
    margin: 1rem auto 0 auto; position: relative; z-index: 1; max-width: 16ch;
    background: linear-gradient(96deg, #00e5ff 0%, #35c4ff 30%, #a855ff 68%, #7b2fff 100%);
    background-size: 220% auto; -webkit-background-clip: text; background-clip: text;
    color: transparent; -webkit-text-fill-color: transparent;
    /* Multi-layered ambient glow radiating from behind the letters (drop-shadow works with clipped text) */
    filter:
        drop-shadow(0 0 6px rgba(0,234,255,0.6))
        drop-shadow(0 0 18px rgba(0,234,255,0.45))
        drop-shadow(0 0 40px rgba(168,85,255,0.4))
        drop-shadow(0 0 70px rgba(123,47,255,0.28));
    animation: gradientShift 7s ease infinite, headingGlow 5s ease-in-out infinite;
}
.hero-sub {
    font-size: 0.95rem; color: var(--text-muted); margin: 1rem auto 0 auto;
    letter-spacing: 1.2px; position: relative; z-index: 1; font-weight: 400; max-width: 60ch;
}

/* ══════════════════════════════════════ PAGE TITLE ══════════════════════════════════════ */
.page-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; color: var(--text-primary);
    margin: 0.3rem 0 1.5rem 0; padding-left: 16px; position: relative; display: flex; align-items: center; gap: 4px;
    animation: slideInLeft 0.6s cubic-bezier(0.16,1,0.3,1) both; text-shadow: 0 0 30px rgba(0,234,255,0.22);
}
.page-title::before {
    content:''; position:absolute; left:0; top:12%; bottom:12%; width:5px; border-radius:5px;
    background: linear-gradient(180deg, var(--accent), var(--accent-violet)); box-shadow: 0 0 16px rgba(0,234,255,0.6);
}

/* ══════════════════════════════════════ STATUS BAR ═════════════����════════════════════════ */
.status-bar {
    display:flex; flex-wrap:wrap; gap:1.6rem; align-items:center;
    background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border); border-radius: 14px; padding: 0.9rem 1.5rem; margin-bottom: 1.5rem;
    animation: fadeInUp 0.6s ease-out 0.1s both;
}
.status-item { display:flex; align-items:center; gap:8px; font-size:0.78rem; color:var(--text-muted); }
.status-value { font-weight:700; }

/* ══════════════════════════ METRIC CARDS — staggered load, pixel-aligned ═══════════════════ */
.metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.6rem; align-items:stretch; }
.metric-card {
    background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border); border-radius: var(--radius); padding: 1.5rem 1.6rem;
    position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: center; min-height: 120px;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s ease, border-color 0.3s ease;
    opacity: 0; animation: fadeInUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg, var(--accent), var(--accent-violet)); opacity:0.8; }
.metric-row .metric-card:nth-child(1) { animation-delay: 0.10s; }
.metric-row .metric-card:nth-child(2) { animation-delay: 0.22s; }
.metric-row .metric-card:nth-child(3) { animation-delay: 0.34s; }
.metric-row .metric-card:nth-child(4) { animation-delay: 0.46s; }
.metric-card:hover {
    transform: translateY(-6px) scale(1.05); border-color: var(--glass-border-hover);
    box-shadow: 0 18px 50px rgba(0,0,0,0.6), 0 0 28px rgba(0,234,255,0.26);
}
.metric-label { font-size:0.66rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:var(--text-muted); }
.metric-value { font-size:2.7rem; font-weight:800; color:var(--accent); margin-top:0.5rem; text-shadow:0 0 28px rgba(0,234,255,0.42); line-height:1; font-variant-numeric:tabular-nums; letter-spacing:1px; }

/* ══════════════════════════════════════ SECTION CARDS ═══════════════════════════════════ */
.section-card {
    background: var(--glass-bg); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border); border-radius: 18px; padding: 1.6rem 1.7rem; margin-bottom: 1.4rem;
    position: relative; overflow: hidden; animation: fadeInUp 0.6s cubic-bezier(0.16,1,0.3,1) both;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.section-card:hover { border-color: var(--glass-border-hover); box-shadow: 0 14px 46px rgba(0,0,0,0.5), 0 0 24px rgba(0,234,255,0.14); }
.section-title { font-size:0.98rem; font-weight:700; letter-spacing:0.8px; color:var(--accent); margin-bottom:1.1rem; text-shadow:0 0 14px rgba(0,234,255,0.3); display:flex; align-items:center; gap:8px; }

.feature-item { display:flex; align-items:flex-start; gap:14px; padding:12px 13px; border-radius:12px; margin-bottom:6px; border:1px solid transparent; transition: all 0.26s cubic-bezier(0.34,1.56,0.64,1); }
.feature-item:hover { background:rgba(0,234,255,0.05); border-color:rgba(0,234,255,0.24); transform:translateX(6px); }
.step-item { display:flex; align-items:center; gap:14px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05); transition: all 0.24s ease; }
.step-item:hover { transform:translateX(5px); }
.step-num { font-size:0.8rem; font-weight:800; color:#000; background:linear-gradient(135deg, var(--accent), var(--accent-violet)); min-width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:9px; box-shadow:0 0 16px rgba(0,234,255,0.35); flex-shrink:0; }
.step-text { font-size:0.84rem; color:var(--text-primary); letter-spacing:0.2px; }

/* ══════════════════════════ RISK BADGES + SCORE BAR ═════════════════════════ */
.risk-badge { display:inline-block; font-size:0.72rem; font-weight:800; letter-spacing:2px; padding:5px 17px; border-radius:8px; text-transform:uppercase; }
.risk-low    { color:var(--success); background:rgba(0,255,163,0.10); border:1px solid rgba(0,255,163,0.42); box-shadow:0 0 16px rgba(0,255,163,0.2); animation: pulseGlow 3s ease-in-out infinite; }
.risk-medium { color:var(--warning); background:rgba(255,196,0,0.10); border:1px solid rgba(255,196,0,0.42); box-shadow:0 0 16px rgba(255,196,0,0.22); animation: pulseGlow 2.2s ease-in-out infinite; }
.risk-high   { color:var(--danger); background:rgba(255,45,107,0.12); border:1px solid rgba(255,45,107,0.5); box-shadow:0 0 22px rgba(255,45,107,0.32); animation: pulseGlow 1.3s ease-in-out infinite; }
.score-bar-wrap { width:100%; height:13px; background:rgba(0,0,0,0.5); border:1px solid rgba(0,234,255,0.14); border-radius:8px; overflow:hidden; position:relative; }
.score-bar-fill { height:100%; border-radius:8px; position:relative; box-shadow:0 0 18px rgba(0,234,255,0.5); animation: barSweep 1.1s cubic-bezier(0.16,1,0.3,1) both; }
.score-bar-fill::after { content:''; position:absolute; inset:0; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent); background-size:200% 100%; animation: shimmer 2s linear infinite; }

/* ══════════════════════════════════════ CHAT BUBBLES ══════════════════════════════════ */
.chat-label { font-size:0.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--text-dim); margin-bottom:4px; }
.chat-user, .chat-ai { border-radius:14px; padding:13px 17px; margin-bottom:12px; font-size:0.86rem; line-height:1.6; animation: fadeInUp 0.4s ease-out both; max-width:90%; letter-spacing:0.2px; }
.chat-user { background:linear-gradient(135deg, rgba(0,234,255,0.13), rgba(0,184,212,0.06)); border:1px solid rgba(0,234,255,0.3); color:var(--text-primary); margin-left:auto; }
.chat-ai { background:linear-gradient(135deg, rgba(168,85,255,0.13), rgba(123,47,255,0.05)); border:1px solid rgba(168,85,255,0.3); color:var(--text-primary); }
.empty-state { text-align:center; padding:3rem 1.5rem; color:var(--text-muted); font-size:0.9rem; letter-spacing:0.5px; border:1px dashed rgba(0,234,255,0.18); border-radius:16px; animation: fadeIn 0.6s ease-out, borderPulse 3s ease-in-out infinite; }

/* ══════════════════════════════════════ BUTTONS ═══════════════════════════════════════ */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    font-weight:700 !important; font-size:0.82rem !important; letter-spacing:0.6px !important;
    color:var(--accent) !important; background:rgba(0,234,255,0.07) !important;
    border:1px solid rgba(0,234,255,0.32) !important; border-radius:12px !important; padding:0.58rem 1.3rem !important; width:100%;
    transition: all 0.28s cubic-bezier(0.34,1.56,0.64,1) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    color:#000 !important; background:linear-gradient(135deg, var(--accent), var(--accent-violet)) !important;
    border-color:var(--accent) !important; transform:translateY(-2px) scale(1.05) !important;
    box-shadow:0 10px 32px rgba(0,234,255,0.45), 0 0 24px rgba(168,85,255,0.34) !important;
}
.stButton > button:active, .stDownloadButton > button:active { transform:translateY(0) scale(0.99) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER — DEEP DOM HACK (font-size:0 hide + single ::after label)
   ZERO ghosting / overlap / duplicate text.
   ═══════════════════════════════════════════════════════════════════════════ */
/* Dropzone = glass panel, centered column */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass-bg) !important; border: 1.5px dashed rgba(0,234,255,0.32) !important;
    border-radius: 16px !important; padding: 1.6rem 1.5rem !important;
    display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important;
    gap: 0.9rem !important; text-align: center !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    backdrop-filter: blur(var(--glass-blur)) !important; -webkit-backdrop-filter: blur(var(--glass-blur)) !important;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important; background: rgba(0,234,255,0.06) !important;
    transform: scale(1.01) !important; box-shadow: 0 0 32px rgba(0,234,255,0.26) !important;
}
/* Hide ALL native nested span text (kills the double/ghost "Drag and drop" text) */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
    font-size: 0 !important;
    color: transparent !important;
    text-shadow: none !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: 0 !important;
    line-height: 0 !important;
}
/* Neutralize any icon/margins so the injected label sits perfectly centered */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important;
    width: 100% !important; margin: 0 !important; padding: 0 !important; gap: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none !important; }
/* Inject a single crisp label */
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: 'Upload File';
    display: block !important;
    font-family: var(--font-mono) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    line-height: 1.2 !important;
    color: #00e5ff !important;
    -webkit-text-fill-color: #00e5ff !important;
    text-shadow: 0 0 18px rgba(0,229,255,0.5) !important;
    text-align: center !important;
}
/* Secondary hint under the label */
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: 'Drag & drop · or browse';
    display: block !important; order: 2;
    font-family: var(--font-mono) !important; font-size: 0.68rem !important; font-weight: 400 !important;
    letter-spacing: 0.6px !important; line-height: 1.4 !important; margin-top: 6px !important;
    color: var(--text-muted) !important; -webkit-text-fill-color: var(--text-muted) !important; text-align: center !important;
}
/* Browse button: hide native label, inject single clean word */
[data-testid="stFileUploader"] button {
    position: relative !important; font-size: 0 !important; color: transparent !important;
    background: rgba(0,234,255,0.08) !important; border: 1px solid rgba(0,234,255,0.34) !important;
    border-radius: 10px !important; padding: 0.5rem 1.4rem !important; min-height: 38px;
    display: inline-flex !important; align-items: center !important; justify-content: center !important;
    transition: all 0.26s cubic-bezier(0.34,1.56,0.64,1) !important; width: auto !important;
}
[data-testid="stFileUploader"] button * { font-size: 0 !important; color: transparent !important; -webkit-text-fill-color: transparent !important; }
[data-testid="stFileUploader"] button::after {
    content: 'Browse'; font-family: var(--font-mono) !important; font-size: 0.8rem !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; color: var(--accent) !important; -webkit-text-fill-color: var(--accent) !important;
}
[data-testid="stFileUploader"] button:hover { background: linear-gradient(135deg, var(--accent), var(--accent-violet)) !important; border-color: var(--accent) !important; transform: scale(1.05) !important; box-shadow: 0 0 22px rgba(0,234,255,0.45) !important; }
[data-testid="stFileUploader"] button:hover::after { color: #000 !important; -webkit-text-fill-color: #000 !important; }
/* Uploaded-file chip stays readable */
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * { font-size: 0.78rem !important; color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important; font-family: var(--font-mono) !important; }

/* Camera input */
[data-testid="stCameraInput"] button { font-weight:700 !important; color:var(--accent) !important; background:rgba(0,234,255,0.08) !important; border:1px solid rgba(0,234,255,0.3) !important; border-radius:12px !important; transition: all 0.26s cubic-bezier(0.34,1.56,0.64,1) !important; }
[data-testid="stCameraInput"] button:hover { transform:scale(1.05) !important; box-shadow:0 0 20px rgba(0,234,255,0.35) !important; }

/* ══════════════════════════════════════ INPUTS + SELECT ══════════════════════════════════ */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div, [data-baseweb="select"] {
    background: var(--glass-bg) !important; color: var(--text-primary) !important; border-radius: 11px !important;
    border: 1px solid rgba(0,234,255,0.18) !important; transition: all 0.26s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(0,234,255,0.18), 0 0 18px rgba(0,234,255,0.2) !important; }
.stSelectbox div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
label, .stSelectbox label, .stFileUploader > label, .stTextInput label, .stSelectbox > label { color: var(--text-muted) !important; font-weight: 600 !important; font-size: 0.8rem !important; letter-spacing: 0.5px !important; }

/* ════════════════════════════ TABS ═══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] { gap:6px !important; background:transparent !important; border-bottom:1px solid rgba(0,234,255,0.1) !important; }
.stTabs [data-baseweb="tab"] { font-weight:600 !important; font-size:0.8rem !important; color:var(--text-muted) !important; background:var(--glass-bg-soft) !important; border:1px solid var(--glass-border) !important; border-radius:10px 10px 0 0 !important; padding:9px 19px !important; transition: all 0.26s ease !important; }
.stTabs [data-baseweb="tab"]:hover { color:var(--accent) !important; background:rgba(0,234,255,0.06) !important; transform:translateY(-2px) !important; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; background:linear-gradient(135deg, rgba(0,234,255,0.16), rgba(168,85,255,0.1)) !important; border-color:rgba(0,234,255,0.4) !important; box-shadow:0 0 16px rgba(0,234,255,0.2) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent) !important; }

/* ════════════════════════════ IMAGES ═══���������═══════════════════════════ */
[data-testid="stImage"] img { border-radius:14px !important; border:1px solid rgba(0,234,255,0.16) !important; transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important; }
[data-testid="stImage"] img:hover { border-color:rgba(0,234,255,0.4) !important; box-shadow:0 0 28px rgba(0,234,255,0.22) !important; transform:scale(1.01) !important; }
[data-testid="stImage"] figcaption, [data-testid="caption"] { font-family:var(--font-mono) !important; color:var(--text-muted) !important; text-align:center !important; }

/* ═══════════════════════════ TABLES / DATAFRAMES — pixel-aligned ═══════════════════════ */
[data-testid="stTable"] table, .stDataFrame, [data-testid="stDataFrame"] { font-family:var(--font-mono) !important; background:var(--glass-bg) !important; border-radius:12px !important; overflow:hidden !important; border:1px solid var(--glass-border) !important; }
[data-testid="stTable"] th, [data-testid="stTable"] td { font-family:var(--font-mono) !important; border-color:rgba(255,255,255,0.06) !important; padding:10px 14px !important; font-variant-numeric:tabular-nums !important; }
[data-testid="stTable"] th { color:var(--accent) !important; background:rgba(0,234,255,0.06) !important; text-transform:uppercase !important; letter-spacing:1px !important; font-size:0.72rem !important; text-align:left !important; }
[data-testid="stTable"] td { color:var(--text-primary) !important; font-size:0.82rem !important; }
[data-testid="stTable"] td:not(:first-child), [data-testid="stTable"] th:not(:first-child) { text-align:right !important; }

/* Alerts / spinner / progress / native metric */
[data-testid="stAlert"] { font-family:var(--font-mono) !important; border-radius:12px !important; backdrop-filter:blur(10px) !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
[data-testid="stSpinner"] p { font-family:var(--font-mono) !important; color:var(--accent) !important; }
.stProgress > div > div > div { background:linear-gradient(90deg, var(--accent), var(--accent-violet)) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family:var(--font-mono) !important; font-variant-numeric:tabular-nums !important; }

/* Scrollbar */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:#000; }
::-webkit-scrollbar-thumb { background:linear-gradient(180deg, var(--accent), var(--accent-violet)); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }
code, pre { font-family:var(--font-mono) !important; }

/* Stagger utilities */
.stagger-1 { animation: fadeInUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.05s both; }
.stagger-2 { animation: fadeInUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.15s both; }
.stagger-3 { animation: fadeInUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.25s both; }
.stagger-4 { animation: fadeInUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.35s both; }
.stagger-5 { animation: fadeInUp 0.55s cubic-bezier(0.16,1,0.3,1) 0.45s both; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:0.001ms !important; animation-iteration-count:1 !important; transition-duration:0.001ms !important; } }
@media (max-width: 820px) {
    .metric-row { grid-template-columns:repeat(2,1fr); }
    .hero-title { font-size:2.1rem; }
    .page-title { font-size:1.5rem; }
}
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
    st.session_state.total_scans = len(load_history())

def cached_load_history() -> list:
    """Load history from session_state."""
    return load_history()

def invalidate_history_cache():
    """No-op for ephemeral history."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────
# Build a custom HTML header with brand, then use st.radio horizontal for nav
st.markdown(
    """
<div class='top-navbar'>
    <div class='navbar-brand'>
        <div class='brand-dot'></div>
        SmartDetect
    </div>
    <div class='navbar-divider'></div>
</div>
""",
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🖼️ Image",
        "📷 Camera",
        "🎬 Video",
        "🌍 Geo Change",
        "🤖 AI Chat",
        "📋 History",
        "ℹ️ About",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)


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
if page == "🏠 Dashboard":
    st.markdown(
        """
    <div class='hero-banner'>
        <div>
            <span class='hero-badge'>AI-POWERED</span>
            <span class='hero-badge'>REAL-TIME</span>
            <span class='hero-badge'>GROQ AI</span>
        </div>
        <div class='hero-title' style='margin-top:0.8rem'>
            Anomaly Detection and Correction
        </div>
        <div class='hero-sub'>
            Computer vision · Geo change detection · LLM-powered explanations · Risk scoring
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # System status bar (moved from sidebar)
    history_data = cached_load_history()
    total = len(history_data)
    highrisk = sum(1 for h in history_data if h.get("risk_level") == "HIGH")
    mediumrisk = sum(1 for h in history_data if h.get("risk_level") == "MEDIUM")
    lowrisk = sum(1 for h in history_data if h.get("risk_level") == "LOW")

    groq_health = check_groq_status()
    if groq_health["online"]:
        groq_dot = "<span style='color:#00e676'>●</span>"
        groq_text = "<span class='status-value' style='color:#00e676'>ONLINE</span>"
    else:
        groq_dot = "<span style='color:#ff1744'>●</span>"
        groq_text = "<span class='status-value' style='color:#ff1744'>OFFLINE</span>"

    st.markdown(
        f"""
    <div class='status-bar'>
        <div class='status-item'>
            <span>📊</span>
            <span>Scans:</span>
            <span class='status-value' style='color:var(--accent)'>{total}</span>
        </div>
        <div class='status-item'>
            <span>🔴</span>
            <span>High Risk:</span>
            <span class='status-value' style='color:#ff1744'>{highrisk}</span>
        </div>
        <div class='status-item'>
            <span>🟡</span>
            <span>Medium:</span>
            <span class='status-value' style='color:#ffab00'>{mediumrisk}</span>
        </div>
        <div class='status-item'>
            <span>🟢</span>
            <span>Low:</span>
            <span class='status-value' style='color:#00e676'>{lowrisk}</span>
        </div>
        <div class='status-item'>
            {groq_dot}
            <span>Groq:</span>
            {groq_text}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metric cards
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
            ("🤖", "Groq Vision AI", "Ultra-fast Groq inference for structural analysis"),
            ("📊", "Risk Score System", "0–100 scoring with LOW/MED/HIGH"),
            ("📋", "Detection History", "Persistent JSON log with replay"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
            <div class='feature-item'>
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
            ("01", "Select any module from the top navigation bar"),
            ("02", "Upload image/video or start camera"),
            ("03", "View detection results & risk score"),
            ("04", "Get AI explanation from Groq"),
            ("05", "Download report or check history"),
        ]
        for num, text in steps:
            st.markdown(
                f"""
            <div class='step-item'>
                <div class='step-num'>{num}</div>
                <div class='step-text'>{text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div style='margin-top:1rem; padding:12px; background:rgba(123,47,255,0.06);
                    backdrop-filter:blur(8px); border:1px solid rgba(123,47,255,0.18);
                    border-radius:10px'>
            <div style='font-size:0.72rem; font-family:Space Mono,monospace;
                        color:#7b2fff; letter-spacing:1px; margin-bottom:6px'>GROQ SETUP</div>
            <code style='font-size:0.72rem; color:#a8c4e0; display:block; line-height:1.8'>
            $ pip install groq<br/>
            $ export GROQ_API_KEY=your_key
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
                            padding:7px 0; border-bottom:1px solid rgba(0,229,255,0.04);
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
# ────────────────────────────────────────────��────────────────────────────────
elif page == "🖼️ Image":
    st.markdown(
        "<div class='page-title'>🖼️ Image Anomaly Detection</div>",
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
                    # Fallback to OpenCV if Groq is not running OR failed to provide boxes
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
                            "recommendation": "Manual review required. Start Groq for AI analysis.",
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
        with st.expander("🤖 Generate AI Explanation (Groq)", expanded=False):
            if st.button("✨ Get AI Explanation", key="img_explain"):
                with st.spinner("Connecting to Groq..."):
                    explanation = generate_explanation(result, score, risk_level, cv_img)
                st.markdown(
                    f"""
                <div class='chat-ai'>
                    <div class='chat-label' style='color:#00e5ff'>SmartDetect · Groq</div>
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
elif page == "📷 Camera":
    st.markdown(
        "<div class='page-title'>📷 Real-Time Camera Detection</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>ℹ️ How It Works</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0; line-height:1.6'>
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
                            "recommendation": "Manual review required. Start Groq for AI analysis.",
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

        # AI Explanation
        with st.expander("🤖 Generate AI Explanation (Groq)", expanded=False):
            if st.button("✨ Get AI Explanation", key="cam_explain"):
                with st.spinner("Connecting to Groq..."):
                    explanation = generate_explanation(result, score, risk_level, cv_img)
                st.markdown(
                    f"""
                <div class='chat-ai'>
                    <div class='chat-label' style='color:#00e5ff'>SmartDetect · Groq</div>
                    {explanation}
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
elif page == "🎬 Video":
    st.markdown(
        "<div class='page-title'>🎬 Video Anomaly Analysis</div>",
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

            analyze_clicked = st.button("▶️ Analyze Video")
            vid_cache_key = f"{video_file.name}_{video_file.size}_{max_frames}_{fps_rate}"

            if analyze_clicked:
                with st.spinner("🎬 Extracting and analyzing frames..."):
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix="." + video_file.name.split(".")[-1]
                    ) as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    frame_results = process_video_frames(tmp_path, max_frames=max_frames, fps_rate=fps_rate)
                    os.unlink(tmp_path)
                    
                    st.session_state._vid_cache_key = vid_cache_key
                    st.session_state._vid_results = frame_results

            if st.session_state.get("_vid_cache_key") == vid_cache_key:
                frame_results = st.session_state._vid_results
                
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

                    # AI Explanation for Video
                    with st.expander("🤖 Generate AI Summary (Groq)", expanded=False):
                        if st.button("✨ Get AI Explanation", key="vid_explain"):
                            with st.spinner("Connecting to Groq..."):
                                mock_result = {
                                    "anomaly_type": "Video Analysis Peak",
                                    "contour_count": "Multiple frames",
                                    "anomaly_area_pct": max_score,
                                    "edge_density": 0.0,
                                    "brightness_std": 0.0,
                                }
                                explanation = generate_explanation(mock_result, int(max_score), overall_risk)
                            st.markdown(
                                f"""
                            <div class='chat-ai'>
                                <div class='chat-label' style='color:#00e5ff'>SmartDetect · Groq</div>
                                {explanation}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

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
            <p style='font-size:0.85rem; color:#a8c4e0; margin:0; line-height:1.6'>
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
                                    backdrop-filter:blur(12px);
                                    border:1px solid var(--glass-border); border-radius:14px'>
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




# ─────��───────────────────────────────────────────────────────────────────────
# PAGE: GEO CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍 Geo Change":
    st.markdown(
        "<div class='page-title'>🌍 Geo Change Detection</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>ℹ️ Satellite / Aerial Image Comparison</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0; line-height:1.7'>
        Upload two images of the same geographic region taken at different times.
        The system aligns images, generates a change heatmap, extracts candidate regions
        via OpenCV, and validates each region using <b style='color:#00e5ff'>Groq vision AI</b>
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
            source_label = "🤖 Groq Vision AI" if geo_result.get("groq_used") else "🔬 CV Analysis (Offline)"
            source_color = "#00e5ff" if geo_result.get("groq_used") else "#ffab00"
            st.markdown(
                f"""
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:1rem; flex-wrap:wrap'>
                <div style='color:#00e676; font-weight:600; font-size:0.9rem'>
                    ✅ Images matched (SSIM: {geo_result['ssim']:.3f})
                </div>
                <div style='background:rgba(0,229,255,0.06); backdrop-filter:blur(8px);
                            border:1px solid {source_color}40;
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
                        "groq_used": geo_result.get("groq_used", False),
                    },
                )
                invalidate_history_cache()
                st.session_state["_geo_saved_key"] = geo_save_key

            # AI Explanation
            with st.expander("🤖 Generate AI Explanation (Groq)", expanded=False):
                if st.button("✨ Get AI Explanation", key="geo_explain"):
                    with st.spinner("Connecting to Groq..."):
                        mock_result = {
                            "anomaly_type": "Geographic Change Detection",
                            "contour_count": len(real_regions),
                            "anomaly_area_pct": geo_result["change_pct"],
                            "edge_density": 0.0,
                            "brightness_std": 0.0,
                        }
                        explanation = generate_explanation(mock_result, int(geo_result["change_pct"]), geo_risk, geo_result["new_resized"])
                    st.markdown(
                        f"""
                    <div class='chat-ai'>
                        <div class='chat-label' style='color:#00e5ff'>SmartDetect · Groq</div>
                        {explanation}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI CHAT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 AI Chat":
    st.markdown(
        "<div class='page-title'>🤖 AI Chat Assistant</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='section-card'>
        <div class='section-title'>💬 Powered by Groq (Ultra-Fast AI)</div>
        <p style='font-size:0.85rem; color:#a8c4e0; margin:0; line-height:1.6'>
        Ask questions about anomaly detection, risk levels, or get AI-powered explanations.
        Connects to Groq's ultra-fast inference engine running Llama 3.3 70B Versatile.
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
            <div class='empty-state'>
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
                    <div class='chat-label' style='color:#00e5ff'>SmartDetect · Groq</div>
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
                with st.spinner("🤖 Groq is thinking..."):
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
elif page == "📋 History":
    st.markdown(
        "<div class='page-title'>📋 Detection History</div>",
        unsafe_allow_html=True,
    )

    history_data = cached_load_history()

    if not history_data:
        st.markdown(
            """
        <div class='empty-state'>
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
elif page == "ℹ️ About":
    st.markdown(
        "<div class='page-title'>ℹ️ About SmartDetect</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class='section-card stagger-1'>
            <div class='section-title'>🚀 What is SmartDetect?</div>
            <p style='font-size:0.95rem; line-height:1.6; color:#e8f4fd; margin-top:10px;'>
                <b>SmartDetect</b> is a state-of-the-art AI-powered anomaly detection and computer vision analysis platform. 
                Built natively on Python and Streamlit, it leverages the cutting-edge <b>Groq API</b> to perform 
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
        <div class='section-card stagger-2'>
            <div class='section-title'>🛠️ Core Technologies</div>
            <ul style='font-size:0.95rem; line-height:1.8; color:#e8f4fd; margin-top:10px;'>
                <li><b>OpenCV</b>: High-performance computer vision for contour mapping, edge detection, and real-time video processing.</li>
                <li><b>Groq (Llama 3.2 Vision + Llama 3.3 70B)</b>: Ultra-fast AI inference for intelligent structural analysis and deep explanatory text.</li>
                <li><b>Streamlit</b>: The robust Python framework powering this dynamic, responsive, and data-driven user interface.</li>
                <li><b>SSIM & CLAHE</b>: Advanced geographic image alignment, histogram matching, and structural similarity calculations.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='section-card stagger-3'>
            <div class='section-title'>🎯 Key Features</div>
            <div style='display:flex; flex-direction:column; gap:12px; margin-top:10px;'>
                <div class='feature-item'>
                    <div style='font-size:1.3rem; min-width:30px'>🖼️</div>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem'>Image & Structural Defect Analysis</div>
                        <div style='font-size:0.85rem; color:#6b8cad; margin-top:2px'>Upload photos of materials, roads, or infrastructure to detect cracks, dents, and contamination using AI bounding boxes.</div>
                    </div>
                </div>
                <div class='feature-item'>
                    <div style='font-size:1.3rem; min-width:30px'>🌍</div>
                    <div>
                        <div style='font-weight:600; font-size:0.95rem'>Geographic Change Detection</div>
                        <div style='font-size:0.85rem; color:#6b8cad; margin-top:2px'>Compare "Before" and "After" satellite imagery to automatically highlight and categorize new buildings, roads, or environmental shifts.</div>
                    </div>
                </div>
                <div class='feature-item'>
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
        <div class='section-card stagger-4'>
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
