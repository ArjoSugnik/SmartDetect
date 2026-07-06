"""
utils.py — Shared utility functions
Covers: history I/O, report generation, image conversion, overlay drawing.
"""

import cv2
import json
import numpy as np
import os
import tempfile
from datetime import datetime
from PIL import Image

import streamlit as st


# ── History persistence ────────────────────────────────────────────────────────

def load_history() -> list:
    """Load detection history from Streamlit session state."""
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history


def save_history_entry(
    source: str,
    anomaly_type: str,
    score: int,
    risk_level: str,
    details: dict,
) -> None:
    """
    Append a new detection result to the session history.
    Strips non-serializable fields (raw contours) before saving.
    """
    history = load_history()

    # Remove OpenCV contour objects (not JSON-serializable)
    clean_details = {
        k: v
        for k, v in details.items()
        if k not in ("contours",) and _is_json_serializable(v)
    }

    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "anomaly_type": anomaly_type,
        "score": score,
        "risk_level": risk_level,
        "details": clean_details,
    }
    history.append(entry)
    st.session_state.history = history


def _is_json_serializable(val) -> bool:
    """Check if a value can be JSON-serialized."""
    try:
        json.dumps(val)
        return True
    except (TypeError, ValueError):
        return False


# ── Report generation ──────────────────────────────────────────────────────────

def generate_text_report(
    filename: str,
    result: dict,
    score: int,
    risk_level: str,
) -> str:
    """
    Generate a plain-text anomaly detection report.
    Returns the report as a string (downloadable via Streamlit).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60

    lines = [
        separator,
        "        AI ANOMALY DETECTION SYSTEM — REPORT",
        separator,
        f"  Generated     : {now}",
        f"  Source File   : {filename}",
        separator,
        "",
        "DETECTION SUMMARY",
        "-" * 40,
        f"  Anomaly Type       : {result.get('anomaly_type', 'N/A')}",
        f"  Risk Score         : {score}/100",
        f"  Risk Level         : {risk_level}",
    ]

    # Handle both numeric and string values from AI mode
    contour_val = result.get('contour_count', 0)
    area_val = result.get('anomaly_area_pct', 0)
    edge_val = result.get('edge_density', 0)
    bright_mean = result.get('brightness_mean', 0)
    bright_std = result.get('brightness_std', 0)

    if isinstance(contour_val, (int, float)):
        lines.append(f"  Contours Found     : {contour_val}")
    else:
        lines.append(f"  Contours Found     : {contour_val}")

    if isinstance(area_val, (int, float)):
        lines.append(f"  Anomaly Area       : {area_val:.2f}%")
    else:
        lines.append(f"  Anomaly Area       : {area_val}")

    if isinstance(edge_val, (int, float)):
        lines.append(f"  Edge Density       : {edge_val:.4f}")
    else:
        lines.append(f"  Edge Density       : {edge_val}")

    if isinstance(bright_mean, (int, float)):
        lines.append(f"  Brightness Mean    : {bright_mean:.1f}")
    else:
        lines.append(f"  Brightness Mean    : {bright_mean}")

    if isinstance(bright_std, (int, float)):
        lines.append(f"  Brightness Std Dev : {bright_std:.1f}")
    else:
        lines.append(f"  Brightness Std Dev : {bright_std}")

    # Add AI description and recommendation if present
    if result.get('description'):
        lines += ["", "AI FINDINGS", "-" * 40, f"  {result['description']}"]
    if result.get('recommendation'):
        lines += [f"  Recommendation: {result['recommendation']}"]

    lines += [
        "",
        "BOUNDING BOXES (TOP ANOMALOUS REGIONS)",
        "-" * 40,
    ]

    bboxes = result.get("bboxes", [])
    if bboxes:
        for i, bb in enumerate(bboxes, 1):
            lines.append(
                f"  [{i}] x={bb['x']} y={bb['y']} w={bb['w']} h={bb['h']}"
            )
    else:
        lines.append("  No significant regions detected.")

    lines += [
        "",
        "RISK INTERPRETATION",
        "-" * 40,
        _risk_interpretation(risk_level),
        "",
        separator,
        "  SmartDetect Detection System — For research purposes only.",
        separator,
    ]

    return "\n".join(lines)


def _risk_interpretation(risk_level: str) -> str:
    interp = {
        "LOW": (
            "  LOW RISK: Minor or no anomalies detected. The image appears\n"
            "  largely normal. No immediate action required."
        ),
        "MEDIUM": (
            "  MEDIUM RISK: Moderate anomalies detected. Some regions show\n"
            "  unusual patterns. Manual review recommended."
        ),
        "HIGH": (
            "  HIGH RISK: Significant anomalies detected. Multiple regions\n"
            "  show abnormal characteristics. Immediate review advised."
        ),
    }
    return interp.get(risk_level, "  Risk level undetermined.")


# ── Image conversion ───────────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a BGR OpenCV ndarray."""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV ndarray to a PIL RGB image."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Overlay drawing ────────────────────────────────────────────────────────────

def draw_anomaly_overlay(cv_img: np.ndarray, result: dict) -> np.ndarray:
    """
    Draw anomaly contours and bounding boxes on a copy of the image.
    - Green bounding boxes around anomalous regions
    - Cyan contour outlines
    - Red text labels
    """
    overlay = cv_img.copy()
    contours = result.get("contours", [])

    # Draw filled semi-transparent contour areas
    mask = np.zeros_like(overlay)
    cv2.drawContours(mask, contours, -1, (0, 255, 200), -1)
    overlay = cv2.addWeighted(overlay, 1.0, mask, 0.25, 0)

    # Draw contour outlines
    cv2.drawContours(overlay, contours, -1, (0, 229, 255), 2)

    # Draw bounding boxes with labels
    for i, bb in enumerate(result.get("bboxes", []), 1):
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 2)
        label = f"#{i} {result['anomaly_type'][:12]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y), (0, 255, 80), -1)
        cv2.putText(
            overlay, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 10, 0), 1, cv2.LINE_AA
        )

    # Watermark
    h_img, w_img = overlay.shape[:2]
    cv2.putText(
        overlay, "SmartDetect",
        (w_img - 100, h_img - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 229, 255), 1, cv2.LINE_AA,
    )

    return overlay


def draw_ai_visuals(cv_img: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Draw bounding boxes provided by AI (normalized 0-1000).
    """
    canvas = cv_img.copy()
    h, w = canvas.shape[:2]
    
    for i, bbox in enumerate(bboxes, 1):
        try:
            # Handle both list and dict formats if necessary
            if isinstance(bbox, list) and len(bbox) == 4:
                ymin, xmin, ymax, xmax = bbox
            elif isinstance(bbox, dict):
                ymin, xmin, ymax, xmax = bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']
            else:
                continue

            # Denormalize
            left = int(xmin * w / 1000)
            top = int(ymin * h / 1000)
            right = int(xmax * w / 1000)
            bottom = int(ymax * h / 1000)

            # Draw Neon Box
            cv2.rectangle(canvas, (left, top), (right, bottom), (0, 229, 255), 2)
            
            # Prevent label from going off-screen
            label_top = max(top, 25)
            cv2.rectangle(canvas, (left, label_top - 25), (left + 85, label_top), (0, 229, 255), -1)
            cv2.putText(
                canvas, f"DEFECT #{i}", (left + 5, label_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )
        except Exception:
            continue
            
    return canvas


def generate_ai_heatmap(cv_img: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Generate a 'better' hybrid heatmap using AI regions + CV edge refinement.
    This anchors AI detections to actual physical textures in the image.
    """
    h, w = cv_img.shape[:2]
    ai_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Pre-process edges to find physical structural breaks
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    edges_blur = cv2.GaussianBlur(edges, (15, 15), 0) / 255.0

    found_any = False
    for bbox in bboxes:
        try:
            if isinstance(bbox, list) and len(bbox) == 4:
                ymin, xmin, ymax, xmax = bbox
            elif isinstance(bbox, dict):
                ymin, xmin, ymax, xmax = bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']
            else:
                continue

            # Denormalize with safety clamping
            left = max(0, min(w-1, int(xmin * w / 1000)))
            top = max(0, min(h-1, int(ymin * h / 1000)))
            right = max(0, min(w-1, int(xmax * w / 1000)))
            bottom = max(0, min(h-1, int(ymax * h / 1000)))

            if right <= left or bottom <= top:
                continue

            # Apply "heat" to the bbox, amplified by local edges
            # This prevents heat from floating in empty areas (like sky)
            roi_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(roi_mask, (left, top), (right, bottom), 1.0, -1)
            
            # Combine AI intent with CV reality
            ai_mask += (roi_mask * (0.3 + 0.7 * edges_blur))
            found_any = True
        except Exception:
            continue

    # Final visual processing
    if found_any and np.max(ai_mask) > 0:
        # Normalize and Blur for glow
        ai_mask = cv2.GaussianBlur(ai_mask, (41, 41), 0)
        ai_mask = (ai_mask / np.max(ai_mask) * 255).astype(np.uint8)
        
        # Use an 'Inferno' style colormap for a more premium look
        heatmap = cv2.applyColorMap(ai_mask, cv2.COLORMAP_JET)
        
        # Create a cinematic dark background
        background = cv2.addWeighted(cv_img, 0.4, np.zeros_like(cv_img), 0.6, 0)
        
        # Add the heat
        result = cv2.addWeighted(background, 1.0, heatmap, 0.5, 0)
        return result
    else:
        # Fallback: if AI failed or hallucinated outside edges, show a faint edge map
        fallback = (edges_blur * 255).astype(np.uint8)
        # INFERNO maps 0 to black, avoiding blue tint on background
        fallback_heatmap = cv2.applyColorMap(fallback, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(cv_img, 0.8, fallback_heatmap, 0.4, 0)
