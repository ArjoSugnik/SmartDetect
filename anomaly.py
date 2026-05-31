"""
anomaly.py — Core anomaly detection logic
Uses OpenCV contour analysis + thresholding to detect anomalies in images.
"""

import cv2
import numpy as np
from typing import Tuple


# ── Anomaly type labels ────────────────────────────────────────────────────────
ANOMALY_TYPES = [
    "Structural Change",
    "Object Detected",
    "Texture Anomaly",
    "Lighting Irregularity",
    "Edge Distortion",
    "Pattern Break",
    "Unknown Intrusion",
]

# Specific structural defects for AI-assisted detection
STRUCTURAL_ANOMALY_TYPES = [
    "Surface Scratches",
    "Dents and Bumps",
    "Cracks",
    "Contamination/Soiling",
    "Holes/Punctures",
    "Fraying/Broken Edges",
]


def detect_image_anomaly(cv_img: np.ndarray) -> dict:
    """
    Detect anomalies in a BGR OpenCV image.

    Pipeline:
      1. Convert to grayscale
      2. Apply Gaussian blur to reduce noise
      3. Adaptive threshold → binary mask
      4. Find contours → count anomalous regions
      5. Compute anomaly area percentage
      6. Classify anomaly type based on features

    Returns a dict with detection metadata.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Denoise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Adaptive thresholding — works better than global threshold on varied lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=4
    )

    # Morphological clean-up: close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny noise contours (< 50 px area)
    significant = [c for c in contours if cv2.contourArea(c) > 50]

    # Total anomalous pixel area
    total_pixels = cv_img.shape[0] * cv_img.shape[1]
    anomaly_px = sum(cv2.contourArea(c) for c in significant)
    anomaly_area_pct = min((anomaly_px / total_pixels) * 100, 100.0)

    # Edge density via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / total_pixels

    # Brightness statistics
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    # Classify anomaly type based on heuristics
    anomaly_type = _classify_type(
        contour_count=len(significant),
        anomaly_area_pct=anomaly_area_pct,
        edge_density=edge_density,
        brightness_std=brightness_std,
    )

    # Bounding boxes of top-5 largest contours
    bboxes = []
    sorted_contours = sorted(significant, key=cv2.contourArea, reverse=True)[:5]
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {
        "anomaly_type": anomaly_type,
        "contour_count": len(significant),
        "anomaly_area_pct": round(anomaly_area_pct, 3),
        "edge_density": round(edge_density, 4),
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std": round(brightness_std, 2),
        "bboxes": bboxes,
        "contours": sorted_contours,  # raw contours for overlay drawing
    }


def _classify_type(
    contour_count: int,
    anomaly_area_pct: float,
    edge_density: float,
    brightness_std: float,
) -> str:
    """
    Rule-based anomaly type classification.
    Returns a human-readable category string.
    """
    if contour_count == 0:
        return "No Anomaly Detected"
    if edge_density > 0.12:
        return "Edge Distortion"
    if anomaly_area_pct > 35:
        return "Structural Change"
    if contour_count > 20:
        return "Pattern Break"
    if brightness_std > 70:
        return "Lighting Irregularity"
    if anomaly_area_pct > 10:
        return "Object Detected"
    if brightness_std < 20 and contour_count > 3:
        return "Texture Anomaly"
    return "Unknown Intrusion"


def compute_risk_score(result: dict) -> Tuple[int, str]:
    """
    Compute a 0-100 risk score from detection result dict.

    Scoring weights:
      - Anomaly area %  : 40 pts max
      - Contour count   : 30 pts max
      - Edge density    : 20 pts max
      - Brightness std  : 10 pts max

    Returns (score: int, risk_level: str).
    """
    area_pct   = result.get("anomaly_area_pct", 0.0)
    raw_contours = result.get("contour_count", 0)
    n_contours = raw_contours if isinstance(raw_contours, (int, float)) else 0
    edge_dens  = result.get("edge_density", 0.0)
    br_std     = result.get("brightness_std", 0.0)

    # Clamp each component to its max
    area_score    = min(area_pct / 100 * 40, 40)
    contour_score = min(n_contours / 50 * 30, 30)
    edge_score    = min(edge_dens / 0.2 * 20, 20)
    std_score     = min(br_std / 100 * 10, 10)

    total = int(area_score + contour_score + edge_score + std_score)
    total = max(0, min(total, 100))

    if total < 30:
        level = "LOW"
    elif total < 65:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return total, level
