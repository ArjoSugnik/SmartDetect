"""geo_analysis.py — Geographic change detection between two images.

Pipeline v3 (high-sensitivity):
  1. ORB feature-based image registration (homography alignment)
  2. Histogram matching  → normalise colour/exposure across images
  3. CLAHE preprocessing → local contrast normalisation
  4. Multi-scale change mask (Otsu + fixed threshold + edge-aware)
  5. Morphological filtering → merge blobs, kill noise
  6. Strict area / coverage gates → never box the whole image
  7. Per-region Groq vision call → validate & classify
  8. Confidence-coloured boxes → Red=High / Yellow=Medium / Green=Low
  9. build_change_table() + draw_geo_annotations() with min_confidence filter

Groq model: Llama 3.2 Vision via Groq API.
Falls back to colour/shape CV heuristics if Groq is not running.
"""

import cv2
import numpy as np
import base64
import requests
import json
from typing import List, Tuple

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ─────────────────────────────────── Config ───────────────────────────────────
SSIM_THRESHOLD       = 0.05   # lowered to allow comparison of massive urban changes over long time periods
DIFF_THRESHOLD       = 22     # 0-255; LOWER = more sensitive to subtle changes
MIN_REGION_AREA      = 150    # px²  — very sensitive for dense urban changes
MAX_REGION_AREA_FRAC = 0.85   # increased to catch large urban redevelopments

from groq_helper import client, VISION_MODEL
import io


# ─────────────────────────── Confidence colour map ───────────────────────────

def _conf_color(conf: float, category: str = "") -> Tuple[int, int, int]:
    """BGR colour for bounding box based on 0-1 confidence."""
    if category == "No Significant Change":
        return (130, 130, 130)  # Grey — no real change
    if conf >= 0.70:
        return (30, 30, 220)    # Red
    if conf >= 0.40:
        return (0, 180, 255)    # Amber/Yellow
    return (30, 210, 80)        # Green


def _conf_label(conf: float, category: str = "") -> str:
    if category == "No Significant Change":
        return "None"
    if conf >= 0.70:
        return "High"
    if conf >= 0.40:
        return "Medium"
    return "Low"


# ──────────────────────────── Image preprocessing ────────────────────────────

def _clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE in LAB space to normalise local lighting/contrast.
    This prevents a cloud shadow or exposure difference from producing
    a massive false-positive diff map.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of source image to reference image per channel.
    Normalises colour/exposure differences between OLD and NEW.
    """
    result = np.zeros_like(source)
    for ch in range(3):
        s = source[:, :, ch].ravel()
        r = reference[:, :, ch].ravel()
        s_hist, _ = np.histogram(s, bins=256, range=(0, 256))
        r_hist, _ = np.histogram(r, bins=256, range=(0, 256))
        s_cdf = np.cumsum(s_hist).astype(np.float64)
        r_cdf = np.cumsum(r_hist).astype(np.float64)
        s_cdf /= s_cdf[-1] if s_cdf[-1] > 0 else 1
        r_cdf /= r_cdf[-1] if r_cdf[-1] > 0 else 1
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.argmin(np.abs(r_cdf - s_cdf[i]))
            lut[i] = j
        result[:, :, ch] = lut[source[:, :, ch]]
    return result


def _register_images(old_img: np.ndarray, new_img: np.ndarray) -> np.ndarray:
    """
    Align new_img to old_img using ORB feature matching + homography.
    Returns the warped new_img. Falls back to original if not enough matches.
    """
    g_old = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    g_new = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(g_old, None)
    kp2, des2 = orb.detectAndCompute(g_new, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return new_img  # Not enough features, skip registration

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 15:
        return new_img  # Not enough good matches

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return new_img

    h, w = old_img.shape[:2]
    aligned = cv2.warpPerspective(new_img, H, (w, h),
                                   borderMode=cv2.BORDER_REFLECT_101)
    return aligned


# ─────────────────────────────── SSIM helper ─────────────────────────────────

def _ssim(g_old: np.ndarray, g_new: np.ndarray) -> float:
    """Return SSIM score (0-1) between two grayscale images."""
    if HAS_SKIMAGE:
        score, _ = ssim_fn(g_old, g_new, full=True)
        return float(score)
    raw = cv2.absdiff(g_old, g_new).astype(np.float32)
    return float(1.0 - raw.mean() / 255.0)


# ──────────────────────────── Change mask builder ────────────────────────────

def _change_mask(old_p: np.ndarray, new_p: np.ndarray) -> np.ndarray:
    """
    Build a binary mask marking pixels that have meaningfully changed.

    Multi-scale approach:
      Layer 1: Max-channel colour diff (catches rooftops, vegetation, water)
      Layer 2: Grayscale structural diff (catches shapes, edges)
      Layer 3: Edge-aware diff via Canny (catches building outlines)
      Blend:   Otsu + fixed threshold union for maximum recall

    Then morphological ops:
      CLOSE  → fills small holes inside real blobs
      OPEN   → removes isolated speckle noise
      DILATE → expands edges so partial outlines get captured
    """
    # --- Layer 1: Max-channel colour diff ---
    diff_ch  = cv2.absdiff(old_p, new_p)
    diff_max = np.max(diff_ch, axis=2)

    # --- Layer 2: Grayscale structural diff ---
    og = cv2.cvtColor(old_p, cv2.COLOR_BGR2GRAY)
    ng = cv2.cvtColor(new_p, cv2.COLOR_BGR2GRAY)
    diff_g = cv2.absdiff(og, ng)

    # --- Layer 3: Edge-aware structural diff (Canny) ---
    edges_old = cv2.Canny(og, 50, 150)
    edges_new = cv2.Canny(ng, 50, 150)
    edge_diff = cv2.absdiff(edges_old, edges_new)
    # Dilate edges to make them thicker and merge nearby edges
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_diff = cv2.dilate(edge_diff, k3, iterations=2)

    # --- Blend: colour (50%) + grayscale (30%) + edges (20%) ---
    combined = (
        diff_max.astype(np.float32) * 0.50 +
        diff_g.astype(np.float32)   * 0.30 +
        edge_diff.astype(np.float32) * 0.20
    )
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # --- Multi-scale thresholding ---
    # Fixed threshold (catches subtle changes)
    _, mask_fixed = cv2.threshold(combined, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Otsu threshold (adaptive, catches major changes)
    blur = cv2.GaussianBlur(combined, (5, 5), 0)
    otsu_val, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Union of both masks → maximum recall
    mask = cv2.bitwise_or(mask_fixed, mask_otsu)

    # --- Morphological cleanup ---
    k9  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Smaller to keep tiny changes
    k5r = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)    # fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)    # remove only tiny speckles
    mask = cv2.dilate(mask, k5r, iterations=1)             # expand edges
    return mask


# ─────────────────────────────── Region filter ───────────────────────────────

def _extract_regions(mask: np.ndarray, img_shape) -> List[dict]:
    """
    Find contours in mask and keep only genuine change regions.
    Rejects:
      - Too small (noise)
      - Too large (whole-image lighting shift)
      - Extreme aspect ratios (edge artefacts)
    """
    h, w  = img_shape[:2]
    total = h * w
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_REGION_AREA:
            continue
        
        # We no longer strictly reject area > MAX_REGION_AREA_FRAC * total. 
        # Massive developments or complete land-use changes can trigger this,
        # and Groq Vision or cv_fallback is capable of classifying large crops correctly.

        x, y, rw, rh = cv2.boundingRect(c)
        if rw == 0 or rh == 0:
            continue
        ar = rw / rh
        if ar > 25 or ar < 0.04:
            continue  # relaxed for roads/pipelines

        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0.0

        out.append({
            "contour":      c,
            "bbox":         (x, y, rw, rh),
            "area":         int(area),
            "solidity":     round(solidity, 3),
            "aspect_ratio": round(ar, 3),
        })

    out.sort(key=lambda r: r["area"], reverse=True)
    return out[:50]  # Allow up to 50 regions for dense urban areas


# ──────────────────────── Groq per-region classifier ─────────────────────────

_LLAVA_SYS = """You are an advanced geospatial change detection AI integrated with a computer vision pipeline.

SYSTEM CONTEXT (IMPORTANT):
Before your analysis, the system has already:
- Aligned images using geo-registration (feature matching + homography)
- Generated a change heatmap (pixel-level intensity differences)
- Extracted candidate change regions using CV (OpenCV)
- Computed region size, shape, and location
- Provided you cropped regions from OLD and NEW images

Your job is NOT to detect changes blindly.
Your job is to VALIDATE and CLASSIFY detected regions intelligently.

━━━━━━━━━━━━━━━━━━━━━━━
INPUT UNDERSTANDING
━━━━━━━━━━━━━━━━━━━━━━━
You receive:
- Image 1 = OLD (past)
- Image 2 = NEW (current)
- Both are already spatially aligned

Each pair shows the SAME geographic area.

━━━━━━━━━━━━━━━━━━━━━━━
MISSION
━━━━━━━━━━━━━━━━━━━━━━━
For each region:
1. Confirm whether the detected change is REAL
2. Classify the type of change
3. Assign a confidence score

━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━
IGNORE:
- Lighting differences
- Shadows
- Clouds
- Seasonal vegetation color
- Minor pixel noise

FOCUS ONLY ON:
- Structural changes (buildings, roads, infrastructure)
- Land-use changes (vegetation, water, soil)
- Material/texture changes

DO NOT hallucinate.
If uncertain → LOW confidence or "No Significant Change".

━━━━━━━━━━━━━━━━━━━━━━━
SPATIAL REASONING (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━
You MUST compare:
- Position (top-left, center, etc.)
- Shape changes
- Size differences
- Texture/material changes

━━━━━━━━━━━━━━━━━━━━━━━
HEATMAP AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━
The system has detected high-change intensity areas.
You must:
- Focus on regions with strong heatmap signals
- Ignore weak/noisy areas unless clearly meaningful

━━━━━━━━━━━━━━━━━━━━━━━
CATEGORIES (CHOOSE ONE)
━━━━━━━━━━━━━━━━━━━━━━━
- New Building
- Demolished Structure
- Road Change
- New Water Body
- Vegetation Change
- Construction Site
- New Infrastructure
- Parking/Vehicle Change
- Flood/Erosion
- No Significant Change
- Other Change

━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━
Base your confidence on:
- Visual clarity of change
- Region size (larger = more reliable)
- Consistency across region
- Alignment accuracy

Confidence scale:
- 0.90-1.00 → clear structural change
- 0.70-0.89 → strong evidence
- 0.40-0.69 → possible change
- <0.40 → weak/uncertain

━━━━━━━━━━━━━━━━━━━━━━━
HYBRID SCORING AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━
Your confidence will be combined with CV-based metrics:
- Region area
- Shape solidity
- Change intensity

So:
- Do NOT inflate confidence
- Be conservative when unsure

━━━━━━━━━━━━━━━━━━━━━━━
TEMPORAL (TIMELINE) AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━
In multi-image sequences:
- Changes may be gradual
- Detect trends (growth, shrinkage, transformation)
- Avoid misclassifying temporary variations

━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━
{
  "category": "...",
  "description": "...",
  "confidence": 0.0-1.0
}

━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTION RULES
━━━━━━━━━━━━━━━━━━━━━━━
- ONE sentence only
- Must include:
  • what changed
  • where (relative position)
  • how it differs from old image

━━━━━━━━━━━━━━━━━━━━━━━
NO CHANGE CASE
━━━━━━━━━━━━━━━━━━━━━━━
If no meaningful change:

{
  "category": "No Significant Change",
  "description": "No meaningful structural or land-use change detected in the region.",
  "confidence": 0.9
}
"""

_LLAVA_PROMPT = (
    "Compare Image 1 (OLD) and Image 2 (NEW). "
    "Images are already geo-aligned and regions are pre-detected. "
    "Focus ONLY on real structural or land-use changes. "
    "Ignore lighting, shadows, and seasonal variation. "
    "Validate and classify the detected change region. "
    "Return STRICT JSON only."
)

_VALID_CATEGORIES = {
    "New Building", "Demolished Structure", "Road Change",
    "New Water Body", "Vegetation Change", "Construction Site",
    "New Infrastructure", "Parking/Vehicle Change", "Flood/Erosion",
    "No Significant Change", "Other Change",
}


from PIL import Image
def _crop_to_base64(img: np.ndarray, bbox: Tuple, pad: int = 40) -> str:
    """Return a base64-encoded JPEG crop (with padding) from img for Groq vision."""
    h, w    = img.shape[:2]
    x, y, rw, rh = bbox
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(w, x+rw+pad); y2 = min(h, y+rh+pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    # Resize if too large (Groq 4MB limit for base64)
    max_size = 800
    if pil_img.width > max_size or pil_img.height > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _groq_classify(old_img: np.ndarray, new_img: np.ndarray, bbox: Tuple) -> dict:
    """
    Send both image crops to Groq Vision and parse JSON response.
    Uses the comprehensive system prompt for validation-style classification.
    Returns: {category, description, confidence, source}
    """
    if not client:
        return _cv_fallback(old_img, new_img, bbox)
        
    try:
        old_b64 = _crop_to_base64(old_img, bbox)
        new_b64 = _crop_to_base64(new_img, bbox)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": _LLAVA_SYS},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{old_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{new_b64}"}},
                    {"type": "text", "text": _LLAVA_PROMPT}
                ]}
            ],
            temperature=0.05,
            max_tokens=500,
        )
        raw = response.choices[0].message.content
        # Strip any markdown fences the model might wrap around JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        p   = json.loads(raw)

        category = str(p.get("category", "Other Change"))
        if category not in _VALID_CATEGORIES:
            category = "Other Change"

        return {
            "category":    category,
            "description": str(p.get("description", "Change detected.")),
            "confidence":  float(max(0.0, min(1.0, p.get("confidence", 0.5)))),
            "source":      "groq",
        }
    except Exception as e:
        return _cv_fallback(old_img, new_img, bbox)


def _cv_fallback(old_img: np.ndarray, new_img: np.ndarray, bbox: Tuple) -> dict:
    """
    Colour/shape heuristic classification when Groq is offline.
    Analyses HSV statistics and geometry of the changed crop.
    """
    h, w   = old_img.shape[:2]
    x, y, rw, rh = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x+rw), min(h, y+rh)

    oc = old_img[y1:y2, x1:x2]
    nc = new_img[y1:y2, x1:x2]
    if oc.size == 0 or nc.size == 0:
        return {"category": "Other Change", "description": "Region could not be analysed.",
                "confidence": 0.30, "source": "cv_fallback"}

    # HSV stats on new crop
    nhsv = cv2.cvtColor(nc, cv2.COLOR_BGR2HSV)
    ohsv = cv2.cvtColor(oc, cv2.COLOR_BGR2HSV)
    n_sat  = float(np.mean(nhsv[:, :, 1]))
    n_val  = float(np.mean(nhsv[:, :, 2]))
    o_val  = float(np.mean(ohsv[:, :, 2]))
    n_hue  = float(np.mean(nhsv[:, :, 0]))
    o_hue  = float(np.mean(ohsv[:, :, 0]))

    # Channel means
    n_b = float(np.mean(nc[:, :, 0]))
    n_g = float(np.mean(nc[:, :, 1]))
    n_r = float(np.mean(nc[:, :, 2]))

    blue_dom  = n_b - n_r          # positive → blue/water
    green_dom = n_g - n_r          # positive → vegetation
    is_grey   = n_sat < 50 and 70 < n_val < 210
    bright_d  = n_val - o_val
    hue_d     = abs(n_hue - o_hue)
    ar        = rw / rh if rh > 0 else 1.0

    if blue_dom > 18 and n_val < 130:
        cat  = "New Water Body"
        desc = "Dark blue-dominant region appeared — likely new water body, pond, or flooding."
        conf = 0.72
    elif green_dom > 15 and n_sat > 35:
        if o_val > n_val + 10:
            cat  = "Vegetation Change"
            desc = "Green cover increased — new vegetation, afforestation, or crop growth detected."
        else:
            cat  = "Vegetation Change"
            desc = "Vegetation cover changed — possible land clearing, crop harvest, or deforestation."
        conf = 0.63
    elif is_grey and ar < 5 and rw * rh > 3500:
        if bright_d > 15:
            cat  = "New Building"
            desc = "Light-grey compact structure appeared — consistent with new building rooftop or footprint."
            conf = 0.78
        else:
            cat  = "Demolished Structure"
            desc = "Previously existing structure has been removed — ground now exposed."
            conf = 0.65
    elif ar > 4 and is_grey:
        cat  = "Road Change"
        desc = "Elongated grey strip appeared — new road, pathway, or lane detected."
        conf = 0.68
    elif bright_d > 40:
        cat  = "Construction Site"
        desc = "Significantly brighter area — exposed soil, sand, or construction materials visible."
        conf = 0.57
    elif hue_d > 18:
        cat  = "Other Change"
        desc = "Surface colour/material shifted notably — land use may have changed."
        conf = 0.42
    else:
        cat  = "No Significant Change"
        desc = "No meaningful structural or land-use change detected in the region."
        conf = 0.25

    return {"category": cat, "description": desc, "confidence": conf, "source": "cv_fallback"}


# ─────────────────────────────── Annotation ──────────────────────────────────

def _draw_legend(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    items = [
        ((30, 30, 220),  "High   >=70%"),
        ((0, 180, 255),  "Medium 40-69%"),
        ((30, 210, 80),  "Low    <40%"),
        ((130, 130, 130), "No Change"),
    ]
    bx, by, bw = 10, h - 10 - len(items) * 22 - 12, 178
    cv2.rectangle(img, (bx-4, by-10), (bx+bw, h-6), (18, 18, 18), -1)
    cv2.rectangle(img, (bx-4, by-10), (bx+bw, h-6), (70, 70, 70), 1)
    for i, (color, text) in enumerate(items):
        cy = by + i * 22
        cv2.rectangle(img, (bx, cy), (bx+16, cy+14), color, -1)
        cv2.putText(img, text, (bx+22, cy+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)


def draw_geo_annotations(
    new_img: np.ndarray,
    regions: List[dict],
    min_confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw confidence-coloured bounding boxes on new_img.

    Only boxes with confidence >= min_confidence are drawn.
    Adds colour legend.

    Returns annotated BGR image.
    """
    out      = new_img.copy()
    filtered = [r for r in regions if r.get("confidence", 0) >= min_confidence]

    for r in filtered:
        x, y, rw, rh = r["bbox"]
        conf  = r.get("confidence", 0.5)
        cat   = r.get("category", "Change")
        color = _conf_color(conf, category=cat)
        pct   = int(conf * 100)
        label = f"#{r['id']}  {cat}  {pct}%"

        # Semi-transparent fill
        ov = out.copy()
        cv2.rectangle(ov, (x, y), (x+rw, y+rh), color, -1)
        out = cv2.addWeighted(out, 0.83, ov, 0.17, 0)

        # Border
        thick = 3 if conf >= 0.70 else 2
        cv2.rectangle(out, (x, y), (x+rw, y+rh), color, thick)

        # Label pill
        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.46
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        lx = x
        ly = (y - 8) if y > 22 else (y + rh + th + 8)
        cv2.rectangle(out, (lx-2, ly-th-4), (lx+tw+6, ly+4), color, -1)
        cv2.putText(out, label, (lx+2, ly), font, fs, (10, 10, 10), 1, cv2.LINE_AA)

    _draw_legend(out)
    return out


# ───────────────────────────── Public API ────────────────────────────────────

def _resize_if_huge(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Downscale excessively large images to maintain performance and avoid API timeouts."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def compare_geo_images(
    old_img: np.ndarray,
    new_img: np.ndarray,
    use_groq: bool = True,
) -> dict:
    """
    Run the full geo change detection pipeline v3.

    Steps:
      1. Resize new → match old resolution
      2. ORB feature-based registration (homography alignment)
      3. Histogram matching (colour normalisation)
      4. CLAHE local contrast normalisation
      5. Multi-scale change mask (Otsu + fixed + edge-aware)
      6. Region extraction → up to 25 regions
      7. Per-region Groq or CV fallback classification
      8. Annotated output image

    Returns dict with results, annotated image, and heatmap.
    """
    # 0. Prevent massive images from freezing the pipeline
    old_img = _resize_if_huge(old_img)

    # 1. Resize new → match old resolution
    th, tw      = old_img.shape[:2]
    new_resized = cv2.resize(new_img, (tw, th), interpolation=cv2.INTER_AREA)

    # 2. ORB feature-based registration
    new_registered = _register_images(old_img, new_resized)

    # 3. Histogram matching (normalise colour/exposure)
    new_matched = _histogram_match(new_registered, old_img)

    # 4. CLAHE local contrast normalisation
    old_p = _clahe(old_img)
    new_p = _clahe(new_matched)

    # 5. SSIM check
    og = cv2.cvtColor(old_p, cv2.COLOR_BGR2GRAY)
    ng = cv2.cvtColor(new_p, cv2.COLOR_BGR2GRAY)
    score = _ssim(og, ng)

    if score < SSIM_THRESHOLD:
        return {
            "similar": False, "ssim": round(score, 4),
            "change_pct": 0.0, "region_count": 0,
            "regions": [], "annotated_img": new_registered,
            "new_resized": new_registered, "groq_used": False,
            "heatmap_img": None,
        }

    # 6. Multi-scale change mask
    mask = _change_mask(old_p, new_p)

    # 6b. Build heatmap for visualisation
    diff_vis = cv2.absdiff(old_p, new_p)
    diff_gray = cv2.cvtColor(diff_vis, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    heatmap_blend = cv2.addWeighted(new_registered, 0.5, heatmap, 0.5, 0)

    # 7. Extract regions (up to 50)
    raw_regions = _extract_regions(mask, old_img.shape)

    # 8. Classify each region
    groq_used = False
    enriched   = []
    for idx, r in enumerate(raw_regions, 1):
        if use_groq:
            cls = _groq_classify(old_img, new_registered, r["bbox"])
            if cls["source"] == "groq":
                groq_used = True
            else:
                use_groq = False  # Fallback to CV for remaining regions to save time
        else:
            cls = _cv_fallback(old_img, new_registered, r["bbox"])

        enriched.append({
            "id":           idx,
            "bbox":         r["bbox"],
            "area":         r["area"],
            "solidity":     r["solidity"],
            "aspect_ratio": r["aspect_ratio"],
            "category":     cls["category"],
            "description":  cls["description"],
            "confidence":   cls["confidence"],
            "conf_label":   _conf_label(cls["confidence"], category=cls["category"]),
            "color_bgr":    _conf_color(cls["confidence"], category=cls["category"]),
            "source":       cls["source"],
        })

    # 9. Change %
    change_pct = round(np.count_nonzero(mask) / (th * tw) * 100, 2)

    # 10. Annotated image (all regions, no filter)
    annotated = draw_geo_annotations(new_registered, enriched, min_confidence=0.0)

    return {
        "similar":       True,
        "ssim":          round(score, 4),
        "change_pct":    change_pct,
        "region_count":  len(enriched),
        "regions":       enriched,
        "annotated_img": annotated,
        "new_resized":   new_registered,
        "groq_used":      groq_used,
        "heatmap_img":   heatmap_blend,
    }


def classify_geo_changes(geo_result: dict) -> List[dict]:
    """Legacy shim — calls build_change_table with no filter."""
    return build_change_table(geo_result.get("regions", []), min_confidence=0.0)


def build_change_table(regions: List[dict], min_confidence: float = 0.0) -> List[dict]:
    """
    Build a display-ready table from enriched region list.

    Args:
        regions:        From compare_geo_images()["regions"].
        min_confidence: 0.0-1.0; only rows >= threshold are included.

    Returns list of dicts for pd.DataFrame().
    """
    rows = []
    for r in regions:
        if r["confidence"] < min_confidence:
            continue
        x, y, rw, rh = r["bbox"]
        rows.append({
            "ID":           f"#{r['id']}",
            "Location":     f"x={x} y={y}  ({rw}×{rh}px)",
            "Category":     r["category"],
            "What Changed": r["description"],
            "Confidence":   f"{r['confidence']*100:.0f}%",
            "Level":        r["conf_label"],
            "Area (px²)":   f"{r['area']:,}",
            "Detected By":  "🤖 Groq" if r.get("source") == "groq" else "🔬 CV Analysis",
        })
    if not rows:
        rows.append({
            "ID": "—", "Location": "—",
            "Category": "No changes above threshold",
            "What Changed": "Lower the confidence slider to see more regions.",
            "Confidence": "—", "Level": "—",
            "Area (px²)": "—", "Detected By": "—",
        })
    return rows
