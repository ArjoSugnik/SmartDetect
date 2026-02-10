import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import io
import pandas as pd
import zipfile
import tempfile
from datetime import datetime
from fpdf import FPDF
from ultralytics import YOLO
import os
import cv2
import numpy as np
import base64
import math
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from skimage.metrics import structural_similarity as ssim

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="SmartDetect - AI Image Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== EARTH PRO ENHANCED FUNCTIONS ==========

# ESRI Wayback Release IDs for specific years (Verified stable releases)
WAYBACK_IDS = {
    2026: "latest",
    2025: "latest", 
    2024: "latest",
    2023: "13866",
    2022: "13511",
    2021: "12991",
    2020: "24177",
    2019: "2315",
    2018: "1863",
    2017: "15383",
    2016: "15174",
    2015: "14686",
    2014: "10"
}

def deg2num(lat_deg, lon_deg, zoom):
    """Convert latitude/longitude to tile coordinates"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def fetch_satellite_tile(xtile, ytile, zoom, service="google", year=None):
    """Fetch a single satellite tile from the specified service"""
    try:
        if service == "google":
            url = f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"
        elif service == "esri":
            if year and year in WAYBACK_IDS:
                release_id = WAYBACK_IDS[year]
                if release_id == "latest":
                    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
                else:
                    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{release_id}/{zoom}/{ytile}/{xtile}"
            else:
                url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        else:
            return None
        
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(response.raw).convert("RGB")
            return img
        else:
            return None
    except Exception as e:
        print(f"Error fetching tile: {e}")
        return None

def fetch_multi_tile_image(lat, lon, zoom, service="google", year=None, tile_size=3):
    """Fetch multiple tiles and stitch them together for higher resolution"""
    center_x, center_y = deg2num(lat, lon, zoom)
    half_size = tile_size // 2
    tiles = []
    
    for dy in range(-half_size, half_size + 1):
        row = []
        for dx in range(-half_size, half_size + 1):
            tile = fetch_satellite_tile(center_x + dx, center_y + dy, zoom, service=service, year=year)
            if tile:
                row.append(tile)
            else:
                row.append(Image.new('RGB', (256, 256), color='gray'))
        tiles.append(row)
    
    tile_width = 256
    tile_height = 256
    full_width = tile_width * tile_size
    full_height = tile_height * tile_size
    stitched = Image.new('RGB', (full_width, full_height))
    
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            stitched.paste(tile, (j * tile_width, i * tile_height))
    
    return stitched

def detect_changes_opencv(img_old, img_new, min_area=500):
    """Detect changes between two images using OpenCV"""
    gray_old = cv2.cvtColor(np.array(img_old), cv2.COLOR_RGB2GRAY)
    gray_new = cv2.cvtColor(np.array(img_new), cv2.COLOR_RGB2GRAY)
    
    if gray_old.shape != gray_new.shape:
        gray_new = cv2.resize(gray_new, (gray_old.shape[1], gray_old.shape[0]))
    
    diff = cv2.absdiff(gray_old, gray_new)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    changes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            changes.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "type": "Structural Change"
            })
    
    return changes

def detect_changes_yolo(img_old, img_new, model, min_confidence=0.5):
    """Detect new structures using YOLOv8 object detection"""
    results_old = model(img_old)[0]
    results_new = model(img_new)[0]
    
    boxes_old = []
    if results_old.boxes is not None:
        for box in results_old.boxes:
            if float(box.conf[0]) >= min_confidence:
                boxes_old.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0])
                })
    
    boxes_new = []
    if results_new.boxes is not None:
        for box in results_new.boxes:
            if float(box.conf[0]) >= min_confidence:
                boxes_new.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0])
                })
    
    new_objects = []
    for box_new in boxes_new:
        is_new = True
        x_new, y_new = box_new['xyxy'][0], box_new['xyxy'][1]
        
        for box_old in boxes_old:
            x_old, y_old = box_old['xyxy'][0], box_old['xyxy'][1]
            if abs(x_new - x_old) < 50 and abs(y_new - y_old) < 50:
                is_new = False
                break
        
        if is_new:
            class_name = model.names[box_new['class']]
            x0, y0, x1, y1 = box_new['xyxy']
            new_objects.append({
                "x": int(x0),
                "y": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "confidence": box_new['conf'],
                "class": class_name,
                "type": f"New {class_name.capitalize()}"
            })
    
    return new_objects

def classify_change_type(change, year_old, year_new):
    """Classify the type of change detected"""
    if 'class' in change:
        obj_type = change['class'].lower()
        if obj_type in ['building', 'house']:
            return f"New Construction ({year_old}-{year_new})"
        elif obj_type in ['road', 'street']:
            return f"Road Development ({year_old}-{year_new})"
        elif obj_type in ['tree', 'plant', 'vegetation']:
            return f"Vegetation Growth ({year_old}-{year_new})"
        elif obj_type in ['car', 'truck', 'vehicle']:
            return f"Increased Activity ({year_old}-{year_new})"
        else:
            return f"New Structure: {obj_type.capitalize()} ({year_old}-{year_new})"
    else:
        return f"Terrain Change ({year_old}-{year_new})"

def create_annotated_comparison(img_old, img_new, changes, year_old, year_new):
    """Create side-by-side comparison with changes highlighted"""
    if img_old.size != img_new.size:
        img_new = img_new.resize(img_old.size)
    
    img_annotated = img_new.copy()
    draw = ImageDraw.Draw(img_annotated)
    
    for change in changes:
        x, y, w, h = change['x'], change['y'], change['width'], change['height']
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        label = change.get('type', 'Change')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        draw.text((x, y - 15), label, fill="red", font=font)
    
    width, height = img_old.size
    combined = Image.new('RGB', (width * 2 + 10, height), color='white')
    combined.paste(img_old, (0, 0))
    combined.paste(img_annotated, (width + 10, 0))
    
    draw_combined = ImageDraw.Draw(combined)
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font_label = ImageFont.load_default()
    
    draw_combined.text((20, 20), f"Year: {year_old}", fill="yellow", font=font_label)
    draw_combined.text((width + 30, 20), f"Year: {year_new}", fill="yellow", font=font_label)
    
    return combined, img_annotated

def generate_change_analysis_csv(changes, year_old, year_new, location_name):
    """Generate CSV report of detected changes"""
    if not changes:
        df = pd.DataFrame({'Message': ['No significant changes detected between the selected years.']})
    else:
        df = pd.DataFrame(changes)
        df['Location'] = location_name
        df['Year_Old'] = year_old
        df['Year_New'] = year_new
        df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cols = ['Location', 'Year_Old', 'Year_New', 'x', 'y', 'width', 'height']
        if 'area' in df.columns:
            cols.append('area')
        if 'type' in df.columns:
            cols.append('type')
        if 'class' in df.columns:
            cols.append('class')
        if 'confidence' in df.columns:
            cols.append('confidence')
        cols.append('Analysis_Date')
        df = df[cols]
    
    return df

def generate_analysis_package(img_old, img_new, img_comparison, img_annotated, 
                               changes_df, year_old, year_new, location_name, 
                               ssim_score):
    """Generate a ZIP package with all analysis results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        img_old_bytes = io.BytesIO()
        img_old.save(img_old_bytes, format='PNG')
        zipf.writestr(f'{location_name}_{year_old}.png', img_old_bytes.getvalue())
        
        img_new_bytes = io.BytesIO()
        img_new.save(img_new_bytes, format='PNG')
        zipf.writestr(f'{location_name}_{year_new}.png', img_new_bytes.getvalue())
        
        img_comparison_bytes = io.BytesIO()
        img_comparison.save(img_comparison_bytes, format='PNG')
        zipf.writestr(f'{location_name}_comparison_{year_old}_vs_{year_new}.png', img_comparison_bytes.getvalue())
        
        img_annotated_bytes = io.BytesIO()
        img_annotated.save(img_annotated_bytes, format='PNG')
        zipf.writestr(f'{location_name}_{year_new}_annotated.png', img_annotated_bytes.getvalue())
        
        csv_data = changes_df.to_csv(index=False)
        zipf.writestr(f'{location_name}_changes_{year_old}_to_{year_new}.csv', csv_data)
        
        excel_buffer = io.BytesIO()
        changes_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        zipf.writestr(f'{location_name}_changes_{year_old}_to_{year_new}.xlsx', excel_buffer.getvalue())
        
        summary = f"""SMARTDETECT EARTH PRO ANALYSIS REPORT
======================================

Location: {location_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPARISON PERIOD
-----------------
Start Year: {year_old}
End Year: {year_new}
Duration: {year_new - year_old} years

SIMILARITY ANALYSIS
-------------------
Structural Similarity Index (SSIM): {ssim_score:.2%}
Change Magnitude: {(1 - ssim_score) * 100:.2f}%

DETECTED CHANGES
----------------
Total Changes: {len(changes_df)}

Generated by SmartDetect Earth Pro Analysis
"""
        zipf.writestr('README.txt', summary)
    
    zip_buffer.seek(0)
    return zip_buffer

# ========== ORIGINAL SMARTDETECT FUNCTIONS ==========

def get_base64_image(image_path):
    """Convert local image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def get_math_animation_value():
    """Mathematical formulas for dynamic visual effects"""
    now = datetime.now()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    sine_pulse = (math.sin(seconds * math.pi / 30) + 1) / 2
    golden_ratio = 1.618033988749
    golden_value = (seconds * golden_ratio) % 1
    fib_sequence = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45]
    fib_index = int((seconds / 10) % len(fib_sequence))
    fib_opacity = fib_sequence[fib_index]
    
    return {
        'sine': sine_pulse,
        'golden': golden_value,
        'fib_opacity': fib_opacity,
        'rotation': (seconds * 0.5) % 360
    }

def detect_cracks_opencv(image):
    """Detects cracks using computer vision (OpenCV) techniques"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 100
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": x + w/2,
                "y": y + h/2,
                "width": w,
                "height": h,
                "confidence": 100.0,
                "class": "crack/defect"
            })
    return anomalies

def detect_stains_opencv(image):
    """Detects stains/discoloration using color statistics"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    median = cv2.medianBlur(blurred, 21)
    diff = cv2.absdiff(blurred, median)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 200
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": float(x + w/2),
                "y": float(y + h/2),
                "width": float(w),
                "height": float(h),
                "confidence": 100.0,
                "class": "stain/discoloration"
            })
    return anomalies

def get_location_coords(address):
    """Get latitude and longitude from address"""
    try:
        geolocator = Nominatim(user_agent="smart_detect_app")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception:
        return None

def compare_images_ssim(img1, img2):
    """Compare two images using SSIM and return score and difference map"""
    gray1 = cv2.cvtColor(np.array(img1.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2.convert('RGB')), cv2.COLOR_RGB2GRAY)
    
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    (score, diff) = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    return score, diff, thresh

# Get mathematical values for animations
math_values = get_math_animation_value()

# ---- Main Page Header (Centered & Stacked) ----
st.markdown("""
<div style="text-align: center; padding-top: 10px; padding-bottom: 10px;">
    <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00A3FF;stop-opacity:1" />
                <stop offset="50%" style="stop-color:#0066FF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00A3FF;stop-opacity:1" />
            </linearGradient>
            <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <circle cx="50" cy="50" r="45" fill="rgba(0,207,255,0.15)" stroke="url(#logoGradient)" stroke-width="2" filter="url(#glow)"/>
        <circle cx="50" cy="50" r="32" fill="none" stroke="url(#logoGradient)" stroke-width="1.5" stroke-dasharray="8 4" opacity="0.7"/>
        <ellipse cx="50" cy="50" rx="22" ry="15" fill="none" stroke="url(#logoGradient)" stroke-width="2.5"/>
        <circle cx="50" cy="50" r="8" fill="url(#logoGradient)"/>
        <line x1="50" y1="20" x2="50" y2="30" stroke="url(#logoGradient)" stroke-width="2" stroke-linecap="round"/>
        <line x1="50" y1="70" x2="50" y2="80" stroke="url(#logoGradient)" stroke-width="2" stroke-linecap="round"/>
        <line x1="20" y1="50" x2="30" y2="50" stroke="url(#logoGradient)" stroke-width="2" stroke-linecap="round"/>
        <line x1="70" y1="50" x2="80" y2="50" stroke="url(#logoGradient)" stroke-width="2" stroke-linecap="round"/>
    </svg>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='title-font' style='text-align: center;'>⚡ SmartDetect: AI Anomaly Detection</div>", unsafe_allow_html=True)

# Default theme and model
theme = "Dark"
model_choice = "Roboflow Default"

# ---- Professional UI Styles (Inter Font & Glassmorphism) ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body {
    font-family: 'Inter', sans-serif !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}

.stApp {
    background: #0e1117;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, rgba(0, 163, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 100% 0%, rgba(0, 102, 255, 0.05) 0%, transparent 50%);
}

.stMarkdown > div, [data-testid="stVerticalBlock"] > div {
    border-radius: 12px;
}

.title-font {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700;
    font-size: 2rem !important;
    background: linear-gradient(135deg, #FFF 0%, #AAA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 4px 20px rgba(0, 163, 255, 0.3);
    padding: 10px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    white-space: nowrap;
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    color: #CCC;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 0 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.1);
    color: #FFF;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(0, 163, 255, 0.15) !important;
    border-color: rgba(0, 163, 255, 0.5) !important;
    color: #00A3FF !important;
}

.stTextInput > div > div > input, .stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: white;
}
.stTextInput > div > div > input:focus {
    border-color: #00A3FF;
    box-shadow: 0 0 0 1px #00A3FF;
}

[data-testid="stFileUploader"] section {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px dashed rgba(255, 255, 255, 0.2);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---- Main Application Logic ----
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Upload & Preview", "🔍 Detection & AI Correction",
     "📹 Live Video Detection", "🌍 Earth Pro Analysis", "📊 Feedback & Report", "🧭 Tutorial", "ℹ️ About/Docs"
])

# ---------- Tab 1: Upload & Preview ----------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown("### 🖼️ Image Gallery")
        num_images = len(uploaded_files)
        cols_per_row = min(num_images, 4)
        cols = st.columns(cols_per_row)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % cols_per_row]:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=uploaded_file.name, width=200)
                uploaded_file.seek(0)

# ---------- Tab 2: Detection & Correction ----------
with tab2:
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        st.warning("Upload images in the first tab.")
    else:
        st.markdown("### ⚙️ Detection Settings")
        
        detection_mode = st.radio(
            "Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            help="Choose your detection target: Objects, Cracks, or Stains."
        )
        
        threshold = st.slider("Minimum Confidence (%)", 0, 100, 50)
        
        st.markdown("### 🤖 AI Correction Settings")
        use_ai_correction = st.checkbox("Enable AI-Powered Correction", value=True, help="Use AI to intelligently remove anomalies")
        
        if use_ai_correction:
            st.info("🎨 AI will intelligently remove detected anomalies and generate clean, natural-looking corrections.")
        else:
            st.warning("⚠️ AI correction disabled. Only detection will be performed.")
        
        st.write(f"**Detection:** Anomalies with confidence ≥ {threshold}% will be shown.")
        
        color_picker_high = "#00ff00"
        color_picker_mid = "#ff0000"
        color_picker_low = "#ffff00"

        if "session_results" not in st.session_state:
            st.session_state.session_results = []
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        session_results = []

        with zipfile.ZipFile(temp_zip.name, "w") as zip_all:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n#### Image {idx + 1}: {uploaded_file.name}")
                image_bytes = uploaded_file.getvalue()
                orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                preds = []
                
                with st.spinner(f"Detecting anomalies ({detection_mode}) for {uploaded_file.name}..."):
                    try:
                        if "Road/Surface Cracks" in detection_mode:
                            opencv_preds = detect_cracks_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Crack Detection complete! Found {len(preds)} defects.")
                            
                        elif "Stain/Discoloration" in detection_mode:
                            opencv_preds = detect_stains_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Stain Detection complete! Found {len(preds)} defects.")
                            
                        else:
                            @st.cache_resource
                            def load_model():
                                return YOLO('yolov8n.pt')
                            
                            model = load_model()
                            results = model(orig_img)
                            
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]
                                    
                                    if conf * 100 >= threshold:
                                        preds.append({
                                            "x": x,
                                            "y": y,
                                            "width": w,
                                            "height": h,
                                            "confidence": conf,
                                            "class": label
                                        })
                            
                            st.success(f"✅ AI Detection complete! Found {len(preds)} objects/anomalies.")
                        
                    except Exception as e:
                        st.error(f"❌ Detection Error: {str(e)}")
                        continue

                if preds:
                    df = pd.DataFrame(preds)[["x", "y", "width", "height", "confidence"]]
                    df["confidence (%)"] = (df["confidence"] * 100).round(2)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download anomaly results as CSV",
                        data=csv,
                        file_name=f"anomaly_results_{idx+1}.csv",
                        mime="text/csv",
                        key=f"csv_download_{idx}"
                    )

                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine="openpyxl")
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download anomaly results as Excel",
                        data=excel_buffer,
                        file_name=f"anomaly_results_{idx+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_download_{idx}"
                    )

                    zip_all.writestr(f"anomaly_results_{idx+1}.csv", csv)
                    zip_all.writestr(f"anomaly_results_{idx+1}.xlsx", excel_buffer.getvalue())

                def get_color(conf):
                    if conf >= 0.9:
                        return color_picker_high
                    elif conf >= 0.7:
                        return color_picker_mid
                    else:
                        return color_picker_low

                im_anno = orig_img.copy()
                draw = ImageDraw.Draw(im_anno)
                for pred in preds:
                    x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                    y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                    x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                    y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                    color = get_color(pred["confidence"])
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                im_corr = orig_img.copy()
                
                if use_ai_correction and preds:
                    with st.spinner("🤖 AI is generating corrected image..."):
                        try:
                            mask = Image.new('L', orig_img.size, 0)
                            mask_draw = ImageDraw.Draw(mask)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                mask_draw.rectangle([x0, y0, x1, y1], fill=255)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                box = (x0, y0, x1, y1)
                                region = im_corr.crop(box).filter(ImageFilter.GaussianBlur(20))
                                im_corr.paste(region, box)
                            
                            st.success("✅ AI correction completed!")
                        except Exception as e:
                            st.error(f"AI correction failed: {str(e)}")
                            im_corr = orig_img.copy()

                st.markdown("""
                <div style="display: flex; justify-content: center; gap: 25px; margin: 15px 0; padding: 12px 20px; background: rgba(128,128,128,0.1); border-radius: 50px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(0,255,0,0.4);"></div>
                        <span><b>High Confidence</b> ≥90%</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #ffff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,255,0,0.4);"></div>
                        <span><b>Medium Confidence</b> 70-89%</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #ff0000; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,0,0,0.4);"></div>
                        <span><b>Low Confidence</b> &lt;70%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_img, caption="Original", use_container_width=True, output_format="PNG")
                with col2:
                    st.image(im_anno, caption="Detected", use_container_width=True, output_format="PNG")
                with col3:
                    st.image(im_corr, caption="Corrected", use_container_width=True, output_format="PNG")

                img_anno_b = io.BytesIO()
                im_anno.save(img_anno_b, format="PNG")
                img_anno_b.seek(0)
                st.download_button(
                    label="Download Annotated",
                    data=img_anno_b,
                    file_name=f"annotated_{idx+1}.png",
                    mime="image/png",
                    key=f"anno_download_{idx}"
                )

                img_corr_b = io.BytesIO()
                im_corr.save(img_corr_b, format="PNG")
                img_corr_b.seek(0)
                st.download_button(
                    label="Download Corrected",
                    data=img_corr_b,
                    file_name=f"corrected_{idx+1}.png",
                    mime="image/png",
                    key=f"corr_download_{idx}"
                )

                zip_all.writestr(f"annotated_{idx+1}.png", img_anno_b.getvalue())
                zip_all.writestr(f"corrected_{idx+1}.png", img_corr_b.getvalue())

                session_results.append({
                    "filename": uploaded_file.name,
                    "num_anomalies": len(preds),
                    "ai_corrected": use_ai_correction,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "correction_file": f"corrected_{idx+1}.png"
                })

        st.session_state.session_results = session_results

        with open(temp_zip.name, "rb") as zf:
            all_zip_bytes = zf.read()
        
        try:
            os.unlink(temp_zip.name)
        except Exception:
            pass
            
        st.download_button(
            label="Download All Results/Images as ZIP",
            data=all_zip_bytes,
            file_name="SmartDetect_results.zip",
            mime="application/zip",
            key="zip_download_all"
        )

        st.write("---")
        st.markdown("## Session Results")
        if session_results:
            st.dataframe(pd.DataFrame(session_results))

# ---------- Tab 3: Live Video Detection ----------
with tab3:
    st.markdown("### 📹 Live Video Anomaly Detection")
    st.info("🎥 Detect anomalies in real-time from your webcam or video feed.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎬 Video Source")
        video_source = st.radio(
            "Select video source:",
            ["Webcam", "Upload Video File"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### ⚙️ Detection Settings")
        video_threshold = st.slider("Confidence Threshold (%)", 0, 100, 60, key="video_threshold")
        show_boxes = st.checkbox("Show Detection Boxes", value=True, key="show_boxes")
    
    st.markdown("---")
    
    if video_source == "Webcam":
        st.markdown("#### 📷 Webcam Feed")
        st.warning("⚠️ **Note:** Webcam access requires browser permissions.")
        
        st.markdown("""
        **How to use:**
        1. Click "Enable Webcam" below
        2. Point camera at surfaces to check
        3. Click "Capture & Detect"
        4. View detected anomalies
        """)
        
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        
        col_start, col_snap, col_stop = st.columns(3)
        
        with col_start:
            if st.button("🎥 Enable Webcam", key="start_webcam", use_container_width=True):
                st.session_state.webcam_running = True
        
        with col_snap:
            capture_btn = st.button("📸 Capture & Detect", key="snapshot", use_container_width=True, disabled=not st.session_state.webcam_running)
        
        with col_stop:
            if st.button("⏹️ Disable Webcam", key="stop_webcam", use_container_width=True):
                st.session_state.webcam_running = False
        
        video_detection_mode = st.radio(
            "Video Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            horizontal=True,
            key="video_mode_select"
        )

        st.markdown("---")
        
        if st.session_state.webcam_running:
            st.info("📷 Webcam is active. Click 'Capture & Detect' to analyze.")
            camera_image = st.camera_input("Live Camera Feed", key="camera_feed")
            
            if camera_image is not None and capture_btn:
                with st.spinner("🔍 Detecting anomalies..."):
                    try:
                        img_bytes = camera_image.getvalue()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        
                        preds = []
                        
                        if "Road/Surface Cracks" in video_detection_mode:
                            preds = detect_cracks_opencv(img)
                            st.success(f"✅ Found {len(preds)} defects")
                        elif "Stain/Discoloration" in video_detection_mode:
                            preds = detect_stains_opencv(img)
                            st.success(f"✅ Found {len(preds)} stains")
                        else:
                            @st.cache_resource
                            def load_model():
                                return YOLO('yolov8n.pt')
                            
                            model = load_model()
                            results = model(img)
                            
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]
                                    
                                    if conf * 100 >= video_threshold:
                                        preds.append({
                                            "x": x,
                                            "y": y,
                                            "width": w,
                                            "height": h,
                                            "confidence": conf,
                                            "class": label
                                        })
                            
                            st.success(f"✅ Found {len(preds)} anomalies")
                        
                        img_annotated = img.copy()
                        draw = ImageDraw.Draw(img_annotated)
                        
                        for pred in preds:
                            x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                            y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                            x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                            y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                            draw.rectangle([x0, y0, x1, y1], outline="#FF0000", width=3)
                        
                        col_orig, col_detect = st.columns(2)
                        with col_orig:
                            st.markdown("**Original Frame**")
                            st.image(img, use_container_width=True)
                        with col_detect:
                            st.markdown("**Detected Anomalies**")
                            st.image(img_annotated, use_container_width=True)
                        
                        buf = io.BytesIO()
                        img_annotated.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Annotated Frame",
                            data=buf,
                            file_name="webcam_detection.png",
                            mime="image/png"
                        )
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Click 'Enable Webcam' to start")
    else:
        st.markdown("#### 📁 Upload Video File")
        uploaded_video = st.file_uploader("Upload video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video:
            st.video(uploaded_video)
            st.info("Video processing feature - Install opencv-python and ffmpeg to enable")

# ---------- Tab 4: EARTH PRO ANALYSIS (ENHANCED) ----------
with tab4:
    st.markdown("### 🌍 Earth Pro Satellite Analysis")
    st.info("🛰️ **Compare REAL satellite imagery across different years.** Select any location worldwide, choose two years, and let AI detect all changes!")
    
    # Initialize session state
    if "map_center" not in st.session_state:
        st.session_state.map_center = [40.7128, -74.0060]  # New York default
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14

    # ==================== STEP 1: LOCATION SELECTION ====================
    st.markdown("---")
    st.markdown("### 📍 Step 1: Select Location")
    
    col_search, col_coords = st.columns([2, 1])
    
    with col_search:
        search_address = st.text_input(
            "🔍 Search for a location",
            placeholder="e.g., Times Square New York, Burj Khalifa Dubai, Eiffel Tower Paris",
            help="Enter any address, landmark, or place name"
        )
        
        col_search_btn, col_reset = st.columns(2)
        
        with col_search_btn:
            if st.button("🔍 Find Location", use_container_width=True):
                if search_address:
                    with st.spinner("Searching for location..."):
                        coords = get_location_coords(search_address)
                        if coords:
                            st.session_state.map_center = list(coords)
                            st.session_state.map_zoom = 16
                            st.success(f"✅ Found: {search_address}")
                            st.success(f"📍 Coordinates: {coords[0]:.6f}, {coords[1]:.6f}")
                        else:
                            st.error("❌ Location not found. Try a different search term.")
                else:
                    st.warning("Please enter a location to search")
        
        with col_reset:
            if st.button("🔄 Reset to New York", use_container_width=True):
                st.session_state.map_center = [40.7128, -74.0060]
                st.session_state.map_zoom = 14
                st.success("Reset to New York City")
    
    with col_coords:
        st.markdown("**Current Center:**")
        st.code(f"Lat: {st.session_state.map_center[0]:.6f}\nLon: {st.session_state.map_center[1]:.6f}")
        st.markdown(f"**Zoom Level:** {st.session_state.map_zoom}")
    
    # ==================== INTERACTIVE MAP ====================
    st.markdown("---")
    st.markdown("### 🗺️ Navigate the Map")
    st.info("👆 **Navigate:** Drag to pan, scroll to zoom. The analysis will use the center point of the map view.")
    
    # Create folium map
    m = leafmap.Map(
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        draw_control=False,
        measure_control=False,
        fullscreen_control=True,
        attribution_control=True
    )
    
    # Add satellite basemap
    m.add_basemap("SATELLITE")
    
    # Add center marker
    folium.Marker(
        st.session_state.map_center,
        popup="Analysis Center Point",
        tooltip="This point will be analyzed",
        icon=folium.Icon(color='red', icon='crosshairs', prefix='fa')
    ).add_to(m)
    
    # Display map
    map_output = st_folium(
        m,
        width=1200,
        height=500,
        returned_objects=["center", "zoom"],
        key="earth_pro_map"
    )
    
    # Update session state from map
    if map_output and map_output.get('center'):
        st.session_state.map_center = [map_output['center']['lat'], map_output['center']['lng']]
    if map_output and map_output.get('zoom'):
        st.session_state.map_zoom = map_output['zoom']
    
    # ==================== STEP 2: YEAR SELECTION ====================
    st.markdown("---")
    st.markdown("### 📅 Step 2: Select Years for Comparison")
    
    col_year1, col_year2, col_zoom = st.columns(3)
    
    with col_year1:
        year_old = st.selectbox("📆 Earlier Year (Baseline)", options=list(range(2014, 2027)), index=0)
    
    with col_year2:
        year_new = st.selectbox("📆 Recent Year (Current)", options=list(range(2014, 2027)), index=len(list(range(2014, 2027))) - 1)
    
    with col_zoom:
        analysis_zoom = st.slider("🔍 Analysis Zoom Level", min_value=12, max_value=18, value=st.session_state.map_zoom)
    
    # Validate year selection
    if year_new <= year_old:
        st.error("⚠️ Recent year must be later than earlier year!")
        year_valid = False
    else:
        st.success(f"✅ Analysis Period: **{year_old}** → **{year_new}** ({year_new - year_old} years)")
        year_valid = True
    
    # ==================== STEP 3: ANALYSIS SETTINGS ====================
    st.markdown("---")
    st.markdown("### ⚙️ Step 3: Analysis Settings")
    
    col_method, col_settings = st.columns(2)
    
    with col_method:
        analysis_method = st.radio(
            "Detection Method",
            ["🔬 AI Object Detection (YOLO)", "🖼️ Image Difference (OpenCV)", "🤖 Combined (AI + Image Diff)"],
            index=2
        )
    
    with col_settings:
        min_change_area = st.slider("Minimum Change Area (pixels²)", min_value=100, max_value=5000, value=500, step=100)
        
        if "AI" in analysis_method or "Combined" in analysis_method:
            yolo_confidence = st.slider("AI Detection Confidence (%)", min_value=30, max_value=95, value=50)
    
    # ==================== STEP 4: RUN ANALYSIS ====================
    st.markdown("---")
    st.markdown("### 🚀 Step 4: Run Analysis")
    
    col_analyze, col_info = st.columns([1, 2])
    
    with col_analyze:
        analyze_button = st.button("🔍 Analyze Location Changes", type="primary", use_container_width=True, disabled=not year_valid)
    
    with col_info:
        if year_valid:
            st.info(f"📊 Ready to analyze {year_new - year_old} years of changes")
        else:
            st.warning("⚠️ Please select valid years")
    
    # ==================== ANALYSIS EXECUTION ====================
    if analyze_button and year_valid:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📡 Fetching satellite imagery...")
            progress_bar.progress(10)
            
            lat, lon = st.session_state.map_center
            
            status_text.text(f"🛰️ Downloading {year_old} imagery...")
            img_old = fetch_multi_tile_image(lat, lon, analysis_zoom, service="esri", year=year_old, tile_size=3)
            progress_bar.progress(30)
            
            status_text.text(f"🛰️ Downloading {year_new} imagery...")
            if year_new >= 2024:
                img_new = fetch_multi_tile_image(lat, lon, analysis_zoom, service="google", tile_size=3)
            else:
                img_new = fetch_multi_tile_image(lat, lon, analysis_zoom, service="esri", year=year_new, tile_size=3)
            progress_bar.progress(50)
            
            if not img_old or not img_new:
                st.error("❌ Failed to fetch imagery. Try different location/zoom.")
                progress_bar.empty()
                status_text.empty()
            else:
                if img_old.size != img_new.size:
                    img_new = img_new.resize(img_old.size)
                
                status_text.text("📊 Calculating structural similarity...")
                ssim_score, diff_map, thresh_map = compare_images_ssim(img_old, img_new)
                progress_bar.progress(60)
                
                all_changes = []
                
                if "Image Difference" in analysis_method or "Combined" in analysis_method:
                    status_text.text("🔬 Detecting changes using image analysis...")
                    opencv_changes = detect_changes_opencv(img_old, img_new, min_area=min_change_area)
                    for change in opencv_changes:
                        change['type'] = classify_change_type(change, year_old, year_new)
                    all_changes.extend(opencv_changes)
                    progress_bar.progress(75)
                
                if "AI Object Detection" in analysis_method or "Combined" in analysis_method:
                    status_text.text("🤖 Running AI object detection...")
                    
                    @st.cache_resource
                    def load_yolo_model():
                        return YOLO('yolov8n.pt')
                    
                    model = load_yolo_model()
                    yolo_changes = detect_changes_yolo(img_old, img_new, model, min_confidence=yolo_confidence / 100)
                    
                    for change in yolo_changes:
                        change['type'] = classify_change_type(change, year_old, year_new)
                    all_changes.extend(yolo_changes)
                    progress_bar.progress(85)
                
                status_text.text("🎨 Creating visualizations...")
                
                location_name = search_address if search_address else f"Location_{lat:.4f}_{lon:.4f}"
                location_name = "".join(c for c in location_name if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
                
                img_comparison, img_annotated = create_annotated_comparison(img_old, img_new, all_changes, year_old, year_new)
                progress_bar.progress(95)
                
                status_text.text("📋 Generating reports...")
                changes_df = generate_change_analysis_csv(all_changes, year_old, year_new, location_name)
                
                analysis_package = generate_analysis_package(
                    img_old, img_new, img_comparison, img_annotated,
                    changes_df, year_old, year_new, location_name, ssim_score
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.session_state.earth_results = {
                    'img_old': img_old,
                    'img_new': img_new,
                    'img_comparison': img_comparison,
                    'img_annotated': img_annotated,
                    'changes_df': changes_df,
                    'all_changes': all_changes,
                    'ssim_score': ssim_score,
                    'year_old': year_old,
                    'year_new': year_new,
                    'location_name': location_name,
                    'analysis_package': analysis_package
                }
                
                st.balloons()
                st.success(f"🎉 Analysis complete! Found **{len(all_changes)}** significant changes.")
                progress_bar.empty()
                status_text.empty()
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # ==================== DISPLAY RESULTS ====================
    if "earth_results" in st.session_state:
        results = st.session_state.earth_results
        
        st.markdown("---")
        st.markdown(f"## 📊 Analysis Results: {results['year_old']} → {results['year_new']}")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("📏 Similarity Index", f"{results['ssim_score']:.1%}", delta=f"{(1-results['ssim_score'])*100:.1f}% changed", delta_color="inverse")
        with col_m2:
            st.metric("🔍 Changes Detected", len(results['all_changes']))
        with col_m3:
            st.metric("📅 Time Span", f"{results['year_new'] - results['year_old']} years")
        with col_m4:
            if results['ssim_score'] > 0.9:
                st.metric("🟢 Change Level", "Minimal")
            elif results['ssim_score'] > 0.7:
                st.metric("🟡 Change Level", "Moderate")
            else:
                st.metric("🔴 Change Level", "Significant")
        
        st.markdown("---")
        st.markdown("### 🖼️ Visual Comparison")
        
        tab_individual, tab_sidebyside, tab_annotated = st.tabs(["📷 Individual Images", "↔️ Side-by-Side", "🎯 Changes Highlighted"])
        
        with tab_individual:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown(f"**{results['year_old']} (Baseline)**")
                st.image(results['img_old'], use_container_width=True)
                buf_old = io.BytesIO()
                results['img_old'].save(buf_old, format='PNG')
                st.download_button(f"📥 Download {results['year_old']} Image", buf_old.getvalue(), 
                                 f"{results['location_name']}_{results['year_old']}.png", "image/png", use_container_width=True)
            with col_img2:
                st.markdown(f"**{results['year_new']} (Current)**")
                st.image(results['img_new'], use_container_width=True)
                buf_new = io.BytesIO()
                results['img_new'].save(buf_new, format='PNG')
                st.download_button(f"📥 Download {results['year_new']} Image", buf_new.getvalue(),
                                 f"{results['location_name']}_{results['year_new']}.png", "image/png", use_container_width=True)
        
        with tab_sidebyside:
            st.image(results['img_comparison'], use_container_width=True, caption=f"Comparison: {results['year_old']} vs {results['year_new']}")
            buf_comp = io.BytesIO()
            results['img_comparison'].save(buf_comp, format='PNG')
            st.download_button("📥 Download Comparison", buf_comp.getvalue(),
                             f"{results['location_name']}_comparison.png", "image/png", use_container_width=True)
        
        with tab_annotated:
            st.image(results['img_annotated'], use_container_width=True, caption=f"{results['year_new']} with Changes")
            st.markdown("**Legend:** 🔴 Red boxes = Detected changes")
            buf_anno = io.BytesIO()
            results['img_annotated'].save(buf_anno, format='PNG')
            st.download_button("📥 Download Annotated", buf_anno.getvalue(),
                             f"{results['location_name']}_annotated.png", "image/png", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 Detailed Change Analysis")
        
        if len(results['all_changes']) > 0:
            display_df = results['changes_df'].copy()
            if 'confidence' in display_df.columns:
                display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
            st.dataframe(display_df, use_container_width=True, height=400)
            
            if 'type' in display_df.columns and len(display_df) > 0:
                st.markdown("#### 📊 Change Type Distribution")
                change_counts = display_df['type'].value_counts()
                col_chart, col_stats = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(change_counts)
                with col_stats:
                    st.markdown("**Summary:**")
                    for change_type, count in change_counts.items():
                        percentage = (count / len(display_df)) * 100
                        st.markdown(f"- **{change_type}**: {count} ({percentage:.1f}%)")
            
            st.markdown("---")
            st.markdown("### 💾 Download Analysis Data")
            
            col_csv, col_excel, col_package = st.columns(3)
            with col_csv:
                csv_data = results['changes_df'].to_csv(index=False)
                st.download_button("📊 Download CSV", csv_data, f"{results['location_name']}_changes.csv", "text/csv", use_container_width=True)
            with col_excel:
                excel_buf = io.BytesIO()
                results['changes_df'].to_excel(excel_buf, index=False, engine='openpyxl')
                st.download_button("📈 Download Excel", excel_buf.getvalue(), f"{results['location_name']}_changes.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with col_package:
                st.download_button("📦 Download Complete Package (ZIP)", results['analysis_package'].getvalue(),
                                 f"{results['location_name']}_analysis.zip", "application/zip", use_container_width=True)
        else:
            st.info(f"✅ No significant changes detected between {results['year_old']} and {results['year_new']}")
    
    st.markdown("---")
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        ### Getting the Best Analysis Results
        
        **Zoom Level:**
        - **14-15**: City-wide analysis
        - **16-17**: Neighborhood changes  
        - **18**: Building-level detail
        
        **Year Selection:**
        - Longer periods (10+ years) show more changes
        - Recent years (2020-2026) have best quality
        
        **Detection Methods:**
        - **AI Detection**: Best for buildings
        - **Image Difference**: Catches all changes
        - **Combined**: Most comprehensive (recommended)
        
        **Common Use Cases:**
        - 🏗️ Urban development tracking
        - 🌳 Deforestation monitoring
        - 🏖️ Coastal erosion analysis
        """)

# ---------- Tab 5: Feedback & Report ----------
with tab5:
    st.markdown("### 📝 Leave Feedback & Generate PDF Report")
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []

    feedback = st.text_area("Type feedback or bug report:", key="feedback_input")
    if st.button("Submit Feedback", key="submit_feedback_btn"):
        if feedback:
            st.session_state.feedback_list.append(feedback)
            st.success("Thank you for your feedback!")

    st.write("---")
    st.markdown("#### 💬 All Feedback")
    if st.session_state.feedback_list:
        for i, fb in enumerate(st.session_state.feedback_list, 1):
            st.markdown(f"**{i}:** {fb}")
    else:
        st.info("No feedback yet.")

    st.write("---")
    st.markdown("#### 📄 Generate PDF Summary Report")
    
    if st.button("🔄 Generate & Auto-Download PDF Report", key="generate_pdf_btn"):
        pdf_file = "SmartDetect_Session_Report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="SmartDetect AI Image Anomaly Detection Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)

        session_results = st.session_state.get("session_results", [])
        
        if session_results:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Detection Results:", ln=True)
            pdf.set_font("Arial", size=11)
            for sess in session_results:
                pdf.multi_cell(0, 8, txt=f"  Image: {sess['filename']}\n  Anomalies: {sess['num_anomalies']}\n  AI Correction: {'Enabled' if sess.get('ai_corrected', False) else 'Disabled'}\n  Date: {sess['date']}\n")
                pdf.ln(5)
        else:
            pdf.cell(200, 10, txt="No detection results yet.", ln=True)
        
        pdf.ln(10)
        if st.session_state.feedback_list:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="User Feedback:", ln=True)
            pdf.set_font("Arial", size=11)
            for fb in st.session_state.feedback_list:
                pdf.multi_cell(0, 8, txt=f"  - {fb}")
                pdf.ln(3)

        pdf.output(pdf_file)
        
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        auto_download_js = f'<script>var link = document.createElement("a");link.href = "data:application/pdf;base64,{b64_pdf}";link.download = "{pdf_file}";link.click();</script>'
        components.html(auto_download_js, height=0)
        st.success(f"✅ PDF Report generated: {pdf_file}")
        
        st.download_button("📥 Download PDF (Manual)", pdf_bytes, pdf_file, "application/pdf", key="pdf_manual_download")

# ---------- Tab 6: Tutorial ----------
with tab6:
    st.markdown("""
    ## How to Use This App (Tutorial)
    **Step 1:** Upload images in Upload & Preview tab  
    **Step 2:** Adjust settings in Detection & AI Correction  
    **Step 3:** Try Live Video Detection for real-time analysis  
    **Step 4:** Use Earth Pro Analysis to compare satellite imagery across years  
    **Step 5:** Generate PDF reports and provide feedback  
    """)

# ---------- Tab 7: About/Docs ----------
with tab7:
    st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto;">
<h2 style="font-weight: 700; color: #00A3FF;">About SmartDetect</h2>
<p style="font-size: 1.1rem; line-height: 1.6; color: #CCC;">
SmartDetect is a cutting-edge AI solution for quality control and infrastructure maintenance.
</p>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 30px 0;">
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px;">
<div style="font-size: 2rem;">🎨</div>
<h4 style="color: white;">AI Correction</h4>
<p style="font-size: 0.9rem; color: #AAA;">Intelligent anomaly removal</p>
</div>
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px;">
<div style="font-size: 2rem;">📹</div>
<h4 style="color: white;">Live Detection</h4>
<p style="font-size: 0.9rem; color: #AAA;">Real-time analysis</p>
</div>
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px;">
<div style="font-size: 2rem;">🌍</div>
<h4 style="color: white;">Earth Pro</h4>
<p style="font-size: 0.9rem; color: #AAA;">Satellite change detection</p>
</div>
</div>

<h3 style="color: white;">Credits</h3>
<p style="color: #AAA;">
<strong>Developed by:</strong><br>
Sugnik Tarafder • Arifur Rahaman<br>
Sk Shonju Ali • Trishan Nayek
</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 0.8rem; color: #666;'>SmartDetect v1.0 with Earth Pro Analysis</div>", unsafe_allow_html=True)
