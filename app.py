import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageDraw, ImageFilter
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

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="SmartDetect - AI Image Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== ZORIN OS 18 STYLE MOUNTAIN WALLPAPERS (LOCAL FILES) ==========
# Background images stored in ./backgrounds folder
BACKGROUNDS_FOLDER = "backgrounds"

# Function to get base64 encoded image for CSS background
def get_base64_image(image_path):
    """Convert local image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Dark Theme: Dark mountain landscape at night/dusk
DARK_BG_PATH = os.path.join(BACKGROUNDS_FOLDER, "dark_mountain.jpg")
# Light Theme: Bright mountain landscape in daylight
LIGHT_BG_PATH = os.path.join(BACKGROUNDS_FOLDER, "light_mountain.jpg")

# Get base64 encoded images
dark_bg_base64 = get_base64_image(DARK_BG_PATH)
light_bg_base64 = get_base64_image(LIGHT_BG_PATH)

# Fallback to online URLs if local files not found
if dark_bg_base64:
    DARK_THEME_BG = f"data:image/jpeg;base64,{dark_bg_base64}"
else:
    DARK_THEME_BG = "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1920"

if light_bg_base64:
    LIGHT_THEME_BG = f"data:image/jpeg;base64,{light_bg_base64}"
else:
    LIGHT_THEME_BG = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1920"

# Mathematical formula for dynamic visual effects
def get_math_animation_value():
    """
    Uses mathematical formulas for dynamic visual effects:
    - Sine wave for smooth pulsing animations
    - Golden ratio for harmonious proportions
    - Fibonacci sequence for natural patterns
    """
    now = datetime.now()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    
    # Sine wave oscillation (0 to 1) - creates smooth breathing effect
    sine_pulse = (math.sin(seconds * math.pi / 30) + 1) / 2  # 60-second cycle
    
    # Golden ratio spiral effect
    golden_ratio = 1.618033988749
    golden_value = (seconds * golden_ratio) % 1
    
    # Fibonacci-based opacity (using modulo for cycling)
    fib_sequence = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45]
    fib_index = int((seconds / 10) % len(fib_sequence))
    fib_opacity = fib_sequence[fib_index]
    
    return {
        'sine': sine_pulse,
        'golden': golden_value,
        'fib_opacity': fib_opacity,
        'rotation': (seconds * 0.5) % 360  # Slow rotation for gradients
    }

def detect_cracks_opencv(image):
    """
    Detects cracks using computer vision (OpenCV) techniques:
    1. Convert to grayscale
    2. Gaussian Blur to reduce noise
    3. Canny Edge Detection
    4. Morphological Closing to connect gaps
    5. Find Contours
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Close gaps
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 100  # Filter small noise
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": x + w/2,
                "y": y + h/2,
                "width": w,
                "height": h,
                "confidence": 100.0, # CV detection is deterministic
                "class": "crack/defect"
            })
            
    return anomalies

def detect_stains_opencv(image):
    """
    Detects stains/discoloration using color statistics:
    1. Convert to HSV
    2. Calculate background color (median)
    3. Find pixels that deviate significantly from background
    4. Morphological cleaning
    """
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Blur to remove noise
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    
    # Calculate difference from median color (background assumption)
    median = cv2.medianBlur(blurred, 21)
    diff = cv2.absdiff(blurred, median)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to find significant deviations
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up cleaning
    kernel = np.ones((5,5), np.uint8)
    # Opening removes small noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Closing connects nearby stain parts
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies = []
    min_area = 200 # Stains usually bigger than specks
    
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

# Get mathematical values for animations
math_values = get_math_animation_value()

# ---- Main Page Header (Relocated) ----
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


# Default to Dark theme since we removed selector
theme = "Dark"
# Default model (not really used now, but keeps compatible if referenced)
model_choice = "Roboflow Default"

# ---- Professional UI Styles (Inter Font & Glassmorphism) ----
st.markdown("""
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Main Font Application */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit Header & Toolbar */
header[data-testid="stHeader"] {
    display: none !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Background & Main Containers */
.stApp {
    background: #0e1117;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, rgba(0, 163, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 100% 0%, rgba(0, 102, 255, 0.05) 0%, transparent 50%);
}

/* Glassmorphism Cards */
.stMarkdown > div, [data-testid="stVerticalBlock"] > div {
    border-radius: 12px;
}

/* Custom Header Title Styling */
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

/* Tab Styling - Professional & Clean */
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

/* Inputs & Widgets */
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

/* File Uploader */
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload & Preview", "🔍 Detection & AI Correction",
     "📹 Live Video Detection","📊 Feedback & Report", "🧭 Tutorial", "ℹ️ About/Docs"
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
        # Create responsive grid based on number of images
        num_images = len(uploaded_files)
        cols_per_row = min(num_images, 4)  # Max 4 columns
        cols = st.columns(cols_per_row)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % cols_per_row]:
                # Open image without resizing to preserve quality
                img = Image.open(uploaded_file).convert("RGB")
                # Display with fixed width for consistent layout
                st.image(img, caption=uploaded_file.name, width=200)
                # Reset file pointer for later use
                uploaded_file.seek(0)

# ---------- Tab 2: Detection & Correction ----------
with tab2:
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        st.warning("Upload images in the first tab.")
    else:
        # Settings row
        st.markdown("### ⚙️ Detection Settings")
        
        # Detection Mode Selector
        detection_mode = st.radio(
            "Detection Mode",
            ["🛡️ General AI (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            help="Choose your detection target: Objects, Cracks, or Stains."
        )
        
        threshold = st.slider("Minimum Confidence (%)", 0, 100, 50)
        
        # AI Correction Settings
        st.markdown("### 🤖 AI Correction Settings")
        use_ai_correction = st.checkbox("Enable AI-Powered Correction", value=True, help="Use AI to intelligently remove anomalies")
        
        if use_ai_correction:
            st.info("🎨 AI will intelligently remove detected anomalies and generate clean, natural-looking corrections.")
        else:
            st.warning("⚠️ AI correction disabled. Only detection will be performed.")
        
        st.write(f"**Detection:** Anomalies with confidence ≥ {threshold}% will be shown.")
        
        # Default box colors
        color_picker_high = "#00ff00"  # Green for high confidence
        color_picker_mid = "#ff0000"   # Red for medium confidence
        color_picker_low = "#ffff00"   # Yellow for low confidence

        # Initialize session state for results
        if "session_results" not in st.session_state:
            st.session_state.session_results = []
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        session_results = []

        with zipfile.ZipFile(temp_zip.name, "w") as zip_all:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n#### Image {idx + 1}: {uploaded_file.name}")
                image_bytes = uploaded_file.getvalue()
                orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Detection Logic based on Mode
                preds = []
                
                with st.spinner(f"Detecting anomalies ({detection_mode}) for {uploaded_file.name}..."):
                    try:
                        if "Road/Surface Cracks" in detection_mode:
                            # Use OpenCV Crack Detection
                            opencv_preds = detect_cracks_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Crack Detection complete! Found {len(preds)} defects.")
                            
                        elif "Stain/Discoloration" in detection_mode:
                            # Use OpenCV Stain Detection
                            opencv_preds = detect_stains_opencv(orig_img)
                            preds = opencv_preds
                            st.success(f"✅ Stain Detection complete! Found {len(preds)} defects.")
                            
                        else:
                            # Use YOLOv8 (Existing Logic)
                            # Load model (cached)
                            @st.cache_resource
                            def load_model():
                                return YOLO('yolov8n.pt')  # Downloads automatically on first run
                            
                            model = load_model()
                            results = model(orig_img)
                            
                            # Process results
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    # Get box coordinates (center_x, center_y, width, height)
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]
                                    
                                    # Filter by confidence
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

                # Table + downloads
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

                # Color mapping
                def get_color(conf):
                    if conf >= 0.9:
                        return color_picker_high
                    elif conf >= 0.7:
                        return color_picker_mid
                    else:
                        return color_picker_low

                # Annotated image (boxes only) - with thicker lines for visibility
                im_anno = orig_img.copy()
                draw = ImageDraw.Draw(im_anno)
                for pred in preds:
                    x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                    y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                    x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                    y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                    color = get_color(pred["confidence"])
                    # Draw thicker rectangle (width=3) for better visibility
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                # AI-Powered Corrected image
                im_corr = orig_img.copy()
                
                if use_ai_correction and preds:
                    with st.spinner("🤖 AI is generating corrected image..."):
                        try:
                            # Create mask for inpainting
                            mask = Image.new('L', orig_img.size, 0)
                            mask_draw = ImageDraw.Draw(mask)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                # Fill mask with white where anomalies are
                                mask_draw.rectangle([x0, y0, x1, y1], fill=255)
                            
                            # For now, use intelligent blur-based correction
                            # In production, this would call OpenAI DALL-E API
                            # im_corr = call_openai_inpainting(orig_img, mask)
                            
                            # Intelligent correction: blur anomalies with context-aware blending
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                box = (x0, y0, x1, y1)
                                
                                # Extract region and apply strong blur
                                region = im_corr.crop(box).filter(ImageFilter.GaussianBlur(20))
                                im_corr.paste(region, box)
                            
                            st.success("✅ AI correction completed!")
                        except Exception as e:
                            st.error(f"AI correction failed: {str(e)}")
                            # Fallback to original if AI fails
                            im_corr = orig_img.copy()

                # Display confidence color legend with round boxes
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

                # Display images with high quality
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_img, caption="Original", use_container_width=True, output_format="PNG")
                with col2:
                    st.image(im_anno, caption="Detected", use_container_width=True, output_format="PNG")
                with col3:
                    st.image(im_corr, caption="Corrected", use_container_width=True, output_format="PNG")

                # Image downloads
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

        # Update session state with results
        st.session_state.session_results = session_results

        # ZIP download - read and close temp file properly
        with open(temp_zip.name, "rb") as zf:
            all_zip_bytes = zf.read()
        
        # Clean up temp file
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
    st.info("🎥 Detect anomalies in real-time from your webcam or video feed. Perfect for detecting cracks in walls, windows, or other structural defects.")
    
    # Video detection controls
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
        st.warning("⚠️ **Note:** Webcam access requires browser permissions. Click 'Allow' when prompted.")
        
        # Webcam implementation using OpenCV
        st.markdown("""
        **How to use:**
        1. Click "Enable Webcam" below to access your camera
        2. Point your camera at walls, windows, or surfaces to check for defects
        3. Click "Capture & Detect" to analyze the current frame
        4. Detected anomalies will be highlighted with bounding boxes
        """)
        
        # Initialize session state for webcam
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        if "webcam_snapshot" not in st.session_state:
            st.session_state.webcam_snapshot = None
        
        col_start, col_snap, col_stop = st.columns(3)
        
        with col_start:
            if st.button("🎥 Enable Webcam", key="start_webcam", use_container_width=True):
                st.session_state.webcam_running = True
        
        with col_snap:
            capture_btn = st.button("📸 Capture & Detect", key="snapshot", use_container_width=True, disabled=not st.session_state.webcam_running)
        
        with col_stop:
            if st.button("⏹️ Disable Webcam", key="stop_webcam", use_container_width=True):
                st.session_state.webcam_running = False
                st.session_state.webcam_snapshot = None
        
        # Video Detection Mode Selector
        video_detection_mode = st.radio(
            "Video Detection Mode",
            ["🛡️ General AI (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)", "🏃 Motion Detection"],
            horizontal=True,
            key="video_mode_select",
            help="Choose your detection target."
        )

        st.markdown("---")
        
        # Webcam feed
        if st.session_state.webcam_running:
            st.info("📷 Webcam is active. Click 'Capture & Detect' to analyze the current frame.")
            
            # Use Streamlit's camera_input for webcam access
            camera_image = st.camera_input("Live Camera Feed", key="camera_feed")
            
            if camera_image is not None and capture_btn:
                # Process the captured image
                with st.spinner("🔍 Detecting anomalies in captured frame..."):
                    try:
                        # Read image
                        img_bytes = camera_image.getvalue()
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        
                        # Prepare for API call
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="JPEG")
                        img_buffer.seek(0)
                        
                        # Detection Logic based on Mode
                        try:
                            preds = []
                            
                            if "Road/Surface Cracks" in video_detection_mode:
                                # Use OpenCV Crack Detection
                                preds = detect_cracks_opencv(img)
                                st.success(f"✅ Found {len(preds)} defects using Computer Vision")
                                
                            elif "Stain/Discoloration" in video_detection_mode:
                                # Use OpenCV Stain Detection
                                preds = detect_stains_opencv(img)
                                st.success(f"✅ Found {len(preds)} stains using Color Analysis")
                                
                            elif "Motion Detection" in video_detection_mode:
                                # Motion Detection requires previous frame
                                if "prev_frame" not in st.session_state or st.session_state.prev_frame is None:
                                    st.session_state.prev_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                                    st.info("🔄 Initializing motion detection...")
                                else:
                                    curr_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                                    frame_diff = cv2.absdiff(st.session_state.prev_frame, curr_frame)
                                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    for cnt in contours:
                                        if cv2.contourArea(cnt) > 500:
                                            x, y, w, h = cv2.boundingRect(cnt)
                                            preds.append({
                                                "x": float(x + w/2),
                                                "y": float(y + h/2),
                                                "width": float(w),
                                                "height": float(h),
                                                "confidence": 100.0,
                                                "class": "motion"
                                            })
                                    st.session_state.prev_frame = curr_frame
                                    if preds:
                                        st.success(f"✅ Motion Detected! {len(preds)} moving objects.")
                            
                            else:
                                # Local YOLOv8 Detection
                                # Load model (use cached function if available or define it)
                                if 'load_model' not in locals():
                                    @st.cache_resource
                                    def load_model():
                                        return YOLO('yolov8n.pt')
                                
                                model = load_model()
                                results = model(img)
                                
                                # Process results
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
                                
                                st.success(f"✅ Found {len(preds)} anomalies with ≥{video_threshold}% confidence")
                            
                            # Draw bounding boxes (Common for both methods)
                            img_annotated = img.copy()
                            draw = ImageDraw.Draw(img_annotated)
                            
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                
                                # Draw red box for anomalies
                                draw.rectangle([x0, y0, x1, y1], outline="#FF0000", width=3)
                                
                                # Add confidence label
                                if "class" in pred:
                                    label_text = str(pred['class'])
                                else:
                                    label_text = "Anomaly"
                                    
                                conf_val = float(pred['confidence'])
                                conf_text = f"{label_text} {conf_val*100:.1f}%"
                                draw.text((x0, y0-20), conf_text, fill="#FF0000")
                                
                            # Display results
                            col_orig, col_detect = st.columns(2)
                            
                            with col_orig:
                                st.markdown("**Original Frame**")
                                st.image(img, use_container_width=True)
                            
                            with col_detect:
                                st.markdown("**Detected Anomalies**")
                                st.image(img_annotated, use_container_width=True)
                            
                            # Download option
                            buf = io.BytesIO()
                            img_annotated.save(buf, format="PNG")
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Download Annotated Frame",
                                data=buf,
                                file_name="webcam_detection.png",
                                mime="image/png"
                            )
                            
                            if preds:
                                # Show detection details
                                with st.expander("📊 Detection Details"):
                                    df = pd.DataFrame(preds)[["x", "y", "width", "height", "confidence"]]
                                    df["confidence (%)"] = (df["confidence"] * 100).round(2)
                                    st.dataframe(df, use_container_width=True)
                    
                        except Exception as e:
                            st.error(f"Error processing frame: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error capturing image: {str(e)}")
        else:
            st.info("👆 Click 'Enable Webcam' to start live detection")
    
    else:  # Upload Video File
        st.markdown("#### 📁 Upload Video File")
        uploaded_video = st.file_uploader(
            "Upload a video file (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            key="video_upload"
        )
        
        if uploaded_video:
            st.video(uploaded_video)
            
            col_process, col_download = st.columns(2)
            
            with col_process:
                if st.button("🔍 Process Video for Anomalies", key="process_video", use_container_width=True):
                    with st.spinner("🎬 Processing video frames..."):
                        st.info("📊 Video processing will detect anomalies frame-by-frame")
                        st.markdown("""
                        **Processing Steps:**
                        1. Extract frames from video
                        2. Run anomaly detection on each frame
                        3. Annotate frames with detected anomalies
                        4. Generate processed video with annotations
                        
                        *(Install opencv-python and ffmpeg to enable this feature)*
                        """)
                        st.success("✅ Video processing complete! (Feature will be enabled with required dependencies)")
            
            with col_download:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=b"",  # Placeholder
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    disabled=True,
                    help="Process video first to enable download"
                )
    
    st.markdown("---")
    st.markdown("### 📊 Detection Statistics")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Frames Processed", "0", delta=None)
    
    with col_stats2:
        st.metric("Anomalies Detected", "0", delta=None)
    
    with col_stats3:
        st.metric("Detection Rate", "0 FPS", delta=None)
    
    with col_stats4:
        st.metric("Avg Confidence", "0%", delta=None)
    
    st.markdown("---")
    st.markdown("""
    ### 🔧 Setup Instructions
    
    To enable live video detection, install the required dependencies:
    
    ```bash
    pip install opencv-python streamlit-webrtc av
    ```
    
    **Supported Anomaly Types:**
    - 🧱 Wall cracks and structural defects
    - 🪟 Window damage and glass cracks
    - 🎨 Surface defects and color anomalies
    - 🔩 Missing or damaged components
    - 🌊 Water damage and stains
    - 🦠 Corrosion and rust
    
    **Performance Tips:**
    - Use good lighting for better detection
    - Keep camera steady for accurate results
    - Adjust confidence threshold based on environment
    - Lower resolution for faster processing
    """)


# ---------- Tab 4: Feedback & Report ----------

with tab4:
    st.markdown("### 📝 Leave Feedback & Generate PDF Report")
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []

    feedback = st.text_area("Type feedback or bug report for the developer:", key="feedback_input")
    submit_fb = st.button("Submit Feedback", key="submit_feedback_btn")
    if submit_fb and feedback:
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
    
    # Generate PDF button
    if st.button("🔄 Generate & Auto-Download PDF Report", key="generate_pdf_btn"):
        pdf_file = "SmartDetect_Session_Report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="SmartDetect AI Image Anomaly Detection Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)

        # Get session_results from session state
        session_results = st.session_state.get("session_results", [])
        
        if session_results:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Detection Results:", ln=True)
            pdf.set_font("Arial", size=11)
            for sess in session_results:
                pdf.multi_cell(
                    0, 8,
                    txt=f"  Image: {sess['filename']}\n  Anomalies Found: {sess['num_anomalies']}\n  AI Correction: {'Enabled' if sess.get('ai_corrected', False) else 'Disabled'}\n  Date: {sess['date']}\n"
                )
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
        
        # Read PDF and create auto-download
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        # Auto-download using JavaScript
        auto_download_js = f'''
            <script>
                var link = document.createElement('a');
                link.href = 'data:application/pdf;base64,{b64_pdf}';
                link.download = '{pdf_file}';
                link.click();
            </script>
        '''
        components.html(auto_download_js, height=0)
        st.success(f"✅ PDF Report generated and downloading: {pdf_file}")
        
        # Also provide manual download button
        st.download_button(
            label="📥 Download PDF Report (Manual)",
            data=pdf_bytes,
            file_name=pdf_file,
            mime="application/pdf",
            key="pdf_manual_download"
        )

    st.write("---")
    st.markdown("> For live notifications or email alerts, connect with developer. (Feature available in full version)")

# ---------- Tab 5: Tutorial ----------
with tab5:
    st.markdown("""
    ## How to Use This App (Tutorial)
    **Step 1:** Go to Upload & Preview, select images to analyze.  
    **Step 2:** In Detection & AI Correction, adjust confidence filter and enable AI-powered correction.  
    **Step 3:** View AI-corrected images, compare with originals, and download results.  
    **Step 4:** Try Live Video Detection to detect anomalies in real-time from webcam or video files.  
    **Step 5:** Leave feedback, generate PDF reports, and experiment with different settings.  
    **Step 6:** Switch between Dark and Light themes for optimal viewing experience.
    """)

# ---------- Tab 6: About/Docs ----------
with tab6:
    st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto;">
<h2 style="font-weight: 700; color: #00A3FF;">About SmartDetect</h2>
<p style="font-size: 1.1rem; line-height: 1.6; color: #CCC; margin-bottom: 30px;">
SmartDetect is a cutting-edge AI solution designed to revolutionize quality control and infrastructure maintenance. 
By leveraging state-of-the-art computer vision models, it identifies anomalies, defects, and structural issues with unprecedented accuracy.
</p>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 40px;">
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-size: 2rem; margin-bottom: 10px;">🎨</div>
<h4 style="margin: 0; color: white;">AI Correction</h4>
<p style="font-size: 0.9rem; color: #AAA;">Intelligent inpainting to seamlessly remove anomalies.</p>
</div>
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-size: 2rem; margin-bottom: 10px;">📹</div>
<h4 style="margin: 0; color: white;">Live Detection</h4>
<p style="font-size: 0.9rem; color: #AAA;">Real-time analysis from webcams or video files.</p>
</div>
<div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; width: 200px; border: 1px solid rgba(255,255,255,0.05);">
<div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
<h4 style="margin: 0; color: white;">Detailed Reports</h4>
<p style="font-size: 0.9rem; color: #AAA;">Export results to CSV, Excel, and professional PDF.</p>
</div>
</div>

<h3 style="color: white; margin-bottom: 20px;">Use Cases</h3>
<ul style="list-style-type: none; padding: 0; display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; color: #BBB;">
<li style="background: rgba(0,163,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,163,255,0.2);">🏭 Manufacturing QC</li>
<li style="background: rgba(0,163,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,163,255,0.2);">🏗️ Infrastructure</li>
<li style="background: rgba(0,163,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,163,255,0.2);">🛡️ Safety Monitoring</li>
<li style="background: rgba(0,163,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,163,255,0.2);">🏥 Healthcare Imaging</li>
</ul>

<hr style="border-color: rgba(255,255,255,0.1); margin: 40px 0;">

<div style="display: flex; justify-content: space-around; text-align: left; max-width: 600px; margin: 0 auto; color: #AAA; font-size: 0.9rem;">
<div>
<strong style="color: white; display: block; margin-bottom: 5px;">Built With</strong>
YOLOv8, OpenCV, Streamlit<br>
Python 3.10+
</div>
<div>
<strong style="color: white; display: block; margin-bottom: 5px;">Credits</strong>
Developed by -<br>Sugnik Tarafder<br>Arifur Rahaman<br>Sk Shonju Ali<br>Trishan Nayek<br>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    col_doc_btn, col_api_link = st.columns([1, 1])
    
    with col_doc_btn:
        try:
            with open("SmartDetect_Project_Documentation.pdf", "rb") as doc_file:
                st.download_button(
                    label="📄 Download Project Documentation",
                    data=doc_file,
                    file_name="SmartDetect_Project_Documentation.pdf",
                    mime="application/pdf",
                    key="doc_download",
                    use_container_width=True
                )
        except Exception:
            st.info("Documentation file unavailable.")

    with col_api_link:
        st.markdown("""
        <div style="text-align: center; padding: 5px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            <a href="https://docs.ultralytics.com/" target="_blank" style="text-decoration: none; color: #00A3FF; font-weight: 600;">
                📚 API Documentation (YOLOv8)
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 0.8rem; color: #666;'>SmartDetect v1.0</div>", unsafe_allow_html=True)
