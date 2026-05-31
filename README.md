<div align="center">
  <h1>⬡ SmartDetect</h1>
  <p><strong>Next-Generation AI Anomaly & Geographic Change Detection System</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg)](https://opencv.org/)
  [![Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-00E5FF.svg)](https://deepmind.google/technologies/gemini/)
</div>

<br/>

## 🚀 Overview

**SmartDetect** is a state-of-the-art computer vision platform designed to bridge the gap between raw pixel data and actionable intelligence. Built natively with Python and Streamlit, it leverages the blazing-fast reasoning of the **Google Gemini API** alongside traditional high-performance **OpenCV** algorithms to provide real-time spatial, structural, and geographic analysis.

Whether you're inspecting infrastructure for micro-cracks, tracking urban redevelopment from satellite imagery, or running a live security feed, SmartDetect provides the insights you need instantly.

---

## 🎯 Key Features

- **🖼️ Image & Structural Defect Analysis**  
  Upload high-resolution photos of materials, roads, or infrastructure. The system automatically detects cracks, dents, and anomalies, drawing bounding boxes and assigning a calculated **Risk Score (0-100)**.
- **🌍 Geographic Change Detection**  
  Compare "Before" and "After" satellite or drone imagery. Our pipeline uses ORB feature-based alignment, SSIM, and CLAHE to map precise structural shifts (e.g., new buildings, removed structures) and categorizes them with an AI-generated explanation.
- **🔴 Live CCTV Simulation**  
  Connects directly to your webcam for real-time motion detection and human tracking, processing up to **60 FPS**.
- **🤖 Intelligent AI Chat**  
  Got questions about an anomaly? The integrated SmartDetect AI (powered by Gemini) can comprehensively explain detection results, risk methodologies, and computer vision strategies.
- **📋 Persistent History Logging**  
  Automatically logs every analysis run into a searchable, exportable JSON database so you never lose an insight.

---

## 🛠️ Core Technologies

- **Frontend & Routing:** [Streamlit](https://streamlit.io/)
- **Computer Vision:** [OpenCV](https://opencv.org/) & `skimage`
- **AI / LLM / VLM Backend:** [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) (via `google-genai`)
- **Data Handling:** `numpy`, `pandas`, `pillow`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SmartDetect.git
cd SmartDetect
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API
You must have a valid Google Gemini API key. Add it to `gemini_helper.py` or export it as an environment variable (recommended for production):
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```
*The web interface will automatically open at `http://localhost:8501`.*

---

## 👨‍💻 Built By

This project was architected, developed, and designed by the **SmartDetect Team**:
- ⬡ **Sugnik Tarafder**
- ⬡ **Arifur Rahman**
- ⬡ **Sk Shonju Ali**
- ⬡ **Trishan Nayek**

---

<div align="center">
  <i>"Bridging the gap between pixels and intelligence."</i>
</div>
