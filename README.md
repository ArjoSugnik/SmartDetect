<div align="center">
  <h1>⬡ SmartDetect</h1>
  <p><strong>Your Intelligent Assistant for Vision-Based Anomaly Detection & Correction</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg)](https://opencv.org/)
  [![Groq](https://img.shields.io/badge/Powered%20by-Groq%20AI-f55036.svg)](https://groq.com/)
</div>

<br/>

Welcome to **SmartDetect**! 👋

SmartDetect acts as a tireless set of eyes, using AI to scan images, live camera feeds, videos, and satellite maps to find anomalies that humans might miss. Powered by OpenCV and **Groq's ultra-fast Llama models**, it features a stunning **dark cyber-industrial UI** and lightning-fast inference.

> 🌐 **Live demo:** [https://smartdetect.streamlit.app](https://smartdetect.streamlit.app/)

---

## ✨ Features at a Glance

The sleek top navigation bar provides access to **8 core sections**:

| Tab | What it does |
|-----|--------------|
| 🏠 **Dashboard** | Landing overview with quick stats, system capabilities, and detection history summary. |
| 🖼️ **Image** | Upload a photo and scan for structural anomalies (cracks, scratches, dents). Draws bounding boxes + a 0–100 risk score, with an optional Groq AI explanation and a heatmap view. |
| 📷 **Camera** | Turns your webcam into a live CCTV — motion tracking + human/body detection in real time. |
| 🎬 **Video** | Upload a video (or run a live simulation). Extracts frames and charts a chronological **Risk Timeline** showing exactly when anomalies occur. |
| 🌍 **Geo Change** | Upload "before" and "after" satellite images. Uses SSIM + absolute-difference mapping to highlight deforestation, new construction, erosion, etc. |
| 🤖 **AI Chat** | Chat directly with the Groq-powered assistant about your results, risk scoring, or anything vision-related. |
| 📋 **History** | Every saved analysis is logged here for later review (exported to JSON). |
| ℹ️ **About** | Project info and credits. |

---

## 🚀 Step 1: Prerequisites

### 1. Install Python
SmartDetect needs Python **3.10 or higher**.
- **Windows / Mac:** Download from the [official Python website](https://www.python.org/downloads/).
- **⚠️ Windows users:** On the first installer screen, check **"Add Python to PATH"** before clicking Install.

### 2. Get the project code
```bash
git clone https://github.com/yourusername/smartdetect-anomaly-detection-correction.git
```
Or click **"Download ZIP"** on the repo page and extract it.

---

## 🔑 Step 2: Get Your API Key

SmartDetect uses Groq for ultra-fast AI vision and text inference.

1. Go to [GroqCloud](https://console.groq.com/).
2. Sign in → **API Keys** → **Create API Key**.
3. Copy the key and keep it safe.

---

## ⚙️ Step 3: Set Up the Project

1. **Open a terminal** and navigate to the project folder:
   ```bash
   cd smartdetect-anomaly-detection-correction
   ```
2. **Create & activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your secrets.** Create a file at `.streamlit/secrets.toml` and fill in your keys:
   ```toml
   # ---- AI ----
   GROQ_API_KEY = "paste_your_groq_key_here"
   ```
   > 🔒 **Never commit `secrets.toml` to a public repo.** Keep your keys private.
   > Alternatively, you can export `GROQ_API_KEY` as an environment variable, and the app can also support a comma-separated list of keys to manage rate limits automatically.

---

## 🎉 Step 4: Run the Application

With your `(venv)` active:
```bash
streamlit run app.py
```
The app opens in your browser at `http://localhost:8501`.

---

## ☁️ Step 5: Deploy to the Cloud (Streamlit Community Cloud)

Want a public link anyone can open? Deploy for free:

1. **Push your code to GitHub** (keep the repo **Private** if it contains secrets).
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Pick your repo, branch (`main`), and main file (`app.py`).
4. Open **Advanced settings → Secrets** and paste the same contents as your `secrets.toml`.
5. Click **Deploy**. Streamlit installs everything from `requirements.txt` and `packages.txt` automatically.

**Files that make cloud deployment work:**
- `requirements.txt` — Python packages (uses `opencv-contrib-python-headless` so it runs on a server with no display).
- `packages.txt` — system libraries (e.g., `libgl1`).

**To make it public:** In the app's **Settings → Sharing**, set it so anyone with the link can view.

---

## 🗂️ Project Structure

```
app.py                     # Main Streamlit app (UI, routing)
anomaly.py                 # Image anomaly detection + risk scoring
video_processing.py        # Frame processing, motion & human detection
geo_analysis.py            # SSIM-based geographical change detection
groq_helper.py             # Groq AI integration for vision & chat
utils.py                   # History, reports, image helpers
requirements.txt           # Python dependencies
packages.txt               # System libraries for cloud deploy
.streamlit/secrets.toml    # Your private keys (do NOT commit publicly)
```

---

## 🧑‍💻 Built By

Architected, developed, and designed as a Final Year Project by the **SmartDetect Team**:
- ⬡ **Sugnik Tarafder**
- ⬡ **Arifur Rahman**
- ⬡ **Sk Shonju Ali**
- ⬡ **Trishan Nayek**

*Enjoy exploring the future of computer vision!*
