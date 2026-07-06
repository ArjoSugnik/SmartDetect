"""
groq_helper.py — Integration with Groq API (Ultra-Fast Inference)
Replaces Gemini with Groq's Llama Vision & Chat models for text, vision, and structural analysis.
"""

from groq import Groq
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
import os

# Fetch the API key dynamically to avoid exposing it on GitHub.
# Streamlit Community Cloud automatically injects keys from its secrets manager into environment variables,
# but we also check st.secrets and local environment variables for maximum compatibility.
GROQ_API_KEY_ENV = os.environ.get("GROQ_API_KEY")

api_keys_raw = GROQ_API_KEY_ENV
if not api_keys_raw:
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            api_keys_raw = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

API_KEYS = []
if api_keys_raw:
    # Support comma-separated list of keys
    API_KEYS = [k.strip() for k in api_keys_raw.split(",") if k.strip()]

current_key_idx = 0
client = None

# ── Model configuration ───────────────────────────────────────────────────────
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # Best accuracy for image analysis
CHAT_MODEL   = "llama-3.3-70b-versatile"         # Best quality for chat


def _init_client():
    global client
    try:
        if API_KEYS:
            client = Groq(api_key=API_KEYS[current_key_idx])
        else:
            client = Groq()  # Falls back to GROQ_API_KEY env var
    except Exception:
        client = None

_init_client()


def _rotate_api_key():
    """Rotates to the next API key and re-initializes the client."""
    global current_key_idx
    if len(API_KEYS) > 1:
        current_key_idx = (current_key_idx + 1) % len(API_KEYS)
        _init_client()
        return True
    return False


def _image_to_base64(image_bgr: np.ndarray, max_size: int = 800) -> str:
    """Convert an OpenCV BGR image to a base64-encoded JPEG string, resized to max_size."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    # Resize to prevent exceeding Groq's 4MB limit for base64 images
    if pil_img.width > max_size or pil_img.height > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Health check ───────────────────────────────────────────────────────────────

def check_groq_status() -> dict:
    """
    Check if Groq client is initialized successfully.
    """
    if client is not None:
        return {"online": True, "models": [VISION_MODEL, CHAT_MODEL]}
    return {"online": False, "models": []}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_explanation(result: dict, score: int, risk_level: str, image_bgr: np.ndarray = None) -> str:
    """
    Generate an AI explanation for a detected anomaly using Groq Vision.
    """
    if not client:
        return _fallback_response("explain")

    system_instruction = (
        "You are an expert computer vision analyst. "
        "Provide detailed, comprehensive, and professional explanations of image anomaly detection results. "
        "CRITICAL RULES: \n"
        "- DO NOT write any introduction or preamble (e.g., 'As an expert...', 'Here is the analysis...').\n"
        "- DO NOT refer to yourself.\n"
        "- If an image is provided, rely primarily on what you visually see in the image to determine what the anomaly is (e.g. a road crack, a dent, etc.). Use the provided metric stats only as supporting context.\n"
        "- Jump straight into the analysis using clear bullet points.\n"
        "- Aim for 300-500 words to give a thorough breakdown of potential causes and impacts.\n"
        "- End with a clear, multi-step recommended action plan."
    )

    context = (
        f"Anomaly detection metrics from the system:\n"
        f"- Type: {result.get('anomaly_type', 'Unknown')}\n"
        f"- Risk Score: {score}/100 ({risk_level} risk)\n"
        f"- Contours found: {result.get('contour_count', 0)}\n"
        f"- Anomalous area: {result.get('anomaly_area_pct', 0):.2f}%\n"
        f"- Edge density: {result.get('edge_density', 0):.4f}\n"
        f"- Brightness std dev: {result.get('brightness_std', 0):.1f}\n\n"
        "Please look at the provided image (if any) and these metrics. Identify what the physical object is and what the actual defect is in the real world, explain its potential severity, and why."
    )

    # Build message content
    content = []
    if image_bgr is not None:
        b64_img = _image_to_base64(image_bgr)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })
    content.append({"type": "text", "text": context})

    # Choose model based on whether image is included
    model = VISION_MODEL if image_bgr is not None else CHAT_MODEL

    max_attempts = len(API_KEYS) if API_KEYS else 1
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower()) and _rotate_api_key():
                continue
            return f"⚠️ Groq API error: {err_str[:120]}"
    return "⚠️ Groq API error: All provided API keys have exceeded their rate limit."


def chat_with_assistant(user_message: str) -> str:
    """
    Chat-style interaction with the Groq assistant.
    """
    if not client:
        return _fallback_response(user_message)

    system_instruction = (
        "You are SmartDetect, an intelligent assistant specialising in "
        "computer vision, anomaly detection, and remote sensing / GIS change detection. "
        "If anyone asks who built or created you, you must say that you were developed by the team of: "
        "Sugnik Tarafder, Arifur Rahman, Sk Shonju Ali, and Trishan Nayek. Do NOT say you were made by Google or Meta. "
        "Answer questions clearly, concisely, and helpfully. "
        "When asked about risk levels or scores, explain the 0-100 scale. "
        "Keep responses under 250 words unless a detailed explanation is needed."
    )

    max_attempts = len(API_KEYS) if API_KEYS else 1
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.65,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower()) and _rotate_api_key():
                continue
            return f"⚠️ Groq chat error: {err_str[:120]}"
    return "⚠️ Groq chat error: All provided API keys have exceeded their rate limit."


def analyze_image_structural(image_bgr: np.ndarray, anomaly_type: str) -> dict:
    """
    Perform structural anomaly detection using Groq Vision.
    """
    if not client:
        return {
            "error": "Groq client not initialized",
            "detected": False,
            "risk_score": 0,
            "description": "Failed to connect to Groq API.",
            "recommendation": "Check API key.",
            "bboxes": []
        }

    b64_img = _image_to_base64(image_bgr)

    prompt = (
        f"TASK: Detect all instances of '{anomaly_type}' in this image.\n"
        "SPATIAL REASONING:\n"
        "- The image is a grid from 0 to 1000 (y is top-to-bottom, x is left-to-right).\n"
        "- Identify the exact boundaries [ymin, xmin, ymax, xmax] of each defect.\n"
        "- Focus ONLY on the physical surface of the object (e.g., the road, the wall, the material).\n"
        "- Ignore the background, sky, or trees.\n\n"
        "Return your findings strictly in this JSON format (with no markdown block ticks):\n"
        "{\n"
        "  \"detected\": boolean,\n"
        "  \"risk_score\": integer (0-100),\n"
        "  \"description\": \"detailed explanation\",\n"
        "  \"recommendation\": \"one-line action\",\n"
        "  \"bboxes\": [[ymin, xmin, ymax, xmax], ...] \n"
        "}\n\n"
        "Provide coordinates for EVERY significant area found."
    )

    max_attempts = len(API_KEYS) if API_KEYS else 1
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        },
                        {"type": "text", "text": prompt}
                    ]}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content

            # Strip any markdown fences the model might wrap around JSON
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            # Parse JSON from response
            try:
                result = json.loads(response_text)
            except Exception:
                result = {
                    "detected": "true" in response_text.lower(),
                    "risk_score": 50 if "true" in response_text.lower() else 0,
                    "description": response_text[:200],
                    "recommendation": "Manual review required.",
                    "bboxes": []
                }

            if "bboxes" not in result:
                result["bboxes"] = []

            return result
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower()) and _rotate_api_key():
                continue
            return {
                "error": err_str,
                "detected": False,
                "risk_score": 0,
                "description": f"Failed to connect to Groq vision model: {err_str[:120]}",
                "recommendation": "Check Groq API integration.",
                "bboxes": []
            }

    return {
        "error": "Rate limit exceeded on all keys",
        "detected": False,
        "risk_score": 0,
        "description": "All provided API keys have exceeded their rate limit.",
        "recommendation": "Provide additional valid API keys or wait for quota reset.",
        "bboxes": []
    }


# ── Intelligent fallback (offline mode) ───────────────────────────────────────

def _fallback_response(prompt: str) -> str:
    """
    Rule-based fallback response if Groq fails.
    """
    prompt_lower = prompt.lower()

    if any(k in prompt_lower for k in ["explain", "anomaly", "what", "indicate"]):
        return (
            "🔌 <em>Groq API offline — showing built-in response</em><br/><br/>"
            "<b>Anomaly Analysis:</b><br/>"
            "• The detected anomaly suggests irregular patterns in the image.<br/>"
            "• Elevated contour count may indicate multiple objects of interest.<br/>"
            "• High anomaly area percentage suggests widespread changes.<br/>"
            "• Edge density reflects the complexity of structural boundaries.<br/>"
            "<br/><b>Recommended Action:</b> Review highlighted regions manually "
            "and cross-reference with historical data."
        )

    if any(k in prompt_lower for k in ["risk", "score", "level"]):
        return (
            "🔌 <em>Groq API offline — built-in response</em><br/><br/>"
            "<b>Risk Score System (0–100):</b><br/>"
            "• <b>0–29 LOW:</b> Normal or near-normal conditions. Monitor periodically.<br/>"
            "• <b>30–64 MEDIUM:</b> Notable anomalies. Manual inspection recommended.<br/>"
            "• <b>65–100 HIGH:</b> Significant irregularities. Immediate review required.<br/>"
            "<br/>Scores are computed from contour count, anomalous area, edge density, "
            "and brightness variance."
        )

    if "ssim" in prompt_lower:
        return (
            "🔌 <em>Groq API offline — built-in response</em><br/><br/>"
            "<b>SSIM (Structural Similarity Index Measure):</b><br/>"
            "• Compares two images across luminance, contrast, and structure.<br/>"
            "• Range: -1 to 1, where 1 = identical images.<br/>"
            "• Scores below 0.3 indicate the images are too dissimilar for geo comparison.<br/>"
            "• Scores 0.3–0.7 indicate moderate similarity with potential changes.<br/>"
            "• Scores above 0.7 indicate high structural similarity."
        )

    if any(k in prompt_lower for k in ["geo", "satellite", "change"]):
        return (
            "🔌 <em>Groq API offline — built-in response</em><br/><br/>"
            "<b>Geo Change Detection Tips:</b><br/>"
            "• Use images of the same region from different time periods.<br/>"
            "• Ensure similar lighting/weather conditions for best accuracy.<br/>"
            "• Align images to the same resolution before comparison.<br/>"
            "• Change categories: New Building, Road Change, Removed Area, Vegetation Loss.<br/>"
            "• SSIM > 0.3 required for valid comparison."
        )

    # Generic fallback
    return (
        "🔌 <em>Groq API is unavailable. Please check your API key.</em><br/><br/>"
        "I can answer questions about:<br/>"
        "• Anomaly detection methodology<br/>"
        "• Risk scores and interpretation<br/>"
        "• Geo change detection and SSIM<br/>"
        "• OpenCV image processing techniques"
    )
