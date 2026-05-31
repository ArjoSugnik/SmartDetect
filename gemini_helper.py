"""
gemini_helper.py — Integration with Google Gemini API
Replaces local Ollama with cloud-based Gemini for text, vision, and structural analysis.
"""

from google import genai
from google.genai import types
import json
import base64
import cv2
import numpy as np
from PIL import Image

import os

# Fetch the API key dynamically to avoid exposing it on GitHub.
# Streamlit Community Cloud automatically injects keys from its secrets manager into environment variables,
# but we also check st.secrets and local environment variables for maximum compatibility.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

try:
    # Initialize the client. The SDK will also automatically look for GEMINI_API_KEY in the environment.
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        client = genai.Client()
except Exception as e:
    client = None

MODEL = "gemini-2.5-flash"


# ── Health check ───────────────────────────────────────────────────────────────

def check_gemini_status() -> dict:
    """
    Check if Gemini client is initialized successfully.
    """
    if client is not None:
        return {"online": True, "models": [MODEL]}
    return {"online": False, "models": []}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_explanation(result: dict, score: int, risk_level: str) -> str:
    """
    Generate an AI explanation for a detected anomaly using Gemini.
    """
    if not client:
        return _fallback_response("explain")

    system_instruction = (
        "You are an expert computer vision analyst. "
        "Provide detailed, comprehensive, and professional explanations of image anomaly detection results. "
        "CRITICAL RULES: \n"
        "- DO NOT write any introduction or preamble (e.g., 'As an expert...', 'Here is the analysis...').\n"
        "- DO NOT refer to yourself.\n"
        "- Jump straight into the analysis using clear bullet points.\n"
        "- Aim for 300-500 words to give a thorough breakdown of potential causes and impacts.\n"
        "- End with a clear, multi-step recommended action plan."
    )

    context = (
        f"Anomaly detection result:\n"
        f"- Type: {result.get('anomaly_type', 'Unknown')}\n"
        f"- Risk Score: {score}/100 ({risk_level} risk)\n"
        f"- Contours found: {result.get('contour_count', 0)}\n"
        f"- Anomalous area: {result.get('anomaly_area_pct', 0):.2f}%\n"
        f"- Edge density: {result.get('edge_density', 0):.4f}\n"
        f"- Brightness std dev: {result.get('brightness_std', 0):.1f}\n\n"
        "Provide a comprehensive analysis explaining what this anomaly may indicate in the real world, the potential severity, and why."
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=1000
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)[:120]}"


def chat_with_assistant(user_message: str) -> str:
    """
    Chat-style interaction with the Gemini assistant.
    """
    if not client:
        return _fallback_response(user_message)

    system_instruction = (
        "You are SmartDetect, an intelligent assistant specialising in "
        "computer vision, anomaly detection, and remote sensing / GIS change detection. "
        "If anyone asks who built or created you, you must say that you were developed by the team of: "
        "Sugnik Tarafder, Arifur Rahman, Sk Shonju Ali, and Trishan Nayek. Do NOT say you were made by Google. "
        "Answer questions clearly, concisely, and helpfully. "
        "When asked about risk levels or scores, explain the 0-100 scale. "
        "Keep responses under 250 words unless a detailed explanation is needed."
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.65,
                max_output_tokens=1000
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini chat error: {str(e)[:120]}"


def analyze_image_structural(image_bgr: np.ndarray, anomaly_type: str) -> dict:
    """
    Perform structural anomaly detection using Gemini Vision.
    """
    if not client:
        return {
            "error": "Gemini client not initialized",
            "detected": False,
            "risk_score": 0,
            "description": "Failed to connect to Gemini API.",
            "recommendation": "Check API key.",
            "bboxes": []
        }

    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

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

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        response_text = response.text
        
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
        return {
            "error": str(e),
            "detected": False,
            "risk_score": 0,
            "description": f"Failed to connect to Gemini vision model: {e}",
            "recommendation": "Check Gemini API integration.",
            "bboxes": []
        }


# ── Intelligent fallback (offline mode) ───────────────────────────────────────

def _fallback_response(prompt: str) -> str:
    """
    Rule-based fallback response if Gemini fails.
    """
    prompt_lower = prompt.lower()

    if any(k in prompt_lower for k in ["explain", "anomaly", "what", "indicate"]):
        return (
            "🔌 <em>Gemini API offline — showing built-in response</em><br/><br/>"
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
            "🔌 <em>Gemini API offline — built-in response</em><br/><br/>"
            "<b>Risk Score System (0–100):</b><br/>"
            "• <b>0–29 LOW:</b> Normal or near-normal conditions. Monitor periodically.<br/>"
            "• <b>30–64 MEDIUM:</b> Notable anomalies. Manual inspection recommended.<br/>"
            "• <b>65–100 HIGH:</b> Significant irregularities. Immediate review required.<br/>"
            "<br/>Scores are computed from contour count, anomalous area, edge density, "
            "and brightness variance."
        )

    if "ssim" in prompt_lower:
        return (
            "🔌 <em>Gemini API offline — built-in response</em><br/><br/>"
            "<b>SSIM (Structural Similarity Index Measure):</b><br/>"
            "• Compares two images across luminance, contrast, and structure.<br/>"
            "• Range: -1 to 1, where 1 = identical images.<br/>"
            "• Scores below 0.3 indicate the images are too dissimilar for geo comparison.<br/>"
            "• Scores 0.3–0.7 indicate moderate similarity with potential changes.<br/>"
            "• Scores above 0.7 indicate high structural similarity."
        )

    if any(k in prompt_lower for k in ["geo", "satellite", "change"]):
        return (
            "🔌 <em>Gemini API offline — built-in response</em><br/><br/>"
            "<b>Geo Change Detection Tips:</b><br/>"
            "• Use images of the same region from different time periods.<br/>"
            "• Ensure similar lighting/weather conditions for best accuracy.<br/>"
            "• Align images to the same resolution before comparison.<br/>"
            "• Change categories: New Building, Road Change, Removed Area, Vegetation Loss.<br/>"
            "• SSIM > 0.3 required for valid comparison."
        )

    # Generic fallback
    return (
        "🔌 <em>Gemini API is unavailable. Please check your API key.</em><br/><br/>"
        "I can answer questions about:<br/>"
        "• Anomaly detection methodology<br/>"
        "• Risk scores and interpretation<br/>"
        "• Geo change detection and SSIM<br/>"
        "• OpenCV image processing techniques"
    )
