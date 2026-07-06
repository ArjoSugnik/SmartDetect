"""
video_processing.py — Video frame extraction, anomaly analysis, and live simulation.
"""

import cv2
import numpy as np
import threading
import av
from typing import List, Tuple, Optional
from streamlit_webrtc import VideoProcessorBase
from anomaly import detect_image_anomaly, compute_risk_score
from utils import draw_anomaly_overlay

# Load Haar Cascades for human/face detection (CCTV style)
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml'))
BODY_CASCADE = cv2.CascadeClassifier(os.path.join(DATA_DIR, 'haarcascade_upperbody.xml'))


def create_background_subtractor():
    """Create a fresh background subtractor instance (avoids stale state across Streamlit reruns)."""
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)


def process_video_frames(video_path: str, max_frames: int = 20, fps_rate: float = 1.0) -> List[dict]:
    """
    Extract frames from a video at a configurable FPS rate and run anomaly detection on each.

    Args:
        video_path: Path to the video file (temp file).
        max_frames: Maximum number of frames to process.
        fps_rate:   Frames to extract per second of video (default 1.0).

    Returns:
        List of dicts with per-frame detection results including effective FPS used.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # Sample frames based on user-selected fps_rate
    sample_count = min(max_frames, max(1, int(duration_sec * fps_rate)))
    if sample_count == 0:
        sample_count = min(max_frames, total_frames)

    frame_indices = np.linspace(0, max(total_frames - 1, 0), sample_count, dtype=int)
    target_frames = set(int(x) for x in frame_indices)

    results = []
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_no in target_frames:
            timestamp_sec = frame_no / fps
            detection = detect_image_anomaly(frame)
            score, risk_level = compute_risk_score(detection)

            results.append(
                {
                    "frame": int(frame_no),
                    "timestamp": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                    "anomaly_type": detection["anomaly_type"],
                    "score": score,
                    "risk_level": risk_level,
                    "contours": detection["contour_count"],
                    "area_pct": round(detection["anomaly_area_pct"], 2),
                }
            )
            target_frames.remove(frame_no)
            if not target_frames:
                break
                
        frame_no += 1

    cap.release()
    return results


def process_camera_frame(cv_img: np.ndarray) -> dict:
    """
    Run anomaly detection on a single camera capture frame.
    Thin wrapper around detect_image_anomaly for semantic clarity.
    """
    return detect_image_anomaly(cv_img)


class SmartDetectVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor for browser-based live CCTV simulation.

    Runs motion detection (background subtraction), human detection (Haar cascades),
    and annotates frames with a CCTV-style HUD — all inside the recv() callback
    which executes on every incoming WebRTC video frame.

    Thread-safe properties (result, score, risk_level) allow the Streamlit main
    thread to read the latest detection metrics for display alongside the video.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._back_sub = create_background_subtractor()
        self._result: dict = {
            "anomaly_type": "Initializing...",
            "contour_count": 0,
            "anomaly_area_pct": 0.0,
            "human_detected": False,
            "face_count": 0,
            "body_count": 0,
            "brightness_std": 0.0,
            "edge_density": 0.0,
        }
        self._score: int = 0
        self._risk_level: str = "LOW"

    @property
    def result(self) -> dict:
        with self._lock:
            return dict(self._result)

    @property
    def score(self) -> int:
        with self._lock:
            return self._score

    @property
    def risk_level(self) -> str:
        with self._lock:
            return self._risk_level

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each incoming WebRTC video frame with the full CCTV pipeline."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # 1. Motion Detection (Background Subtraction)
        fg_mask = self._back_sub.apply(img)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        motion_px = np.count_nonzero(fg_mask)
        motion_score = (motion_px / (h * w)) * 100

        # 2. Human Detection (Haar Cascades)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        bodies = BODY_CASCADE.detectMultiScale(gray, 1.1, 4)
        found_human = len(faces) > 0 or len(bodies) > 0

        # 3. Annotation (CCTV Style)
        annotated = img.copy()

        # HUD Overlay
        cv2.rectangle(annotated, (0, 0), (w, 40), (10, 10, 10), -1)
        status_color = (0, 0, 255) if (found_human or motion_score > 2) else (0, 255, 0)
        status_text = "ALERT" if found_human else ("MOTION" if motion_score > 2 else "SECURE")

        cv2.putText(annotated, f"SmartDetect CCTV - {status_text}", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Draw motion contours in Cyan
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 500:
                x, y, mw, mh = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x+mw, y+mh), (255, 255, 0), 1)

        # Draw Humans in Red
        for (x, y, fw, fh) in faces:
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), (0, 0, 255), 2)
            cv2.putText(annotated, "HUMAN FACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for (x, y, bw, bh) in bodies:
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            cv2.putText(annotated, "HUMAN BODY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Build metadata
        result = {
            "anomaly_type": status_text if status_text != "SECURE" else "Static Scene",
            "contour_count": len(cnts),
            "anomaly_area_pct": motion_score,
            "human_detected": found_human,
            "face_count": len(faces),
            "body_count": len(bodies),
            "brightness_std": float(np.std(gray)),
            "edge_density": 0.0,  # simplified for speed
        }
        score, risk_level = compute_risk_score(result)

        # Bottom HUD bar with score/risk
        cv2.rectangle(annotated, (0, h - 32), (w, h), (10, 10, 10), -1)
        score_color = (0, 0, 255) if risk_level == "HIGH" else (
            (0, 171, 255) if risk_level == "MEDIUM" else (0, 230, 118)
        )
        cv2.putText(annotated, f"Risk: {score} [{risk_level}]  |  Humans: {len(faces)+len(bodies)}  |  Motion: {motion_score:.1f}%",
                    (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)

        # Thread-safe state update
        with self._lock:
            self._result = result
            self._score = score
            self._risk_level = risk_level

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")




