import os

from dotenv import load_dotenv

load_dotenv()

# --- PATH MANAGEMENT ---
# This automatically finds the absolute path to your project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# --- CAMERA / VIDEO SOURCE ---
# True  -> live RTSP camera (set RTSP_URL in your .env file)
# False -> local test video (videos/bike1.mp4)
USE_RTSP = os.getenv("USE_RTSP", "true").lower() == "true"

# Real camera address lives in .env (which is gitignored) — never hardcode it here.
RTSP_URL = os.getenv("RTSP_URL", "")

VIDEO_SOURCE = RTSP_URL if USE_RTSP else os.path.join(VIDEOS_DIR, "bike1.mp4")
CAMERA_LOCATION = os.getenv("CAMERA_LOCATION", "Main_Entrance")
FRAME_RATE = 30

# --- AI MODELS ---
PLATE_MODEL = os.path.join(MODELS_DIR, "best.pt")
VEHICLE_MODEL = os.path.join(MODELS_DIR, "yolov8n.pt")

# --- API SETTINGS ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
