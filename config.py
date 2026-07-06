import os

# --- PATH MANAGEMENT ---
# This automatically finds the absolute path to your project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# --- CAMERA / VIDEO SOURCE ---
#RTSP_URL = "if you wish to test on an IP CAMERA"
VIDEO_SOURCE = os.path.join(VIDEOS_DIR, "bike1.mp4")
CAMERA_LOCATION = "Main_Entrance"
FRAME_RATE = 30

# --- AI MODELS ---
PLATE_MODEL = os.path.join(MODELS_DIR, "best.pt")
VEHICLE_MODEL = os.path.join(MODELS_DIR, "yolov8n.pt")

# --- API SETTINGS ---
API_BASE_URL = "http://localhost:8000"