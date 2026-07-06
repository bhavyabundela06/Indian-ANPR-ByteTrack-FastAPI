"""
main.py — FastAPI application entry point.

FIXED: The old main.py was just a second copy of the router in routes.py —
there was no FastAPI() app anywhere, so `uvicorn main:app` would have failed
with "Attribute 'app' not found". This file now:
  1. Creates the FastAPI app
  2. Creates the database tables on startup
  3. Mounts /static so plate crop images (evidence) are served over HTTP
  4. Includes the single consolidated router from routes.py
  5. Enables CORS so the Streamlit dashboard can call the API

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import models
from db import engine
from routes import router

# Create tables if they don't exist yet (was never done anywhere before)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AegisVision ANPR Backend",
    version="1.0.0",
    description="Ingests detections from the AI pipeline and serves the dashboard.",
)

# Allow the Streamlit dashboard (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve evidence crops: anpr_colab.py writes static/crops/<PLATE>.jpg and the
# dashboard links to {API_BASE_URL}/static/crops/<PLATE>.jpg
os.makedirs("static/crops", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "online", "service": "AegisVision ANPR API"}