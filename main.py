"""
main.py — FastAPI application entry point.
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import models
from db import engine
from routes import router

# Create tables if they don't exist yet
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

# Serve evidence crops over HTTP
os.makedirs("static/crops", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- NEW: Endpoint to receive physical image uploads from Colab ---
@app.post("/api/v1/upload-evidence")
async def upload_evidence(file: UploadFile = File(...)):
    # ... (rest of the code)
    """Receives the image file from Colab and saves it physically to the Mac."""
    file_path = f"static/crops/{file.filename}"
    
    # Save the physical file to your Mac's hard drive
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Return the local path so Colab can attach it to the database entry
    return {"evidence_url": f"/{file_path}"}


app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "online", "service": "AegisVision ANPR API"}