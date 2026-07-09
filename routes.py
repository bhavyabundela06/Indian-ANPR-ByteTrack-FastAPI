"""
routes.py — the single, consolidated API router.
"""
import os
import shutil
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import crud
from db import get_db

# The router automatically applies /api/v1 to everything below it
router = APIRouter(prefix="/api/v1", tags=["ANPR Core Router Engine"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AIDetectionCreate(BaseModel):
    camera_id: str
    plate_number: str
    vehicle_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_url: Optional[str] = None


class DetectionOut(BaseModel):
    id: int
    camera_id: str
    plate_number: str
    vehicle_type: str
    confidence: float
    timestamp: datetime
    evidence_url: Optional[str] = None

    class Config:
        from_attributes = True  # pydantic v2 (use orm_mode = True on v1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/upload-evidence")
async def upload_evidence(file: UploadFile = File(...)):
    """Receives the image file from Colab and saves it physically to the local machine."""
    os.makedirs("static/crops", exist_ok=True)
    file_path = f"static/crops/{file.filename}"
    
    # Save the physical file to your local hard drive
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Return the local path so Colab can attach it to the database entry
    return {"evidence_url": f"/{file_path}"}


@router.get("/detections", response_model=List[DetectionOut])
def get_live_detections(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Fetches real logs from the database for the dashboard."""
    return crud.get_all_detections(db, skip=skip, limit=limit)


@router.get("/detections/search", response_model=List[DetectionOut])
def search_detections(q: str, db: Session = Depends(get_db)):
    """Partial plate search, e.g. /detections/search?q=MH03"""
    return crud.search_plate(db, q)


@router.get("/detections/{detection_id}", response_model=DetectionOut)
def get_detection_by_id(detection_id: int, db: Session = Depends(get_db)):
    det = crud.get_detections_by_id(db, detection_id)
    if det is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return det


@router.post("/detections/add", response_model=DetectionOut)
def add_ai_detection(data: AIDetectionCreate, db: Session = Depends(get_db)):
    """Receives POST from the AI pipeline (anpr_colab.py) and saves to the DB."""
    return crud.create_detection(
        db=db,
        camera_id=data.camera_id,
        plate_number=data.plate_number,
        vehicle_type=data.vehicle_type,
        confidence=data.confidence,
        evidence_url=data.evidence_url,
    )


@router.get("/analytics")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    return crud.get_statistics(db)