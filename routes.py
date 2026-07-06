"""
routes.py — the single, consolidated API router.

FIXED: main.py and routes.py previously defined two nearly identical routers
with the same prefix and overlapping endpoints. All endpoints now live here;
main.py just includes this router. Also added the endpoints that existed only
in one of the two files (analytics) plus search / get-by-id which crud.py
already supported but were never exposed.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import crud
from db import get_db

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