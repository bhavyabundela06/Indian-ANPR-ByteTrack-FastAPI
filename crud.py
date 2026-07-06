"""
crud.py — database operations.

FIXES:
- get_statistics: `today_count` previously just returned the total count.
  It now actually filters rows whose timestamp falls on the current UTC day.
- Vehicle breakdown counts done with a single grouped query instead of
  one full-table COUNT per type.
"""

from datetime import datetime, time

from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Optional
import models


def get_all_detections(db: Session, skip: int = 0, limit: int = 100):
    return (
        db.query(models.DetectionLog)
        .order_by(models.DetectionLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_detections_by_id(db: Session, detection_id: int):
    return (
        db.query(models.DetectionLog)
        .filter(models.DetectionLog.id == detection_id)
        .first()
    )


def search_plate(db: Session, plate_text: str):
    return (
        db.query(models.DetectionLog)
        .filter(models.DetectionLog.plate_number.contains(plate_text.upper()))
        .order_by(models.DetectionLog.timestamp.desc())
        .all()
    )


def get_statistics(db: Session):
    total = db.query(models.DetectionLog).count()

    # FIXED: actually count today's rows (timestamps are stored in UTC)
    today_start = datetime.combine(datetime.utcnow().date(), time.min)
    today_count = (
        db.query(models.DetectionLog)
        .filter(models.DetectionLog.timestamp >= today_start)
        .count()
    )

    # One grouped query instead of a COUNT per vehicle type
    breakdown_rows = (
        db.query(models.DetectionLog.vehicle_type, func.count(models.DetectionLog.id))
        .group_by(models.DetectionLog.vehicle_type)
        .all()
    )
    counts = {vtype: n for vtype, n in breakdown_rows}
    cars = counts.get("car", 0)
    bikes = counts.get("bike", 0)
    trucks = counts.get("truck", 0)

    return {
        "total_detections": total,
        "today_count": today_count,
        "vehicle_breakdown": {
            "cars": cars,
            "bikes": bikes,
            "trucks": trucks,
            "others": total - (cars + bikes + trucks),
        },
    }


   # add at the top of crud.py

def create_detection(
    db: Session,
    camera_id: str,
    plate_number: str,
    vehicle_type: str,
    confidence: float,
    evidence_url: Optional[str] = None,   # was: str = None
):
    db_log = models.DetectionLog(
        camera_id=camera_id,
        plate_number=plate_number.upper(),
        vehicle_type=vehicle_type,
        confidence=confidence,
        evidence_url=evidence_url,
        timestamp=datetime.utcnow(),
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log