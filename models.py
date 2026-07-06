from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from db import Base

class DetectionLog(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True, nullable=False)
    plate_number = Column(String, index=True, nullable=False)
    vehicle_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    evidence_url = Column(String, nullable=True)