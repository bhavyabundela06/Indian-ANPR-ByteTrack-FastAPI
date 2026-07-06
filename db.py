"""
db.py — database engine and session factory.

FIX: `from sqlalchemy.ext.declarative import declarative_base` is deprecated
(and removed in SQLAlchemy 2.x). Import it from sqlalchemy.orm instead.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Falls back to a local SQLite file when no DATABASE_URL is configured
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aegisvision_production.db")

# SQLite needs check_same_thread=False when used from FastAPI's thread pool
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()