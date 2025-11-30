from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from datetime import datetime
from app.database import Base

class FaceRecord(Base):
    __tablename__ = "face_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    face_embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
