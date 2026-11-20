from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)

    image_path = Column(String(255), nullable=False)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    classification = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    decision = Column(String(255), nullable=True)

    status = Column(String(50), default="PROCESSING")

    created_at = Column(DateTime(timezone=True), server_default=func.now())