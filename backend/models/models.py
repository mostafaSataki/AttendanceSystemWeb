from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database import Base

class Person(Base):
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    personnel_code = Column(String(50), unique=True, nullable=False)
    department = Column(String(100))
    position = Column(String(100))
    email = Column(String(255))
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship with enrollments
    enrollments = relationship("Enrollment", back_populates="person")

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    face_encoding_path = Column(String(500), nullable=False)
    face_image_path = Column(String(500), nullable=False)
    confidence_score = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship with person
    person = relationship("Person", back_populates="enrollments")

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    gate_name = Column(String(100))
    direction = Column(String(20))  # entry, exit
    confidence_score = Column(Integer)
    camera_name = Column(String(100))
    status = Column(String(20))  # authorized, unauthorized
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship with person
    person = relationship("Person")