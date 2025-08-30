from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# Person schemas
class PersonBase(BaseModel):
    first_name: str
    last_name: str
    personnel_code: str
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    is_active: bool = True

class PersonCreate(PersonBase):
    pass

class PersonUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    personnel_code: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None

class Person(PersonBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Enrollment schemas
class EnrollmentBase(BaseModel):
    person_id: int
    confidence_score: Optional[int] = None
    is_active: bool = True

class EnrollmentCreate(EnrollmentBase):
    pass

class EnrollmentUpdate(BaseModel):
    confidence_score: Optional[int] = None
    is_active: Optional[bool] = None

class Enrollment(EnrollmentBase):
    id: int
    face_encoding_path: str
    face_image_path: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    person: Person
    
    class Config:
        from_attributes = True

# Attendance Log schemas
class AttendanceLogBase(BaseModel):
    person_id: int
    gate_name: Optional[str] = None
    direction: Optional[str] = None
    confidence_score: Optional[int] = None
    camera_name: Optional[str] = None
    status: Optional[str] = None

class AttendanceLogCreate(AttendanceLogBase):
    pass

class AttendanceLog(AttendanceLogBase):
    id: int
    created_at: datetime
    person: Person
    
    class Config:
        from_attributes = True

# Response schemas with relations
class PersonWithEnrollments(Person):
    enrollments: List[Enrollment] = []
    
    class Config:
        from_attributes = True