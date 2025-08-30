from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from config.database import get_db
from models import models
from models.schemas import Enrollment, EnrollmentCreate, EnrollmentUpdate
from services.face_service import FaceService

router = APIRouter()

# Initialize face service
face_service = FaceService()

# Create directories if they don't exist
os.makedirs("uploads/encodings", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)

@router.get("/", response_model=List[Enrollment])
async def get_enrollments(
    skip: int = 0,
    limit: int = 100,
    person_id: Optional[int] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Enrollment)
    
    if person_id:
        query = query.filter(models.Enrollment.person_id == person_id)
    
    if is_active is not None:
        query = query.filter(models.Enrollment.is_active == is_active)
    
    enrollments = query.offset(skip).limit(limit).all()
    return enrollments

@router.get("/{enrollment_id}", response_model=Enrollment)
async def get_enrollment(enrollment_id: int, db: Session = Depends(get_db)):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found"
        )
    return enrollment

@router.post("/", response_model=Enrollment, status_code=status.HTTP_201_CREATED)
async def create_enrollment(
    person_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Check if person exists and is active
    person = db.query(models.Person).filter(
        models.Person.id == person_id,
        models.Person.is_active == True
    ).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active person not found"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Generate unique filenames
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        image_filename = f"{uuid.uuid4()}.{file_extension}"
        encoding_filename = f"{uuid.uuid4()}.pkl"
        
        image_path = f"uploads/images/{image_filename}"
        encoding_path = f"uploads/encodings/{encoding_filename}"
        
        # Save uploaded image
        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process face and get encoding
        face_encoding, confidence_score = face_service.process_face(image_path)
        
        # Save face encoding
        face_service.save_face_encoding(face_encoding, encoding_path)
        
        # Create enrollment record
        enrollment_data = {
            "person_id": person_id,
            "face_encoding_path": encoding_path,
            "face_image_path": image_path,
            "confidence_score": confidence_score,
            "is_active": True
        }
        
        db_enrollment = models.Enrollment(**enrollment_data)
        db.add(db_enrollment)
        db.commit()
        db.refresh(db_enrollment)
        
        return db_enrollment
        
    except Exception as e:
        # Clean up files if enrollment fails
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        if 'encoding_path' in locals() and os.path.exists(encoding_path):
            os.remove(encoding_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process face enrollment: {str(e)}"
        )

@router.put("/{enrollment_id}", response_model=Enrollment)
async def update_enrollment(
    enrollment_id: int,
    enrollment_update: EnrollmentUpdate,
    db: Session = Depends(get_db)
):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found"
        )
    
    # Update only provided fields
    update_data = enrollment_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(enrollment, field, value)
    
    db.commit()
    db.refresh(enrollment)
    return enrollment

@router.delete("/{enrollment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_enrollment(enrollment_id: int, db: Session = Depends(get_db)):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found"
        )
    
    # Delete associated files
    if os.path.exists(enrollment.face_image_path):
        os.remove(enrollment.face_image_path)
    if os.path.exists(enrollment.face_encoding_path):
        os.remove(enrollment.face_encoding_path)
    
    db.delete(enrollment)
    db.commit()
    return None

@router.patch("/{enrollment_id}/toggle-active", response_model=Enrollment)
async def toggle_enrollment_active(enrollment_id: int, db: Session = Depends(get_db)):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found"
        )
    
    enrollment.is_active = not enrollment.is_active
    db.commit()
    db.refresh(enrollment)
    return enrollment