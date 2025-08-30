from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from config.database import get_db
from models import models
from models.schemas import Person, PersonCreate, PersonUpdate, PersonWithEnrollments

router = APIRouter()

@router.get("/", response_model=List[Person])
async def get_persons(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    department: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Person)
    
    if is_active is not None:
        query = query.filter(models.Person.is_active == is_active)
    
    if department:
        query = query.filter(models.Person.department.ilike(f"%{department}%"))
    
    if search:
        query = query.filter(
            (models.Person.first_name.ilike(f"%{search}%")) |
            (models.Person.last_name.ilike(f"%{search}%")) |
            (models.Person.personnel_code.ilike(f"%{search}%"))
        )
    
    persons = query.offset(skip).limit(limit).all()
    return persons

@router.get("/{person_id}", response_model=PersonWithEnrollments)
async def get_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    return person

@router.post("/", response_model=Person, status_code=status.HTTP_201_CREATED)
async def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    # Check if personnel code already exists
    existing_person = db.query(models.Person).filter(
        models.Person.personnel_code == person.personnel_code
    ).first()
    if existing_person:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Personnel code already exists"
        )
    
    db_person = models.Person(**person.dict())
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person

@router.put("/{person_id}", response_model=Person)
async def update_person(person_id: int, person_update: PersonUpdate, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    # Check if personnel code is being updated and already exists
    if person_update.personnel_code and person_update.personnel_code != person.personnel_code:
        existing_person = db.query(models.Person).filter(
            models.Person.personnel_code == person_update.personnel_code,
            models.Person.id != person_id
        ).first()
        if existing_person:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Personnel code already exists"
            )
    
    # Update only provided fields
    update_data = person_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(person, field, value)
    
    db.commit()
    db.refresh(person)
    return person

@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    db.delete(person)
    db.commit()
    return None

@router.patch("/{person_id}/toggle-active", response_model=Person)
async def toggle_person_active(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    person.is_active = not person.is_active
    db.commit()
    db.refresh(person)
    return person

@router.patch("/{person_id}/deactivate", response_model=Person)
async def deactivate_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    person.is_active = False
    db.commit()
    db.refresh(person)
    return person

@router.patch("/{person_id}/activate", response_model=Person)
async def activate_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Person not found"
        )
    
    person.is_active = True
    db.commit()
    db.refresh(person)
    return person