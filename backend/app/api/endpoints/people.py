"""
People Management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class PersonCreate(BaseModel):
    first_name: str
    last_name: str
    personnel_code: str
    department: str
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class PersonUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    personnel_code: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class PersonResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    personnel_code: str
    department: str
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    created_at: datetime
    enrollments: List[Dict[str, Any]] = []


from database import PeopleDB


@router.get("/", response_model=List[PersonResponse])
async def get_people(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None)  # 'active', 'inactive', or None for all
):
    """Get list of people with optional filtering"""
    try:
        people = PeopleDB.get_all(skip=skip, limit=limit, search=search, status=status)
        return people
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get people: {str(e)}")


@router.post("/", response_model=PersonResponse)
async def create_person(person: PersonCreate):
    """Create a new person"""
    try:
        person_data = person.dict()
        new_person = PeopleDB.create(person_data)
        
        if new_person:
            return new_person
        else:
            raise HTTPException(status_code=400, detail="Personnel code already exists or creation failed")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create person: {str(e)}")


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(person_id: int):
    """Get a specific person by ID"""
    try:
        person = PeopleDB.get_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        return person
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get person: {str(e)}")


@router.put("/{person_id}", response_model=PersonResponse)
async def update_person(person_id: int, person_update: PersonUpdate):
    """Update a person's information"""
    try:
        update_data = person_update.dict(exclude_unset=True)
        updated_person = PeopleDB.update(person_id, update_data)
        
        if updated_person:
            return updated_person
        else:
            raise HTTPException(status_code=404, detail="Person not found or personnel code already exists")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update person: {str(e)}")


@router.delete("/{person_id}")
async def delete_person(person_id: int):
    """Delete a person"""
    try:
        # Get person info first for the response message
        person = PeopleDB.get_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        success = PeopleDB.delete(person_id)
        if success:
            return {
                "status": "success",
                "message": f"Person {person['first_name']} {person['last_name']} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Person not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete person: {str(e)}")


@router.post("/{person_id}/toggle-status")
async def toggle_person_status(person_id: int):
    """Toggle person's active status"""
    try:
        updated_person = PeopleDB.toggle_status(person_id)
        if not updated_person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        status = "activated" if updated_person["is_active"] else "deactivated"
        
        return {
            "status": "success",
            "message": f"Person {status} successfully",
            "person": updated_person
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle status: {str(e)}")


@router.get("/{person_id}/enrollments")
async def get_person_enrollments(person_id: int):
    """Get all enrollments for a specific person"""
    try:
        person = next((p for p in people_db if p["id"] == person_id), None)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return {
            "status": "success",
            "person_id": person_id,
            "enrollments": person.get("enrollments", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enrollments: {str(e)}")