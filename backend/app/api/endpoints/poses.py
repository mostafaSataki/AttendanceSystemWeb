"""
Face Poses API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from pathlib import Path

from app.core.config import settings

router = APIRouter()


class PoseResponse(BaseModel):
    id: int
    person_id: int
    pose_type: str  # FRONT, LEFT, RIGHT, UP, DOWN
    image_path: str
    quality_score: float
    created_at: str


# Mock poses database - replace with actual database implementation
poses_db = [
    {
        "id": 1,
        "person_id": 1,
        "pose_type": "FRONT",
        "image_path": "/face_images/person_1_front.jpg",
        "quality_score": 0.92,
        "created_at": "2024-01-10T09:20:00Z"
    },
    {
        "id": 2,
        "person_id": 1,
        "pose_type": "LEFT",
        "image_path": "/face_images/person_1_left.jpg",
        "quality_score": 0.88,
        "created_at": "2024-01-10T09:21:00Z"
    },
    {
        "id": 3,
        "person_id": 1,
        "pose_type": "RIGHT",
        "image_path": "/face_images/person_1_right.jpg",
        "quality_score": 0.90,
        "created_at": "2024-01-10T09:22:00Z"
    }
]


@router.get("/{person_id}", response_model=List[PoseResponse])
async def get_person_poses(person_id: int):
    """Get all poses for a specific person"""
    try:
        person_poses = [pose for pose in poses_db if pose["person_id"] == person_id]
        
        # Group poses by type for easy frontend consumption
        poses_by_type = {}
        for pose in person_poses:
            poses_by_type[pose["pose_type"]] = pose
        
        return {
            "status": "success",
            "person_id": person_id,
            "poses": person_poses,
            "poses_by_type": poses_by_type,
            "total_poses": len(person_poses)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get poses: {str(e)}")


@router.get("/")
async def get_all_poses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    person_id: Optional[int] = Query(None),
    pose_type: Optional[str] = Query(None)
):
    """Get poses with optional filtering"""
    try:
        filtered_poses = poses_db.copy()
        
        # Apply person filter
        if person_id:
            filtered_poses = [pose for pose in filtered_poses if pose["person_id"] == person_id]
        
        # Apply pose type filter
        if pose_type:
            if pose_type not in ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]:
                raise HTTPException(status_code=400, detail="Invalid pose type")
            filtered_poses = [pose for pose in filtered_poses if pose["pose_type"] == pose_type]
        
        # Apply pagination
        paginated_poses = filtered_poses[skip:skip + limit]
        
        return {
            "status": "success",
            "poses": paginated_poses,
            "total": len(filtered_poses),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get poses: {str(e)}")


@router.post("/{person_id}/upload")
async def upload_pose_image(
    person_id: int,
    pose_type: str,
    file: UploadFile = File(...)
):
    """Upload a pose image for a specific person"""
    try:
        if pose_type not in ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]:
            raise HTTPException(status_code=400, detail="Invalid pose type")
            
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create filename
        filename = f"person_{person_id}_{pose_type.lower()}.jpg"
        file_path = settings.FACE_IMAGES_DIR / filename
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create pose record
        new_pose = {
            "id": max([p["id"] for p in poses_db], default=0) + 1,
            "person_id": person_id,
            "pose_type": pose_type,
            "image_path": str(file_path),
            "quality_score": 0.85,  # This would be calculated by face quality assessment
            "created_at": "2024-01-15T10:00:00Z"
        }
        
        # Remove existing pose of same type for this person
        poses_db[:] = [p for p in poses_db if not (p["person_id"] == person_id and p["pose_type"] == pose_type)]
        poses_db.append(new_pose)
        
        return {
            "status": "success",
            "message": f"Pose {pose_type} uploaded successfully",
            "pose": new_pose
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload pose: {str(e)}")


@router.get("/{person_id}/image/{pose_type}")
async def get_pose_image(person_id: int, pose_type: str):
    """Get pose image file"""
    try:
        if pose_type not in ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]:
            raise HTTPException(status_code=400, detail="Invalid pose type")
        
        pose = next((p for p in poses_db if p["person_id"] == person_id and p["pose_type"] == pose_type), None)
        if not pose:
            raise HTTPException(status_code=404, detail="Pose not found")
        
        image_path = Path(pose["image_path"])
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(
            image_path,
            media_type="image/jpeg",
            filename=f"person_{person_id}_{pose_type.lower()}.jpg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get image: {str(e)}")


@router.delete("/{person_id}")
async def delete_person_poses(person_id: int):
    """Delete all poses for a specific person"""
    try:
        person_poses = [pose for pose in poses_db if pose["person_id"] == person_id]
        if not person_poses:
            raise HTTPException(status_code=404, detail="No poses found for this person")
        
        # Delete image files
        for pose in person_poses:
            image_path = Path(pose["image_path"])
            if image_path.exists():
                os.remove(image_path)
        
        # Remove from database
        poses_db[:] = [pose for pose in poses_db if pose["person_id"] != person_id]
        
        return {
            "status": "success",
            "message": f"Deleted {len(person_poses)} poses for person {person_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete poses: {str(e)}")


@router.delete("/{person_id}/{pose_type}")
async def delete_specific_pose(person_id: int, pose_type: str):
    """Delete a specific pose for a person"""
    try:
        if pose_type not in ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]:
            raise HTTPException(status_code=400, detail="Invalid pose type")
        
        pose = next((p for p in poses_db if p["person_id"] == person_id and p["pose_type"] == pose_type), None)
        if not pose:
            raise HTTPException(status_code=404, detail="Pose not found")
        
        # Delete image file
        image_path = Path(pose["image_path"])
        if image_path.exists():
            os.remove(image_path)
        
        # Remove from database
        poses_db[:] = [p for p in poses_db if not (p["person_id"] == person_id and p["pose_type"] == pose_type)]
        
        return {
            "status": "success",
            "message": f"Deleted pose {pose_type} for person {person_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pose: {str(e)}")


@router.get("/{person_id}/quality-report")
async def get_pose_quality_report(person_id: int):
    """Get quality report for all poses of a person"""
    try:
        person_poses = [pose for pose in poses_db if pose["person_id"] == person_id]
        if not person_poses:
            return {
                "status": "success",
                "person_id": person_id,
                "total_poses": 0,
                "average_quality": 0,
                "poses": []
            }
        
        total_quality = sum(pose["quality_score"] for pose in person_poses)
        average_quality = total_quality / len(person_poses)
        
        # Sort poses by quality score (descending)
        sorted_poses = sorted(person_poses, key=lambda x: x["quality_score"], reverse=True)
        
        return {
            "status": "success",
            "person_id": person_id,
            "total_poses": len(person_poses),
            "average_quality": round(average_quality, 3),
            "highest_quality": sorted_poses[0]["quality_score"],
            "lowest_quality": sorted_poses[-1]["quality_score"],
            "poses": sorted_poses
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality report: {str(e)}")