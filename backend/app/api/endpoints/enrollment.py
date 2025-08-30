"""
Face Enrollment API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import io
import base64
from PIL import Image
import numpy as np
import cv2
import os
import time
from pathlib import Path
from pydantic import BaseModel

from app.core.config import settings
from app.services.backend_enrollment_service import BackendEnrollmentService

router = APIRouter()

# Initialize enrollment service
enrollment_service = BackendEnrollmentService()


class StartEnrollmentRequest(BaseModel):
    source: Optional[str] = "camera"  # "camera", "video"
    source_config: Optional[Dict[str, Any]] = None


@router.post("/start")
async def start_enrollment(request: Request):
    """Start face enrollment session"""
    try:
        face_service = request.app.state.face_service
        
        # Extract source and config from request body
        source = "camera"  # default
        source_config = {}
        
        # Try to get JSON body
        try:
            body = await request.json()
            source = body.get("source", "camera")
            source_config = body.get("source_config", {})
        except Exception:
            # No body or invalid JSON, use defaults
            pass
        
        print(f"Starting enrollment with source: {source}, config: {source_config}")
        
        # Generate session ID
        session_id = f"enrollment_{source}_{int(time.time())}"
        
        # Initialize video source based on type
        if source == "video" and source_config.get("video_path"):
            video_path = source_config['video_path']
            print(f"📹 Video path: {video_path}")
            
            # Validate video file exists
            if not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
                
            # Initialize video capture
            result = enrollment_service.start_video_enrollment(session_id, video_path)
            
        elif source == "camera":
            camera_index = source_config.get("camera_index", 0)
            rtsp_url = source_config.get("rtsp_url")
            print(f"📷 Camera source - Index: {camera_index}, RTSP: {rtsp_url}")
            
            # Initialize camera capture
            result = enrollment_service.start_camera_enrollment(session_id, camera_index, rtsp_url)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid source or missing source configuration")
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to start enrollment"))
        
        return {
            "status": "success", 
            "message": result.get("message", f"Enrollment started with {source}"),
            "session_id": result.get("session_id", session_id),
            "source": source,
            "source_config": source_config
        }
    except Exception as e:
        print(f"Error starting enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start enrollment: {str(e)}")


@router.post("/stop")
async def stop_enrollment(request: Request):
    """Stop face enrollment session"""
    try:
        face_service = request.app.state.face_service
        result = await face_service.stop_enrollment()
        return {"status": "success", "message": "Enrollment stopped", "collected_poses": result.get("poses", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop enrollment: {str(e)}")


@router.post("/capture-pose")
async def capture_pose(request: Request, pose_type: str):
    """Capture a specific pose during enrollment"""
    try:
        if pose_type not in ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]:
            raise HTTPException(status_code=400, detail="Invalid pose type")
            
        face_service = request.app.state.face_service
        result = await face_service.capture_pose(pose_type)
        
        if result["success"]:
            return {
                "status": "success",
                "pose_type": pose_type,
                "quality_score": result.get("quality", 0.0),
                "image_path": result.get("image_path"),
                "message": f"Pose {pose_type} captured successfully"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to capture pose"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture pose: {str(e)}")


@router.post("/upload-single-image")
async def upload_single_image(person_id: int, file: UploadFile = File(...)):
    """Upload a single face image for a person"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to OpenCV format
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
            
        # Process with enrollment service
        result = enrollment_service.process_single_image(image_cv, person_id)
        
        if result["success"]:
            return {
                "status": "success",
                "message": "Image uploaded and processed successfully",
                "face_detected": result.get("face_detected", False),
                "quality_score": result.get("quality", 0.0),
                "saved_path": result.get("saved_path")
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process image"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/confirm-enrollment")
async def confirm_enrollment(person_id: int, poses_data: List[Dict[str, Any]]):
    """Confirm and save enrollment with collected poses"""
    try:
        # Process enrollment with collected poses
        result = enrollment_service.confirm_enrollment(person_id, poses_data)
        
        if result["success"]:
            return {
                "status": "success",
                "message": "Enrollment confirmed successfully",
                "person_id": person_id,
                "enrolled_poses": len(poses_data),
                "enrollment_id": result.get("enrollment_id")
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to confirm enrollment"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to confirm enrollment: {str(e)}")


@router.get("/poses/{person_id}")
async def get_person_poses(person_id: int):
    """Get all poses for a specific person"""
    try:
        poses = enrollment_service.get_person_poses(person_id)
        return {
            "status": "success",
            "person_id": person_id,
            "poses": poses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get poses: {str(e)}")


@router.delete("/poses/{person_id}")
async def delete_person_poses(person_id: int):
    """Delete all poses for a specific person"""
    try:
        result = enrollment_service.delete_person_poses(person_id)
        if result["success"]:
            return {
                "status": "success",
                "message": f"All poses deleted for person {person_id}"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete poses"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete poses: {str(e)}")


@router.get("/session/status")
async def get_enrollment_status(request: Request):
    """Get current enrollment session status"""
    try:
        face_service = request.app.state.face_service
        status_info = await face_service.get_enrollment_status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")