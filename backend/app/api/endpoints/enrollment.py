"""
Face Enrollment API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
from app.services.face_engine.combination_engine import CombinationType

router = APIRouter()

# Initialize enrollment service
enrollment_service = BackendEnrollmentService()


class StartEnrollmentRequest(BaseModel):
    source: Optional[str] = "camera"  # "camera", "video"
    source_config: Optional[Dict[str, Any]] = None


@router.post("/start")
async def start_enrollment(request: Request):
    """Start multi-pose face enrollment session"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        person_name = body.get("person_name", "temp_enrollment")
        
        print(f"📥 Starting multi-pose enrollment - Source: {source}, Person: {person_name}")
        
        if source == "video":
            video_path = source_config.get("video_path")
            if not video_path or not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail="Valid video file path required")
            
            # Process video for multi-pose enrollment
            result = enrollment_service.enroll_from_video_backend(
                video_path=video_path,
                person_name=person_name,
                auto_save=False  # Don't auto-save, return poses for review
            )
            
        elif source == "camera":
            camera_source = source_config.get("camera_index", 0)
            if "rtsp_url" in source_config:
                camera_source = source_config["rtsp_url"]
            
            # Process camera stream for multi-pose enrollment
            result = enrollment_service.enroll_from_camera_stream(
                camera_source=camera_source,
                person_name=person_name,
                auto_save=False,  # Don't auto-save, return poses for review
                max_enrollment_time=60.0  # 1 minute timeout
            )
        else:
            raise HTTPException(status_code=400, detail="Source must be 'video' or 'camera'")
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Enrollment failed"))
        
        # Convert collected poses to base64 images for frontend
        collected_poses = result.get("collected_poses_data", {})
        pose_images = {}
        
        for pose_name, pose_data in collected_poses.items():
            if "aligned_face" in pose_data:
                # Convert aligned face image to base64
                _, buffer = cv2.imencode('.jpg', pose_data["aligned_face"])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                pose_images[pose_name] = {
                    "image": f"data:image/jpeg;base64,{image_base64}",
                    "quality_score": pose_data.get("quality_score", 0.0),
                    "timestamp": pose_data.get("timestamp")
                }
        
        return {
            "status": "success",
            "message": f"Collected {len(pose_images)} poses for review",
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "person_name": person_name,
            "processing_stats": {
                "frames_processed": result.get("total_frames_processed", 0),
                "processing_time": result.get("processing_time", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


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
async def confirm_enrollment(request: Request):
    """Confirm and save enrollment with collected poses"""
    try:
        body = await request.json()
        person_name = body.get("person_name")
        collected_poses_data = body.get("collected_poses_data", {})
        
        if not person_name:
            raise HTTPException(status_code=400, detail="Person name is required")
        
        if not collected_poses_data:
            raise HTTPException(status_code=400, detail="No poses data provided")
        
        # Reconstruct pose data from base64 images
        pose_data_for_save = {}
        for pose_name, pose_info in collected_poses_data.items():
            # Decode base64 image back to numpy array
            image_data = pose_info["image"]
            if image_data.startswith("data:image/jpeg;base64,"):
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            pose_data_for_save[pose_name] = {
                "aligned_face": image_np,
                "quality_score": pose_info.get("quality_score", 0.0),
                "timestamp": pose_info.get("timestamp")
            }
        
        # Save enrollment using backend service
        result = enrollment_service.save_pending_enrollment(person_name, pose_data_for_save)
        
        if result["success"]:
            return {
                "status": "success",
                "message": f"Enrollment confirmed for {person_name}",
                "person_name": person_name,
                "total_poses": result.get("total_poses", 0),
                "embeddings_generated": result.get("embeddings_generated", 0),
                "embedding_model": result.get("embedding_model", "unknown")
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to save enrollment"))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error confirming enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm enrollment: {str(e)}")


@router.get("/poses/{person_name}")
async def get_person_poses(person_name: str):
    """Get all poses for a specific person"""
    try:
        enrollment_status = enrollment_service.get_enrollment_status(person_name)
        
        if not enrollment_status.get("enrolled"):
            raise HTTPException(status_code=404, detail=f"Person '{person_name}' not enrolled")
        
        poses = enrollment_status.get("poses", {})
        pose_images = {}
        
        # Convert pose images to base64 for frontend display
        person_folder = Path("face_images") / person_name
        for pose_name, pose_data in poses.items():
            filename = pose_data.get("filename")
            if filename:
                image_path = person_folder / filename
                if image_path.exists():
                    # Read and encode image
                    image = cv2.imread(str(image_path))
                    _, buffer = cv2.imencode('.jpg', image)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    pose_images[pose_name] = {
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "quality_score": pose_data.get("quality_score", 0.0),
                        "timestamp": pose_data.get("timestamp"),
                        "has_embedding": pose_data.get("has_embedding", False)
                    }
        
        return {
            "status": "success",
            "person_name": person_name,
            "enrolled_date": enrollment_status.get("enrolled_date"),
            "total_poses": len(poses),
            "embeddings_count": enrollment_status.get("embeddings_count", 0),
            "pose_images": pose_images
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get poses: {str(e)}")


@router.delete("/poses/{person_name}")
async def delete_person_poses(person_name: str):
    """Delete all poses for a specific person"""
    try:
        result = enrollment_service.delete_enrollment(person_name)
        if result["success"]:
            return {
                "status": "success",
                "message": f"All enrollment data deleted for {person_name}"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete enrollment"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete enrollment: {str(e)}")


@router.get("/status")
async def get_enrollment_status():
    """Get enrollment status for all persons"""
    try:
        status = enrollment_service.get_enrollment_status()
        return {
            "status": "success",
            "total_persons": status.get("total_persons", 0),
            "persons": status.get("persons", []),
            "total_embeddings": status.get("total_embeddings", 0),
            "last_updated": status.get("last_updated")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/status/{person_name}")
async def get_person_enrollment_status(person_name: str):
    """Get enrollment status for a specific person"""
    try:
        status = enrollment_service.get_enrollment_status(person_name)
        return {
            "status": "success",
            "person_data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get person status: {str(e)}")

@router.post("/multi-pose")
async def start_multi_pose_enrollment(request: Request):
    """Start multi-pose enrollment and return collected pose images"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        person_name = body.get("person_name", f"temp_{int(time.time())}")
        
        print(f"🎯 Starting multi-pose enrollment for {person_name}")
        
        if source == "video":
            video_path = source_config.get("video_path")
            if not video_path or not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail="Valid video file path required")
            
            # Process video for multi-pose enrollment
            result = enrollment_service.enroll_from_video_backend(
                video_path=video_path,
                person_name=person_name,
                auto_save=False  # Don't auto-save, return poses for review
            )
            
        elif source == "camera":
            camera_source = source_config.get("camera_index", 0)
            if "rtsp_url" in source_config:
                camera_source = source_config["rtsp_url"]
            
            # Process camera stream for multi-pose enrollment
            result = enrollment_service.enroll_from_camera_stream(
                camera_source=camera_source,
                person_name=person_name,
                auto_save=False,  # Don't auto-save, return poses for review
                max_enrollment_time=60.0  # 1 minute timeout
            )
        else:
            raise HTTPException(status_code=400, detail="Source must be 'video' or 'camera'")
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Enrollment failed"))
        
        # Convert collected poses to base64 images for frontend
        collected_poses = result.get("collected_poses_data", {})
        pose_images = []
        
        for pose_name, pose_data in collected_poses.items():
            pose_key = pose_name.value if hasattr(pose_name, 'value') else str(pose_name)
            
            if "aligned_face" in pose_data:
                # Convert aligned face image to base64
                _, buffer = cv2.imencode('.jpg', pose_data["aligned_face"])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                pose_images.append({
                    "pose_name": pose_key,
                    "image": f"data:image/jpeg;base64,{image_base64}",
                    "quality_score": pose_data.get("quality_score", 0.0),
                    "timestamp": pose_data.get("timestamp"),
                    "frame_number": pose_data.get("frame_number")
                })
        
        return {
            "status": "success",
            "message": f"Collected {len(pose_images)} poses for {person_name}",
            "person_name": person_name,
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "processing_stats": {
                "frames_processed": result.get("total_frames_processed", 0),
                "processing_time": result.get("processing_time", 0),
                "poses_collected": result.get("poses_collected", [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in multi-pose enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-pose enrollment failed: {str(e)}")


@router.get("/video-stream")
async def video_stream(request: Request):
    """Stream current enrollment video"""
    try:
        face_service = request.app.state.face_service
        
        def generate_frames():
            while face_service.is_running and face_service.current_capture is not None:
                ret, frame = face_service.current_capture.read()
                if not ret:
                    break
                
                # Process frame for face detection/enrollment
                face_service.current_frame = frame
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        if not face_service.is_running or face_service.current_capture is None:
            raise HTTPException(status_code=404, detail="No active enrollment session")
        
        return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")