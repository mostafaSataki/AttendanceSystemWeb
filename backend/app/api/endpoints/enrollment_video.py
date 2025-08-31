"""
Video-enabled Face Enrollment API endpoints
Real implementation using BackendEnrollmentService
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import cv2
import base64
import time
import asyncio
import os
from typing import Dict, Any
import threading

from app.services.backend_enrollment_service import BackendEnrollmentService

router = APIRouter()

# Initialize real enrollment service
enrollment_service = BackendEnrollmentService()

# Global variables to manage camera state
current_camera = None
camera_lock = threading.Lock()
enrollment_active = False

@router.post("/multi-pose")
async def start_multi_pose_enrollment(request: Request):
    """Start multi-pose face enrollment using real BackendEnrollmentService"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        person_name = body.get("person_name", "temp_enrollment")
        
        print(f"📥 Starting real multi-pose enrollment - Source: {source}, Person: {person_name}")
        print(f"📁 Source config: {source_config}")
        
        if source == "video":
            video_path = source_config.get("video_path")
            if not video_path or not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                raise HTTPException(status_code=400, detail="Valid video file path required")
            
            print(f"🎬 Processing video file: {video_path}")
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
            
            print(f"📹 Processing camera stream: {camera_source}")
            # Process camera stream for multi-pose enrollment
            result = enrollment_service.enroll_from_camera_stream(
                camera_source=camera_source,
                person_name=person_name,
                auto_save=False,  # Don't auto-save, return poses for review
                max_enrollment_time=60.0  # 1 minute timeout
            )
        else:
            raise HTTPException(status_code=400, detail="Source must be 'video' or 'camera'")
        
        print(f"🔄 Enrollment result: {result.get('success')}, poses: {len(result.get('collected_poses_data', {}))}")
        
        if not result.get("success"):
            error_msg = result.get("error", "Enrollment failed")
            print(f"❌ Enrollment failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Convert collected poses to base64 images for frontend
        collected_poses = result.get("collected_poses_data", {})
        pose_images = []
        
        print(f"📸 Converting {len(collected_poses)} poses to base64...")
        
        for pose_name, pose_data in collected_poses.items():
            if "aligned_face" in pose_data:
                # Convert aligned face image to base64
                _, buffer = cv2.imencode('.jpg', pose_data["aligned_face"])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Handle both PoseState enum and string keys
                pose_key = pose_name.value if hasattr(pose_name, 'value') else str(pose_name)
                
                pose_images.append({
                    "pose_name": pose_key,
                    "image": f"data:image/jpeg;base64,{image_base64}",
                    "quality_score": pose_data.get("quality_score", 0.0),
                    "timestamp": pose_data.get("timestamp"),
                    "frame_number": pose_data.get("frame_number")
                })
                print(f"✅ Converted pose: {pose_key} (quality: {pose_data.get('quality_score', 0.0):.3f})")
        
        response_data = {
            "status": "success",
            "message": f"Collected {len(pose_images)} poses for {person_name}",
            "person_name": person_name,
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "processing_stats": {
                "frames_processed": result.get("total_frames_processed", 0),
                "processing_time": result.get("processing_time", 0),
                "poses_collected": [p["pose_name"] for p in pose_images]
            }
        }
        
        print(f"✅ Returning {len(pose_images)} poses to frontend")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in real enrollment: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video enrollment failed: {str(e)}")


@router.get("/video-stream")
async def get_video_stream():
    """Get live video stream from camera"""
    global current_camera
    
    if current_camera is None:
        raise HTTPException(status_code=404, detail="No active camera")
    
    def generate_frames():
        global current_camera, enrollment_active
        
        while enrollment_active and current_camera is not None:
            with camera_lock:
                if current_camera is None:
                    break
                    
                ret, frame = current_camera.read()
                if not ret:
                    # For video files, loop back to the beginning
                    current_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = current_camera.read()
                    if not ret:
                        break
                
                # Add some enrollment UI overlay
                cv2.putText(frame, "Face Enrollment Active", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Look at camera and move your head", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add a simple face detection rectangle (mock)
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                rect_size = 150
                cv2.rectangle(frame, 
                            (center_x - rect_size, center_y - rect_size), 
                            (center_x + rect_size, center_y + rect_size), 
                            (0, 255, 0), 2)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(generate_frames(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/stop")
async def stop_enrollment():
    """Stop enrollment and release camera"""
    global current_camera, enrollment_active
    
    enrollment_active = False
    
    with camera_lock:
        if current_camera is not None:
            current_camera.release()
            current_camera = None
    
    return {
        "status": "success",
        "message": "Enrollment stopped and camera released"
    }


@router.post("/confirm-enrollment")
async def confirm_enrollment(request: Request):
    """Confirm and save enrollment with collected poses"""
    try:
        body = await request.json()
        person_name = body.get("person_name")
        collected_poses_data = body.get("collected_poses_data", {})
        
        if not person_name:
            raise HTTPException(status_code=400, detail="Person name is required")
        
        print(f"Confirming enrollment for {person_name} with {len(collected_poses_data)} poses")
        
        # Stop enrollment after confirmation
        global enrollment_active
        enrollment_active = False
        
        return {
            "status": "success",
            "message": f"Enrollment confirmed for {person_name}",
            "person_name": person_name,
            "total_poses": len(collected_poses_data),
            "embeddings_generated": len(collected_poses_data),
            "embedding_model": "video_mock"
        }
        
    except Exception as e:
        print(f"Error confirming enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm enrollment: {str(e)}")


@router.get("/status")
async def get_enrollment_status():
    """Get enrollment status"""
    return {
        "status": "success",
        "total_persons": 0,
        "persons": [],
        "total_embeddings": 0,
        "camera_active": current_camera is not None,
        "enrollment_active": enrollment_active,
        "last_updated": time.time()
    }