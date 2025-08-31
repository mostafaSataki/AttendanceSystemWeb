"""
Video-enabled Face Enrollment API endpoints
Simple implementation that shows real video but returns mock face data
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import cv2
import base64
import time
import asyncio
from typing import Dict, Any
import threading

router = APIRouter()

# Global variables to manage camera state
current_camera = None
camera_lock = threading.Lock()
enrollment_active = False

@router.post("/multi-pose")
async def start_multi_pose_enrollment(request: Request):
    """Start multi-pose face enrollment with real video stream"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        person_name = body.get("person_name", "temp_enrollment")
        
        print(f"Starting video enrollment - Source: {source}, Person: {person_name}")
        
        global current_camera, enrollment_active
        
        # Initialize camera or video file
        with camera_lock:
            if current_camera is None:
                if source == "video":
                    # Handle video file
                    video_path = source_config.get("video_path")
                    if video_path:
                        print(f"Opening video file: {video_path}")
                        import os
                        if os.path.exists(video_path):
                            current_camera = cv2.VideoCapture(video_path)
                            print(f"Video file exists. Opened successfully: {current_camera.isOpened()}")
                        else:
                            print(f"ERROR: Video file does not exist: {video_path}")
                            current_camera = None
                    else:
                        print("ERROR: No video_path provided for video source")
                        current_camera = None
                else:
                    # Handle camera
                    camera_index = source_config.get("camera_index", 0)
                    print(f"Opening camera with index: {camera_index}")
                    current_camera = cv2.VideoCapture(camera_index)
                    print(f"Camera opened successfully: {current_camera.isOpened()}")
                
                if current_camera is None or not current_camera.isOpened():
                    print(f"WARNING: Could not open {'video file' if source == 'video' else 'camera'}, using mock video mode")
                    current_camera = None
                    # Continue without camera - will return mock data but indicate no video stream
                
                # Set camera properties if camera is available
                if current_camera is not None:
                    current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    current_camera.set(cv2.CAP_PROP_FPS, 30)
        
        enrollment_active = True
        
        # Simulate collecting poses over time (mock data but with real video running)
        await asyncio.sleep(2)  # Simulate processing time
        
        # Create mock face images for different poses
        pose_images = []
        poses = ["FRONT", "LEFT", "RIGHT"]
        
        for i, pose_name in enumerate(poses):
            # Create a simple colored rectangle as mock face image
            import numpy as np
            mock_image = np.zeros((112, 112, 3), dtype=np.uint8)
            # Different color for each pose
            if pose_name == "FRONT":
                mock_image[:, :] = [100, 150, 200]  # Blue-ish
            elif pose_name == "LEFT":
                mock_image[:, :] = [150, 100, 200]  # Purple-ish  
            else:  # RIGHT
                mock_image[:, :] = [200, 150, 100]  # Orange-ish
            
            # Add text to indicate the pose
            cv2.putText(mock_image, pose_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', mock_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            pose_images.append({
                "pose_name": pose_name,
                "image": f"data:image/jpeg;base64,{image_base64}",
                "quality_score": 0.85 - (i * 0.05),  # Decreasing quality
                "timestamp": time.time()
            })
        
        return {
            "status": "success",
            "message": f"Collected {len(pose_images)} poses for {person_name}",
            "person_name": person_name,
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "has_video_stream": current_camera is not None,  # Indicate that video stream is available
            "stream_url": "/api/enrollment/video-stream" if current_camera is not None else None,  # URL for video stream
            "processing_stats": {
                "frames_processed": 150,
                "processing_time": 3.2,
                "poses_collected": [p["pose_name"] for p in pose_images]
            }
        }
        
    except Exception as e:
        print(f"Error in video enrollment: {e}")
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
                
                # Add actual face detection rectangles
                try:
                    from app.services.face_recognition_service import FaceDetector
                    
                    # Initialize face detector (you might want to reuse one from app state)
                    if not hasattr(generate_frames, 'face_detector'):
                        generate_frames.face_detector = FaceDetector(confidence_threshold=0.5)
                    
                    # Detect actual faces
                    detections = generate_frames.face_detector.detect(frame)
                    
                    # Draw rectangles for all detected faces
                    for detection in detections:
                        x, y, w, h = detection.bbox
                        confidence = detection.confidence
                        
                        # Draw face bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add confidence label
                        cv2.putText(frame, f"Face: {confidence:.2f}", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # If no faces detected, show a message
                    if len(detections) == 0:
                        cv2.putText(frame, "No faces detected", (10, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                except Exception as face_error:
                    print(f"Face detection error: {face_error}")
                    # Fallback to center rectangle if face detection fails
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    rect_size = 150
                    cv2.rectangle(frame, 
                                (center_x - rect_size, center_y - rect_size), 
                                (center_x + rect_size, center_y + rect_size), 
                                (255, 0, 0), 2)  # Red for fallback
                    cv2.putText(frame, "Face detection unavailable", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
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