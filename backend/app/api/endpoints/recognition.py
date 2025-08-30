"""
Face Recognition API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import io
import base64
from PIL import Image
import numpy as np
import cv2
import asyncio
from datetime import datetime

from app.core.config import settings

router = APIRouter()


@router.post("/start")
async def start_recognition(request: Request, source: str = "camera", source_config: Dict[str, Any] = None):
    """Start face recognition session"""
    try:
        face_service = request.app.state.face_service
        
        config = source_config or {}
        if source == "camera":
            config.setdefault("camera_index", 0)
        elif source == "rtsp":
            if not config.get("rtsp_url"):
                raise HTTPException(status_code=400, detail="RTSP URL required")
        elif source == "video":
            if not config.get("video_path"):
                raise HTTPException(status_code=400, detail="Video path required")
        
        result = await face_service.start_recognition(source, config)
        
        return {
            "status": "success",
            "message": "Recognition started",
            "session_id": result.get("session_id"),
            "source": source,
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start recognition: {str(e)}")


@router.post("/stop")
async def stop_recognition(request: Request):
    """Stop face recognition session"""
    try:
        face_service = request.app.state.face_service
        result = await face_service.stop_recognition()
        return {
            "status": "success", 
            "message": "Recognition stopped",
            "session_stats": result.get("stats", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop recognition: {str(e)}")


@router.get("/stream")
async def get_video_stream(request: Request):
    """Get live video stream with face recognition"""
    try:
        face_service = request.app.state.face_service
        
        def generate_stream():
            try:
                while True:
                    frame_data = face_service.get_current_frame()
                    if frame_data:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame_data['frame'])
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send a black frame if no data available
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        _, buffer = cv2.imencode('.jpg', black_frame)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Small delay to control frame rate
                    import time
                    time.sleep(1/30)  # ~30 FPS
                    
            except Exception as e:
                print(f"Stream error: {e}")
        
        return StreamingResponse(
            generate_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")


@router.get("/detections")
async def get_current_detections(request: Request):
    """Get current face detections and recognitions"""
    try:
        face_service = request.app.state.face_service
        detections = await face_service.get_current_detections()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get detections: {str(e)}")


@router.post("/recognize-image")
async def recognize_uploaded_image(file: UploadFile = File(...)):
    """Recognize faces in an uploaded image"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
            
        # Process with face service
        # Note: This would need to be implemented in the face service
        result = await recognize_faces_in_image(image_cv)
        
        return {
            "status": "success",
            "faces_detected": len(result.get("faces", [])),
            "recognitions": result.get("recognitions", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recognize faces: {str(e)}")


@router.get("/session/status")
async def get_recognition_status(request: Request):
    """Get current recognition session status"""
    try:
        face_service = request.app.state.face_service
        status_info = await face_service.get_recognition_status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/transactions")
async def get_recent_transactions(request: Request, limit: int = Query(10, ge=1, le=100)):
    """Get recent recognition transactions/events"""
    try:
        face_service = request.app.state.face_service
        transactions = await face_service.get_recent_transactions(limit)
        
        return {
            "status": "success",
            "transactions": transactions,
            "total": len(transactions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transactions: {str(e)}")


async def recognize_faces_in_image(image_cv):
    """Helper function to recognize faces in image"""
    # This would use the face recognition service to process the image
    # For now, return mock data
    return {
        "faces": [],
        "recognitions": []
    }