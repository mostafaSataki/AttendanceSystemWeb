"""
Video Preview API endpoint for showing video files during enrollment
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
import cv2
import os
from pathlib import Path
import mimetypes

router = APIRouter()

@router.get("/stream")
async def stream_video_file(video_path: str = Query(...)):
    """Stream video file for preview"""
    try:
        # Validate file exists and is accessible
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
        
        # Check if it's a video file
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type or not mime_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File is not a video")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file")
        
        def generate_frames():
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Resize frame for web streaming
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control frame rate (adjust as needed)
                import time
                time.sleep(0.033)  # ~30 FPS
        
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as e:
        print(f"Error streaming video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'cap' in locals():
            cap.release()

@router.get("/info")
async def get_video_info(video_path: str = Query(...)):
    """Get video file information"""
    try:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "success": True,
            "video_path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
        }
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        raise HTTPException(status_code=500, detail=str(e))