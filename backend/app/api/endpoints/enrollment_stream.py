"""
Real-time Enrollment Video Streaming Endpoint
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
import base64
import json

from app.services.backend_enrollment_service import BackendEnrollmentService
from app.services.face_engine.head_pose_estimator import PoseState

router = APIRouter()

# Global state for streaming sessions
streaming_sessions: Dict[str, Dict[str, Any]] = {}
enrollment_service = BackendEnrollmentService()

class EnrollmentStreamer:
    def __init__(self, session_id: str, source: str, source_config: Dict):
        self.session_id = session_id
        self.source = source
        self.source_config = source_config
        self.cap = None
        self.is_streaming = False
        self.enrollment_state = None
        self.collected_poses = {}
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def start_capture(self):
        """Start video capture"""
        try:
            if self.source == "video":
                video_path = self.source_config.get("video_path")
                self.cap = cv2.VideoCapture(video_path)
            else:  # camera
                camera_source = self.source_config.get("camera_index", 0)
                if "rtsp_url" in self.source_config:
                    camera_source = self.source_config["rtsp_url"]
                self.cap = cv2.VideoCapture(camera_source)
                
            if not self.cap or not self.cap.isOpened():
                return False
                
            # Set camera properties
            if self.source == "camera":
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
            self.is_streaming = True
            self.enrollment_state = enrollment_service.initialize_enrollment_state(
                f"stream_{self.session_id}"
            )
            return True
            
        except Exception as e:
            print(f"❌ Error starting capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop video capture"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def process_frame(self):
        """Process current frame for enrollment"""
        if not self.cap or not self.is_streaming:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Store current frame for streaming
        with self.frame_lock:
            self.current_frame = frame.copy()
            
        # Process for enrollment
        if self.enrollment_state:
            self.enrollment_state['frame_count'] += 1
            frame_result = enrollment_service.process_frame_for_enrollment(
                frame, self.enrollment_state
            )
            
            # Draw pose information on frame
            if frame_result.get('candidates'):
                candidate = frame_result['candidates'][0]
                bbox = candidate.get('bbox', [0, 0, 50, 50])
                x, y, w, h = bbox
                
                # Draw face rectangle
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                
                # Draw pose information
                pose_detected = frame_result.get('pose_detected', 'UNKNOWN')
                quality = candidate.get('quality_score', 0)
                
                cv2.putText(frame, f"Pose: {pose_detected}", (int(x), int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Quality: {quality:.2f}", (int(x), int(y + h + 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show collected poses
            collected_poses = list(self.enrollment_state.get('collected_poses', {}).keys())
            poses_text = f"Collected: {len(collected_poses)}/5 [{', '.join([p.value if hasattr(p, 'value') else str(p) for p in collected_poses])}]"
            cv2.putText(frame, poses_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update stored poses
            self.collected_poses = self.enrollment_state.get('collected_poses', {})
            
        return frame
    
    def get_stream_frame(self):
        """Get current frame for streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

@router.post("/start-stream")
async def start_enrollment_stream(request: Request):
    """Start enrollment video stream"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        session_id = f"enrollment_{int(time.time())}"
        
        print(f"🎥 Starting enrollment stream: {session_id}")
        
        # Create streamer
        streamer = EnrollmentStreamer(session_id, source, source_config)
        
        if not streamer.start_capture():
            raise HTTPException(status_code=400, detail="Failed to start video capture")
            
        # Store session
        streaming_sessions[session_id] = {
            'streamer': streamer,
            'started_at': time.time()
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "stream_url": f"/api/enrollment-stream/video/{session_id}",
            "message": "Video stream started successfully"
        }
        
    except Exception as e:
        print(f"❌ Error starting stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{session_id}")
async def get_video_stream(session_id: str):
    """Get video stream for enrollment session"""
    if session_id not in streaming_sessions:
        raise HTTPException(status_code=404, detail="Stream session not found")
    
    streamer = streaming_sessions[session_id]['streamer']
    
    def generate_frames():
        while streamer.is_streaming:
            try:
                frame = streamer.process_frame()
                if frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in frame generation: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/stop-stream/{session_id}")
async def stop_enrollment_stream(session_id: str):
    """Stop enrollment video stream and return collected poses"""
    if session_id not in streaming_sessions:
        raise HTTPException(status_code=404, detail="Stream session not found")
    
    try:
        session = streaming_sessions[session_id]
        streamer = session['streamer']
        
        # Get collected poses before stopping
        collected_poses = streamer.collected_poses.copy()
        
        # Stop streaming
        streamer.stop_capture()
        
        # Remove session
        del streaming_sessions[session_id]
        
        # Convert collected poses to response format
        pose_images = []
        for pose_name, pose_data in collected_poses.items():
            if "aligned_face" in pose_data:
                # Convert aligned face image to base64
                _, buffer = cv2.imencode('.jpg', pose_data["aligned_face"])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                pose_key = pose_name.value if hasattr(pose_name, 'value') else str(pose_name)
                pose_images.append({
                    "pose_name": pose_key,
                    "image": f"data:image/jpeg;base64,{image_base64}",
                    "quality_score": pose_data.get("quality_score", 0.0),
                    "timestamp": pose_data.get("timestamp"),
                    "frame_number": pose_data.get("frame_number")
                })
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Stream stopped. Collected {len(pose_images)} poses.",
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "processing_stats": {
                "frames_processed": streamer.enrollment_state.get('frame_count', 0) if streamer.enrollment_state else 0,
                "session_duration": time.time() - session['started_at']
            }
        }
        
    except Exception as e:
        print(f"❌ Error stopping stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}")
async def get_stream_status(session_id: str):
    """Get current status of enrollment stream"""
    if session_id not in streaming_sessions:
        raise HTTPException(status_code=404, detail="Stream session not found")
    
    session = streaming_sessions[session_id]
    streamer = session['streamer']
    
    collected_poses = list(streamer.collected_poses.keys())
    pose_names = [p.value if hasattr(p, 'value') else str(p) for p in collected_poses]
    
    return {
        "session_id": session_id,
        "is_streaming": streamer.is_streaming,
        "collected_poses": len(collected_poses),
        "pose_names": pose_names,
        "frame_count": streamer.enrollment_state.get('frame_count', 0) if streamer.enrollment_state else 0,
        "session_duration": time.time() - session['started_at']
    }

@router.get("/sessions")
async def get_active_sessions():
    """Get all active streaming sessions"""
    active_sessions = {}
    current_time = time.time()
    
    for session_id, session_data in streaming_sessions.items():
        active_sessions[session_id] = {
            "session_id": session_id,
            "is_streaming": session_data['streamer'].is_streaming,
            "duration": current_time - session_data['started_at'],
            "collected_poses": len(session_data['streamer'].collected_poses)
        }
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": active_sessions
    }

# Cleanup function to remove old sessions
async def cleanup_old_sessions():
    """Remove sessions older than 10 minutes"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session_data in streaming_sessions.items():
        if current_time - session_data['started_at'] > 600:  # 10 minutes
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        try:
            streaming_sessions[session_id]['streamer'].stop_capture()
            del streaming_sessions[session_id]
            print(f"🧹 Cleaned up old session: {session_id}")
        except:
            pass