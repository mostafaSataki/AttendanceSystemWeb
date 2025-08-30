#!/usr/bin/env python3
"""
RTSP Stream Manager
Handles single RTSP ingest with multi-consumer frame distribution
"""

import asyncio
import subprocess
import threading
import time
import cv2
import numpy as np
import os
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RTSPStream:
    """RTSP stream configuration"""
    camera_id: str
    rtsp_url: str
    name: str
    is_active: bool = False
    ffmpeg_process: Optional[subprocess.Popen] = None
    analytics_enabled: bool = True
    last_frame_time: Optional[datetime] = None

@dataclass
class PlaybackSession:
    """Client playback session"""
    session_id: str
    camera_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    hls_process: Optional[subprocess.Popen] = None
    is_active: bool = False

class RTSPManager:
    """Manages RTSP streams with single ingest and multi-consumer distribution"""
    
    def __init__(self):
        self.streams: Dict[str, RTSPStream] = {}
        self.analytics_consumers: Dict[str, List[Callable]] = {}
        self.playback_sessions: Dict[str, PlaybackSession] = {}
        self.frame_buffers: Dict[str, np.ndarray] = {}
        self.buffer_locks: Dict[str, threading.Lock] = {}
        
        # Configuration
        self.hls_output_dir = Path("hls_output")
        self.hls_output_dir.mkdir(exist_ok=True)
        self.jwt_secret = secrets.token_urlsafe(32)
        self.session_duration = timedelta(minutes=30)
        
        # Start cleanup task
        self._cleanup_task = None
        
    async def add_stream(self, camera_id: str, rtsp_url: str, name: str) -> Dict[str, Any]:
        """Add new RTSP stream configuration"""
        try:
            if camera_id in self.streams:
                return {"success": False, "error": "Camera ID already exists"}
            
            stream = RTSPStream(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                name=name
            )
            
            self.streams[camera_id] = stream
            self.analytics_consumers[camera_id] = []
            self.buffer_locks[camera_id] = threading.Lock()
            
            logger.info(f"Added RTSP stream: {camera_id} - {name}")
            return {
                "success": True,
                "camera_id": camera_id,
                "message": f"Stream {name} added successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding stream {camera_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_stream_ingest(self, camera_id: str) -> Dict[str, Any]:
        """Start FFmpeg ingest for RTSP stream (single process per camera)"""
        try:
            if camera_id not in self.streams:
                return {"success": False, "error": "Camera not found"}
            
            stream = self.streams[camera_id]
            
            if stream.is_active:
                return {"success": False, "error": "Stream already active"}
            
            # Start FFmpeg relay process
            success = await self._start_ffmpeg_relay(stream)
            
            if success:
                stream.is_active = True
                # Start frame reading thread for analytics
                threading.Thread(
                    target=self._frame_reader_thread,
                    args=(stream,),
                    daemon=True
                ).start()
                
                logger.info(f"Started RTSP ingest for {camera_id}")
                return {
                    "success": True,
                    "camera_id": camera_id,
                    "message": "Stream ingest started"
                }
            else:
                return {"success": False, "error": "Failed to start FFmpeg process"}
                
        except Exception as e:
            logger.error(f"Error starting stream {camera_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _start_ffmpeg_relay(self, stream: RTSPStream) -> bool:
        """Start FFmpeg process for RTSP relay"""
        try:
            # Create named pipe for frame extraction
            pipe_path = f"/tmp/camera_{stream.camera_id}_frames"
            
            # FFmpeg command: RTSP input -> Multiple outputs
            cmd = [
                'ffmpeg',
                '-i', stream.rtsp_url,
                '-c:v', 'copy',  # Copy H.264 stream (no re-encoding)
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                pipe_path,  # Raw frames for analytics
            ]
            
            # Start process
            stream.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            # Wait a moment to ensure process started
            await asyncio.sleep(1)
            
            if stream.ffmpeg_process.poll() is None:
                logger.info(f"FFmpeg relay started for {stream.camera_id}")
                return True
            else:
                logger.error(f"FFmpeg process failed for {stream.camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting FFmpeg relay: {e}")
            return False
    
    def _frame_reader_thread(self, stream: RTSPStream):
        """Thread to read frames from FFmpeg output for analytics"""
        logger.info(f"Started frame reader thread for {stream.camera_id}")
        
        # Use OpenCV to read from RTSP (alternative approach)
        cap = cv2.VideoCapture(stream.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)  # Minimize buffer
        
        try:
            while stream.is_active:
                ret, frame = cap.read()
                if ret:
                    # Update shared frame buffer
                    with self.buffer_locks[stream.camera_id]:
                        self.frame_buffers[stream.camera_id] = frame.copy()
                    
                    stream.last_frame_time = datetime.now()
                    
                    # Distribute frame to analytics consumers
                    for consumer in self.analytics_consumers[stream.camera_id]:
                        try:
                            consumer(stream.camera_id, frame)
                        except Exception as e:
                            logger.error(f"Analytics consumer error: {e}")
                else:
                    logger.warning(f"Failed to read frame from {stream.camera_id}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Frame reader error for {stream.camera_id}: {e}")
        finally:
            cap.release()
            logger.info(f"Frame reader stopped for {stream.camera_id}")
    
    def register_analytics_consumer(self, camera_id: str, consumer: Callable):
        """Register analytics service as frame consumer"""
        if camera_id in self.analytics_consumers:
            self.analytics_consumers[camera_id].append(consumer)
            logger.info(f"Registered analytics consumer for {camera_id}")
            return True
        return False
    
    async def create_playback_session(self, camera_id: str, user_id: str = None) -> Dict[str, Any]:
        """Create secure playback session with token"""
        try:
            if camera_id not in self.streams:
                return {"success": False, "error": "Camera not found"}
            
            stream = self.streams[camera_id]
            if not stream.is_active:
                return {"success": False, "error": "Stream not active"}
            
            # Generate session
            session_id = secrets.token_urlsafe(16)
            expires_at = datetime.now() + self.session_duration
            
            # Create JWT token
            token_payload = {
                "session_id": session_id,
                "camera_id": camera_id,
                "user_id": user_id,
                "exp": expires_at.timestamp(),
                "iat": datetime.now().timestamp()
            }
            
            token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")
            
            # Create session
            session = PlaybackSession(
                session_id=session_id,
                camera_id=camera_id,
                token=token,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            self.playback_sessions[session_id] = session
            
            logger.info(f"Created playback session {session_id} for camera {camera_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "token": token,
                "hls_url": f"/api/stream/hls/{session_id}/playlist.m3u8",
                "expires_at": expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating playback session: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_hls_playback(self, session_id: str) -> Dict[str, Any]:
        """Start on-demand HLS generation for playback session"""
        try:
            if session_id not in self.playback_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.playback_sessions[session_id]
            
            if session.expires_at < datetime.now():
                return {"success": False, "error": "Session expired"}
            
            if session.is_active:
                return {"success": True, "message": "HLS already active"}
            
            # Get stream
            stream = self.streams[session.camera_id]
            
            # Create HLS output directory for this session
            session_dir = self.hls_output_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            # FFmpeg command for HLS generation
            hls_cmd = [
                'ffmpeg',
                '-i', stream.rtsp_url,
                '-c:v', 'libx264',  # Re-encode for HLS compatibility
                '-preset', 'ultrafast',  # Low latency
                '-tune', 'zerolatency',
                '-g', '30',  # GOP size
                '-keyint_min', '30',
                '-hls_time', '2',  # 2-second segments
                '-hls_list_size', '5',  # Keep 5 segments
                '-hls_flags', 'delete_segments+independent_segments',
                '-hls_segment_type', 'fmp4',  # fMP4 segments
                '-f', 'hls',
                str(session_dir / 'playlist.m3u8')
            ]
            
            # Start HLS process
            session.hls_process = subprocess.Popen(
                hls_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for playlist to be created
            await asyncio.sleep(3)
            
            if session.hls_process.poll() is None:
                session.is_active = True
                logger.info(f"Started HLS playback for session {session_id}")
                return {
                    "success": True,
                    "message": "HLS playback started",
                    "playlist_url": f"/api/stream/hls/{session_id}/playlist.m3u8"
                }
            else:
                logger.error(f"Failed to start HLS for session {session_id}")
                return {"success": False, "error": "Failed to start HLS generation"}
                
        except Exception as e:
            logger.error(f"Error starting HLS playback: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_hls_playback(self, session_id: str) -> Dict[str, Any]:
        """Stop HLS playback and cleanup"""
        try:
            if session_id not in self.playback_sessions:
                return {"success": False, "error": "Session not found"}
            
            session = self.playback_sessions[session_id]
            
            if session.hls_process and session.hls_process.poll() is None:
                session.hls_process.terminate()
                await asyncio.sleep(1)
                if session.hls_process.poll() is None:
                    session.hls_process.kill()
            
            session.is_active = False
            
            # Cleanup HLS files
            session_dir = self.hls_output_dir / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
            
            logger.info(f"Stopped HLS playback for session {session_id}")
            
            return {"success": True, "message": "HLS playback stopped"}
            
        except Exception as e:
            logger.error(f"Error stopping HLS playback: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT session token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload.get("session_id")
            
            if session_id in self.playback_sessions:
                session = self.playback_sessions[session_id]
                if session.expires_at > datetime.now():
                    return payload
                    
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    async def cleanup_expired_sessions(self):
        """Cleanup expired playback sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.playback_sessions.items():
            if session.expires_at < current_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.stop_hls_playback(session_id)
            del self.playback_sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    async def get_stream_status(self, camera_id: str = None) -> Dict[str, Any]:
        """Get stream status information"""
        if camera_id:
            if camera_id in self.streams:
                stream = self.streams[camera_id]
                return {
                    "camera_id": camera_id,
                    "name": stream.name,
                    "rtsp_url": stream.rtsp_url,
                    "is_active": stream.is_active,
                    "last_frame_time": stream.last_frame_time.isoformat() if stream.last_frame_time else None,
                    "analytics_consumers": len(self.analytics_consumers.get(camera_id, [])),
                    "has_current_frame": camera_id in self.frame_buffers
                }
            else:
                return {"error": "Camera not found"}
        else:
            # Return all streams status
            streams_status = {}
            for cam_id in self.streams:
                streams_status[cam_id] = await self.get_stream_status(cam_id)
            
            return {
                "streams": streams_status,
                "active_sessions": len([s for s in self.playback_sessions.values() if s.is_active]),
                "total_sessions": len(self.playback_sessions)
            }

# Global RTSP manager instance
rtsp_manager = RTSPManager()