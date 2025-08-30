"""
RTSP Streaming API endpoints
Handles camera management, stream control, and secure playback
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pathlib import Path
import os
import logging

from app.services.rtsp_manager import rtsp_manager
from app.services.analytics_integration import analytics_integration

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

class CameraConfig(BaseModel):
    camera_id: str
    rtsp_url: str
    name: str
    enable_analytics: bool = True

class PlaybackRequest(BaseModel):
    camera_id: str
    user_id: Optional[str] = None

@router.post("/cameras/add")
async def add_camera(config: CameraConfig):
    """Add new RTSP camera configuration"""
    try:
        if config.enable_analytics:
            result = await analytics_integration.setup_camera_with_analytics(
                config.camera_id, 
                config.rtsp_url, 
                config.name
            )
        else:
            # Add stream only (no analytics)
            stream_result = await rtsp_manager.add_stream(
                config.camera_id, 
                config.rtsp_url, 
                config.name
            )
            if stream_result['success']:
                ingest_result = await rtsp_manager.start_stream_ingest(config.camera_id)
                result = ingest_result
            else:
                result = stream_result
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding camera: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add camera: {str(e)}")

@router.get("/cameras/status")
async def get_cameras_status(camera_id: Optional[str] = Query(None)):
    """Get status of cameras and streams"""
    try:
        stream_status = await rtsp_manager.get_stream_status(camera_id)
        
        if camera_id:
            # Include analytics stats for specific camera
            analytics_stats = analytics_integration.get_analytics_stats(camera_id)
            return {
                **stream_status,
                "analytics": analytics_stats if analytics_stats['success'] else None
            }
        else:
            # Include analytics stats for all cameras
            analytics_stats = analytics_integration.get_analytics_stats()
            return {
                **stream_status,
                "analytics_summary": analytics_stats if analytics_stats['success'] else None
            }
            
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/cameras/{camera_id}/start")
async def start_camera_stream(camera_id: str):
    """Start RTSP stream ingest for specific camera"""
    try:
        result = await rtsp_manager.start_stream_ingest(camera_id)
        return result
    except Exception as e:
        logger.error(f"Error starting camera stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

@router.post("/cameras/{camera_id}/analytics/start")
async def start_camera_analytics(camera_id: str):
    """Start analytics processing for specific camera"""
    try:
        result = analytics_integration.start_analytics_for_camera(camera_id)
        return result
    except Exception as e:
        logger.error(f"Error starting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analytics: {str(e)}")

@router.post("/cameras/{camera_id}/analytics/stop")
async def stop_camera_analytics(camera_id: str):
    """Stop analytics processing for specific camera"""
    try:
        result = analytics_integration.stop_analytics_for_camera(camera_id)
        return result
    except Exception as e:
        logger.error(f"Error stopping analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop analytics: {str(e)}")

@router.get("/analytics/stats")
async def get_analytics_stats(camera_id: Optional[str] = Query(None)):
    """Get analytics processing statistics"""
    try:
        result = analytics_integration.get_analytics_stats(camera_id)
        return result
    except Exception as e:
        logger.error(f"Error getting analytics stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/playback/create-session")
async def create_playback_session(request: PlaybackRequest):
    """Create secure playback session with token"""
    try:
        result = await rtsp_manager.create_playback_session(
            request.camera_id, 
            request.user_id
        )
        return result
    except Exception as e:
        logger.error(f"Error creating playback session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.post("/playback/{session_id}/start")
async def start_playback(session_id: str):
    """Start HLS playback for session"""
    try:
        result = await rtsp_manager.start_hls_playback(session_id)
        return result
    except Exception as e:
        logger.error(f"Error starting playback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start playback: {str(e)}")

@router.post("/playback/{session_id}/stop")
async def stop_playback(session_id: str):
    """Stop HLS playback for session"""
    try:
        result = await rtsp_manager.stop_hls_playback(session_id)
        return result
    except Exception as e:
        logger.error(f"Error stopping playback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop playback: {str(e)}")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token for HLS access"""
    token = credentials.credentials
    payload = rtsp_manager.verify_session_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

@router.get("/hls/{session_id}/playlist.m3u8")
async def serve_hls_playlist(
    session_id: str, 
    token_payload: dict = Depends(verify_token)
):
    """Serve HLS playlist (protected endpoint)"""
    try:
        # Verify session matches token
        if token_payload.get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="Session ID mismatch")
        
        playlist_path = Path("hls_output") / session_id / "playlist.m3u8"
        
        if not playlist_path.exists():
            raise HTTPException(status_code=404, detail="Playlist not found")
        
        return FileResponse(
            playlist_path,
            media_type="application/vnd.apple.mpegurl",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving HLS playlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve playlist")

@router.get("/hls/{session_id}/{segment_name}")
async def serve_hls_segment(
    session_id: str, 
    segment_name: str,
    token_payload: dict = Depends(verify_token)
):
    """Serve HLS segment (protected endpoint)"""
    try:
        # Verify session matches token
        if token_payload.get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="Session ID mismatch")
        
        # Security: Only allow .m4s and .mp4 files
        if not (segment_name.endswith('.m4s') or segment_name.endswith('.mp4')):
            raise HTTPException(status_code=403, detail="Invalid segment type")
        
        segment_path = Path("hls_output") / session_id / segment_name
        
        if not segment_path.exists():
            raise HTTPException(status_code=404, detail="Segment not found")
        
        return FileResponse(
            segment_path,
            media_type="video/mp4",
            headers={
                "Cache-Control": "public, max-age=31536000",  # Cache segments
                "Accept-Ranges": "bytes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving HLS segment: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve segment")

@router.get("/playback/sessions")
async def list_playback_sessions():
    """List active playback sessions (admin endpoint)"""
    try:
        sessions = []
        for session_id, session in rtsp_manager.playback_sessions.items():
            sessions.append({
                "session_id": session_id,
                "camera_id": session.camera_id,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "is_active": session.is_active
            })
        
        return {
            "success": True,
            "sessions": sessions,
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s["is_active"]])
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

@router.post("/maintenance/cleanup-sessions")
async def cleanup_expired_sessions():
    """Manual cleanup of expired sessions"""
    try:
        await rtsp_manager.cleanup_expired_sessions()
        return {
            "success": True,
            "message": "Expired sessions cleaned up"
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

# Health check endpoint
@router.get("/health")
async def streaming_health():
    """Streaming service health check"""
    try:
        total_streams = len(rtsp_manager.streams)
        active_streams = len([s for s in rtsp_manager.streams.values() if s.is_active])
        active_sessions = len([s for s in rtsp_manager.playback_sessions.values() if s.is_active])
        
        return {
            "status": "healthy",
            "service": "rtsp_streaming",
            "total_cameras": total_streams,
            "active_streams": active_streams,
            "active_playback_sessions": active_sessions,
            "analytics_active": len([active for active in analytics_integration.analytics_active.values() if active])
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")