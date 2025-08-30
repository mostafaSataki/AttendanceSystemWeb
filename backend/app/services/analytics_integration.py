#!/usr/bin/env python3
"""
Analytics Integration Service
Connects RTSP manager with face recognition analytics
"""

import cv2
import numpy as np
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

from .rtsp_manager import rtsp_manager
from .face_recognition_service import FaceRecognitionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalyticsResult:
    """Analytics processing result"""
    camera_id: str
    timestamp: datetime
    detections: List[Dict[str, Any]]
    total_faces: int
    recognized_faces: int
    frame_processed: bool = True

class AnalyticsIntegration:
    """Integrates RTSP streams with face recognition analytics"""
    
    def __init__(self):
        self.face_service = FaceRecognitionService()
        self.analytics_threads: Dict[str, threading.Thread] = {}
        self.analytics_active: Dict[str, bool] = {}
        self.processing_stats: Dict[str, Dict] = {}
        
        # Analytics configuration
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.frame_counters: Dict[str, int] = {}
        
    def start_analytics_for_camera(self, camera_id: str) -> Dict[str, Any]:
        """Start analytics processing for a specific camera"""
        try:
            if camera_id in self.analytics_active and self.analytics_active[camera_id]:
                return {"success": False, "error": "Analytics already active for this camera"}
            
            # Register as consumer with RTSP manager
            def frame_consumer(cam_id: str, frame: np.ndarray):
                self._process_analytics_frame(cam_id, frame)
            
            success = rtsp_manager.register_analytics_consumer(camera_id, frame_consumer)
            
            if success:
                self.analytics_active[camera_id] = True
                self.frame_counters[camera_id] = 0
                self.processing_stats[camera_id] = {
                    'frames_processed': 0,
                    'faces_detected': 0,
                    'faces_recognized': 0,
                    'start_time': datetime.now(),
                    'last_processing_time': None,
                    'avg_processing_time': 0.0,
                    'processing_times': []
                }
                
                logger.info(f"Started analytics for camera {camera_id}")
                return {
                    "success": True,
                    "message": f"Analytics started for camera {camera_id}",
                    "camera_id": camera_id
                }
            else:
                return {"success": False, "error": "Failed to register analytics consumer"}
                
        except Exception as e:
            logger.error(f"Error starting analytics for camera {camera_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_analytics_frame(self, camera_id: str, frame: np.ndarray):
        """Process frame for analytics (called by RTSP manager)"""
        try:
            if not self.analytics_active.get(camera_id, False):
                return
            
            # Frame skipping for performance
            self.frame_counters[camera_id] += 1
            if self.frame_counters[camera_id] % self.process_every_n_frames != 0:
                return
            
            start_time = time.time()
            
            # Process frame with face recognition
            results = self._analyze_frame(camera_id, frame)
            
            # Update statistics
            processing_time = time.time() - start_time
            stats = self.processing_stats[camera_id]
            stats['frames_processed'] += 1
            stats['faces_detected'] += results.total_faces
            stats['faces_recognized'] += results.recognized_faces
            stats['last_processing_time'] = processing_time
            
            # Rolling average of processing times
            stats['processing_times'].append(processing_time)
            if len(stats['processing_times']) > 100:
                stats['processing_times'].pop(0)
            
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            
            # Log results periodically
            if stats['frames_processed'] % 30 == 0:  # Every 30 processed frames
                logger.info(f"Camera {camera_id}: {stats['frames_processed']} frames, "
                           f"{stats['faces_detected']} faces detected, "
                           f"{stats['faces_recognized']} recognized, "
                           f"avg: {stats['avg_processing_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing analytics frame for {camera_id}: {e}")
    
    def _analyze_frame(self, camera_id: str, frame: np.ndarray) -> AnalyticsResult:
        """Analyze frame for face recognition"""
        try:
            # Use face recognition service for processing
            detections = []
            total_faces = 0
            recognized_faces = 0
            
            # Simple face detection using OpenCV (placeholder for full integration)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            total_faces = len(faces)
            
            for (x, y, w, h) in faces:
                detection = {
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,  # Placeholder
                    'person_id': None,
                    'person_name': 'Unknown',
                    'recognition_confidence': 0.0
                }
                
                # Here you would integrate actual face recognition
                # For now, placeholder logic
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    # Placeholder: In real implementation, use face recognition service
                    # recognition_result = self.face_service.recognize_face(face_roi)
                    # if recognition_result.get('recognized', False):
                    #     detection.update(recognition_result)
                    #     recognized_faces += 1
                    pass
                
                detections.append(detection)
            
            return AnalyticsResult(
                camera_id=camera_id,
                timestamp=datetime.now(),
                detections=detections,
                total_faces=total_faces,
                recognized_faces=recognized_faces
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return AnalyticsResult(
                camera_id=camera_id,
                timestamp=datetime.now(),
                detections=[],
                total_faces=0,
                recognized_faces=0,
                frame_processed=False
            )
    
    def stop_analytics_for_camera(self, camera_id: str) -> Dict[str, Any]:
        """Stop analytics processing for a specific camera"""
        try:
            if camera_id in self.analytics_active:
                self.analytics_active[camera_id] = False
                
                logger.info(f"Stopped analytics for camera {camera_id}")
                return {
                    "success": True,
                    "message": f"Analytics stopped for camera {camera_id}",
                    "camera_id": camera_id,
                    "final_stats": self.processing_stats.get(camera_id, {})
                }
            else:
                return {"success": False, "error": "Analytics not active for this camera"}
                
        except Exception as e:
            logger.error(f"Error stopping analytics for camera {camera_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_analytics_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """Get analytics statistics"""
        try:
            if camera_id:
                if camera_id in self.processing_stats:
                    stats = self.processing_stats[camera_id].copy()
                    stats['is_active'] = self.analytics_active.get(camera_id, False)
                    stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
                    return {
                        "success": True,
                        "camera_id": camera_id,
                        "stats": stats
                    }
                else:
                    return {"success": False, "error": "No stats available for this camera"}
            else:
                # Return stats for all cameras
                all_stats = {}
                for cam_id in self.processing_stats:
                    all_stats[cam_id] = self.get_analytics_stats(cam_id).get('stats', {})
                
                return {
                    "success": True,
                    "all_camera_stats": all_stats,
                    "total_active_cameras": sum(1 for active in self.analytics_active.values() if active)
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics stats: {e}")
            return {"success": False, "error": str(e)}
    
    async def setup_camera_with_analytics(self, camera_id: str, rtsp_url: str, name: str) -> Dict[str, Any]:
        """Complete setup: Add camera, start stream, and begin analytics"""
        try:
            # Add stream to RTSP manager
            stream_result = await rtsp_manager.add_stream(camera_id, rtsp_url, name)
            if not stream_result['success']:
                return stream_result
            
            # Start stream ingest
            ingest_result = await rtsp_manager.start_stream_ingest(camera_id)
            if not ingest_result['success']:
                return ingest_result
            
            # Wait a moment for stream to stabilize
            await asyncio.sleep(2)
            
            # Start analytics
            analytics_result = self.start_analytics_for_camera(camera_id)
            if not analytics_result['success']:
                return analytics_result
            
            logger.info(f"Complete camera setup finished for {camera_id}")
            return {
                "success": True,
                "camera_id": camera_id,
                "message": f"Camera {name} fully configured with analytics",
                "stream_active": True,
                "analytics_active": True
            }
            
        except Exception as e:
            logger.error(f"Error in complete camera setup: {e}")
            return {"success": False, "error": str(e)}

# Global analytics integration instance
analytics_integration = AnalyticsIntegration()