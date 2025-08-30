#!/usr/bin/env python3
"""
Face Recognition Service - Backend service for real-time face recognition
Adapted from the original face_recognition.py to work with Qt GUI
"""

import cv2
import numpy as np
import time
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from collections import deque
import threading
import queue
import math
import sys

# Console colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("⚠️ DeepFace not available. Face recognition will be limited.")
    DEEPFACE_AVAILABLE = False

@dataclass
class Detection:
    """Face detection data structure"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    class_id: int = 0  # Face class

@dataclass
class Track:
    """Track data structure with face ID"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: deque = None
    state: str = "tentative"  # tentative, confirmed, lost
    min_hits: int = 3  # Minimum hits to confirm track
    face_id: str = "Unknown"  # Identified face ID
    face_confidence: float = 0.0  # Face identification confidence
    last_verification_frame: int = 0  # Last frame when face was verified
    initial_position: Tuple[int, int] = None  # Initial position when track was created
    has_moved_significantly: bool = False  # Whether track has moved >10% of height
    face_id_votes: Dict[str, int] = None  # Vote counter for each face ID
    permanent_face_id: str = None  # Permanently assigned face ID (after 5 votes)
    verification_count: int = 0  # Total number of verifications performed
    
    # Robustness features
    occlusion_count: int = 0  # Number of times track was occluded
    quality_score: float = 1.0  # Track quality score (0.0 to 1.0)
    velocity: Tuple[float, float] = (0.0, 0.0)  # Current velocity (dx, dy)
    predicted_bbox: Tuple[int, int, int, int] = None  # Predicted next position
    stability_score: float = 1.0  # How stable the track is (based on consistency)
    association_failures: int = 0  # Number of consecutive failed associations
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = deque(maxlen=30)  # Keep last 30 positions
        if self.face_id_votes is None:
            self.face_id_votes = {}  # Initialize vote counter
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.state == "confirmed" or self.hits >= self.min_hits
    
    def mark_confirmed(self):
        """Mark track as confirmed"""
        if self.hits >= self.min_hits:
            self.state = "confirmed"
    
    def is_near_border(self, frame_width: int, frame_height: int, border_threshold: int = 100) -> bool:
        """Check if track is near frame border (increased threshold from 50 to 100 pixels)"""
        x, y, w, h = self.bbox
        return (x <= border_threshold or 
                y <= border_threshold or 
                x + w >= frame_width - border_threshold or 
                y + h >= frame_height - border_threshold)
    
    def get_exit_border(self, frame_width: int, frame_height: int) -> str:
        """Get which border the track is exiting from"""
        x, y, w, h = self.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        if x <= 10:
            return "left"
        elif x + w >= frame_width - 10:
            return "right" 
        elif y <= 10:
            return "top"
        elif y + h >= frame_height - 10:
            return "bottom"
        else:
            return "none"
    
    def update_velocity(self):
        """Update track velocity based on trajectory"""
        if len(self.trajectory) >= 2:
            current_pos = self.trajectory[-1]
            prev_pos = self.trajectory[-2]
            self.velocity = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        else:
            self.velocity = (0.0, 0.0)
    
    def predict_next_position(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Predict next bounding box position based on velocity and trajectory"""
        if not self.velocity or (self.velocity[0] == 0 and self.velocity[1] == 0):
            return self.bbox
        
        x, y, w, h = self.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Use velocity with some damping
        damping = 0.8
        predicted_center_x = center_x + (self.velocity[0] * damping)
        predicted_center_y = center_y + (self.velocity[1] * damping)
        
        # Ensure prediction stays within frame bounds
        predicted_x = max(0, min(frame_width - w, int(predicted_center_x - w // 2)))
        predicted_y = max(0, min(frame_height - h, int(predicted_center_y - h // 2)))
        
        return (predicted_x, predicted_y, w, h)
    
    def update_quality_score(self):
        """Update track quality based on various factors"""
        base_score = 1.0
        
        # Reduce score for association failures
        if self.association_failures > 0:
            base_score *= (0.9 ** self.association_failures)
        
        # Reduce score for frequent occlusions
        if self.occlusion_count > 0:
            base_score *= (0.95 ** self.occlusion_count)
        
        # Increase score for confirmed tracks with face ID
        if self.permanent_face_id is not None:
            base_score *= 1.2
        elif self.face_id != "Unknown":
            base_score *= 1.1
        
        # Reduce score for tracks that haven't moved (potentially stuck/false)
        if not self.has_moved_significantly and self.age > 20:
            base_score *= 0.8
        
        self.quality_score = max(0.1, min(1.0, base_score))
    
    def update_stability_score(self):
        """Update stability score based on trajectory consistency"""
        if len(self.trajectory) < 3:
            self.stability_score = 0.5  # Neutral for new tracks
            return
        
        # Calculate trajectory smoothness
        positions = list(self.trajectory)[-10:]  # Last 10 positions
        if len(positions) < 3:
            return
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(2, len(positions)):
            v1 = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
            v2 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            
            # Calculate angle between vectors
            if v1 != (0, 0) and v2 != (0, 0):
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = (v1[0]**2 + v1[1]**2)**0.5
                mag2 = (v2[0]**2 + v2[1]**2)**0.5
                cos_angle = dot_product / (mag1 * mag2)
                
                # If angle is > 90 degrees, it's a significant direction change
                if cos_angle < 0:
                    direction_changes += 1
        
        # Higher stability for smoother trajectories
        self.stability_score = max(0.1, 1.0 - (direction_changes / len(positions) * 2))
    
    def is_reliable(self) -> bool:
        """Check if track is reliable for important decisions"""
        return (self.quality_score > 0.7 and 
                self.stability_score > 0.6 and 
                self.hits >= 3 and
                self.association_failures < 3)

class FaceDetector:
    """Enhanced Face Detector with YuNet support and adaptive threshold"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.original_threshold = confidence_threshold
        self.net = None
        self.use_yunet = False
        self.detection_history = []  # Track recent detection counts
        self.adaptive_mode = True
        self._load_face_detection_model()
    
    def _load_face_detection_model(self):
        """Load OpenCV DNN face detection model"""
        script_dir = Path(__file__).parent
        # Try multiple potential model paths
        model_paths = [
            script_dir.parent.parent / "models" / "face_detection_yunet_2023mar.onnx",  # backend/models/
            script_dir.parent / "assets" / "model" / "face_detection_yunet_2023mar.onnx",  # app/assets/model/
            script_dir / "face_engine" / ".." / ".." / ".." / "models" / "face_detection_yunet_2023mar.onnx"  # relative to backend/models/
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        try:
            if model_path and model_path.exists():
                self.net = cv2.FaceDetectorYN.create(
                    model=str(model_path),
                    config="",
                    input_size=(320, 240),
                    score_threshold=self.confidence_threshold
                )
                self.use_yunet = True
                print(f"Loaded local YuNet model: {model_path}")
            else:
                try:
                    # Try to find OpenCV's built-in face detector files  
                    pb_file = cv2.samples.findFile("opencv_face_detector_uint8.pb")
                    pbtxt_file = cv2.samples.findFile("opencv_face_detector.pbtxt")
                    
                    if pb_file and pbtxt_file:
                        self.net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
                        self.use_yunet = False
                        self.use_haar = False
                        print("Using OpenCV built-in DNN face detector")
                    else:
                        raise Exception("OpenCV DNN model files not found")
                        
                except Exception as dnn_error:
                    try:
                        # Fallback to default YuNet model
                        self.net = cv2.FaceDetectorYN.create(
                            model="",
                            config="",
                            input_size=(320, 240),
                            score_threshold=self.confidence_threshold
                        )
                        self.use_yunet = True
                        self.use_haar = False
                        print("Using default YuNet model")
                    except Exception as yunet_error:
                        # Final fallback to Haar cascade
                        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        if self.face_cascade.empty():
                            raise Exception("Could not load any face detection model")
                        self.use_haar = True
                        self.use_yunet = False
                        self.net = None
                        print("Using Haar cascade face detector (fallback)")
                        
        except Exception as e:
            print(f"Error loading face detector: {e}")
            print("Available models:")
            if model_path:
                print(f"- Local YuNet: {model_path} (exists: {model_path.exists()})")
            print(f"- OpenCV DNN: {cv2.samples.findFile('opencv_face_detector_uint8.pb')}")
            print(f"- Haar cascade: {cv2.data.haarcascades}haarcascade_frontalface_default.xml")
            # Try Haar cascade as final fallback
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if not self.face_cascade.empty():
                    self.use_haar = True
                    self.use_yunet = False
                    self.net = None
                    print("✅ Using Haar cascade detector as final fallback")
                else:
                    raise Exception("Haar cascade failed to load")
            except Exception as final_error:
                print(f"❌ Final fallback failed: {final_error}")
                raise Exception("No face detection method available")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in frame with adaptive threshold"""
        detections = []
        h, w = frame.shape[:2]
        
        # Use adaptive threshold if enabled
        current_threshold = self._get_adaptive_threshold()
        
        try:
            if self.use_yunet and self.net is not None:
                self.net.setInputSize((w, h))
                _, results = self.net.detect(frame)
                
                if results is not None:
                    for detection in results:
                        x, y, w_box, h_box = detection[:4].astype(int)
                        confidence = detection[14]
                        
                        if confidence > current_threshold:
                            detections.append(Detection(
                                bbox=(x, y, w_box, h_box),
                                confidence=confidence
                            ))
            elif not self.use_yunet and self.net is not None:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
                self.net.setInput(blob)
                results = self.net.forward()
                
                for i in range(results.shape[2]):
                    confidence = results[0, 0, i, 2]
                    
                    if confidence > current_threshold:
                        x1 = int(results[0, 0, i, 3] * w)
                        y1 = int(results[0, 0, i, 4] * h)
                        x2 = int(results[0, 0, i, 5] * w)
                        y2 = int(results[0, 0, i, 6] * h)
                        
                        detections.append(Detection(
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                            confidence=confidence
                        ))
            elif hasattr(self, 'use_haar') and self.use_haar and hasattr(self, 'face_cascade'):
                # Fallback to Haar cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20),
                    maxSize=(300, 300)
                )
                
                for (x, y, w_face, h_face) in faces:
                    # Calculate confidence based on face size and position
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_area = w_face * h_face
                    confidence = min(1.0, (face_area / frame_area) * 8)
                    confidence = max(0.3, min(1.0, confidence))
                    
                    if confidence > current_threshold:
                        detections.append(Detection(
                            bbox=(x, y, w_face, h_face),
                            confidence=confidence
                        ))
        except Exception as e:
            print(f"⚠️ Face detection failed: {e}")
        
        # Update detection history for adaptive threshold
        self._update_detection_history(len(detections))
        
        return detections
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive confidence threshold based on recent detection history"""
        if not self.adaptive_mode or len(self.detection_history) < 10:
            return self.confidence_threshold
        
        # Calculate average detections in last 10 frames
        avg_detections = sum(self.detection_history[-10:]) / 10
        
        # If very few detections, lower threshold
        if avg_detections < 0.5:  # Less than 0.5 faces per frame on average
            adaptive_threshold = max(0.3, self.original_threshold - 0.2)
            if adaptive_threshold != self.confidence_threshold:
                print(f"Adaptive: Lowering detection threshold to {adaptive_threshold:.2f} (avg detections: {avg_detections:.1f})")
            return adaptive_threshold
        elif avg_detections < 1.0:  # Less than 1 face per frame
            adaptive_threshold = max(0.4, self.original_threshold - 0.1)
            return adaptive_threshold
        else:
            # Reset to original threshold if enough detections
            return self.original_threshold
    
    def _update_detection_history(self, detection_count: int):
        """Update detection history for adaptive threshold"""
        self.detection_history.append(detection_count)
        # Keep only last 30 frames of history
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)

class FaceVerifier:
    """Face verification using DeepFace"""
    
    def __init__(self, distance_threshold=0.8, recognition_model="SFace", 
                 detector_backend="opencv", distance_metric="cosine"):
        self.distance_threshold = distance_threshold
        self.recognition_model = recognition_model
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        
        print(f"🔧 Face Recognition Model: {recognition_model}")
        print(f"🔧 Distance Threshold: {distance_threshold}")
    
    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding from face image"""
        if not DEEPFACE_AVAILABLE:
            return None
            
        try:
            # Convert BGR to RGB for DeepFace
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.recognition_model,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            return np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None
    
    def verify_embedding_against_database(self, face_embedding: np.ndarray, 
                                        embeddings_data: Dict[str, Dict[str, np.ndarray]]) -> Tuple[str, float]:
        """Verify face embedding against database of embeddings"""
        if not embeddings_data:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_confidence = 0.0
        min_distance = float('inf')
        
        # Check against all persons and their poses
        for person_name, person_embeddings in embeddings_data.items():
            for pose_name, target_embedding in person_embeddings.items():
                try:
                    # Calculate distance
                    if self.distance_metric == "cosine":
                        try:
                            from scipy.spatial.distance import cosine
                            distance = cosine(face_embedding, target_embedding)
                        except ImportError:
                            # Manual cosine distance calculation
                            dot_product = np.dot(face_embedding, target_embedding)
                            norm_a = np.linalg.norm(face_embedding)
                            norm_b = np.linalg.norm(target_embedding)
                            if norm_a == 0 or norm_b == 0:
                                distance = 1.0  # Maximum distance for zero vectors
                            else:
                                distance = 1 - (dot_product / (norm_a * norm_b))
                    else:
                        distance = np.linalg.norm(face_embedding - target_embedding)
                    
                    # Check if it's a match and better than previous matches
                    if distance <= self.distance_threshold and distance < min_distance:
                        min_distance = distance
                        # Calculate confidence
                        if self.distance_metric == "cosine":
                            confidence = max(0, 1 - distance)
                        else:
                            confidence = max(0, 1 - (distance / self.distance_threshold))
                        
                        best_confidence = confidence
                        best_match = person_name
                        
                except Exception as e:
                    print(f"⚠️ Error comparing embedding for {person_name}-{pose_name}: {e}")
                    continue
        
        return best_match, best_confidence

class ByteTracker:
    """ByteTrack implementation for face tracking"""
    
    def __init__(self, frame_rate=30, track_thresh=0.3, track_buffer=60, match_thresh=0.4):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.track_id_count = 0
    
    def update(self, detections: List[Detection], frame_width: int = 0, frame_height: int = 0) -> List[Track]:
        """Update tracker with new detections"""
        self.frame_id += 1
        
        det_high = [d for d in detections if d.confidence >= self.track_thresh]
        det_low = [d for d in detections if d.confidence < self.track_thresh]
        
        tracks = []
        
        matched, unmatched_dets, unmatched_trks = self._associate(
            self.tracked_stracks, det_high, self.match_thresh
        )
        
        for m in matched:
            track = self.tracked_stracks[m[1]]
            det = det_high[m[0]]
            self._update_track(track, det, frame_width, frame_height)
            tracks.append(track)
        
        if det_low:
            unmatched_tracks = [self.tracked_stracks[i] for i in unmatched_trks]
            matched_low, unmatched_dets_low, unmatched_trks_low = self._associate(
                unmatched_tracks, det_low, 0.5
            )
            
            for m in matched_low:
                track = unmatched_tracks[m[1]]
                det = det_low[m[0]]
                self._update_track(track, det, frame_width, frame_height)
                tracks.append(track)
        
        # Try to re-associate with recently lost tracks before creating new ones
        final_unmatched_dets = []
        for i in unmatched_dets:
            det = det_high[i]
            if det.confidence >= self.track_thresh:
                # Check if this detection could be a recently lost track re-entering
                reactivated_track = self._try_reactivate_lost_track(det, frame_width, frame_height)
                if reactivated_track is not None:
                    tracks.append(reactivated_track)
                else:
                    final_unmatched_dets.append(i)
        
        # Create new tracks for remaining unmatched detections
        for i in final_unmatched_dets:
            det = det_high[i]
            new_track = self._initiate_track(det)
            tracks.append(new_track)
        
        # Handle unmatched tracks (potential occlusions)
        for i in unmatched_trks:
            track = self.tracked_stracks[i]
            track.association_failures += 1
            track.occlusion_count += 1
            track.update_quality_score()
            
            # Keep track in list but mark as potentially occluded
            if track.is_reliable():
                tracks.append(track)  # Keep reliable tracks even if unmatched
        
        self.tracked_stracks = tracks
        self._remove_old_tracks(frame_width, frame_height)
        
        return self.tracked_stracks
    
    def _associate(self, tracks: List[Track], detections: List[Detection], thresh: float):
        """Associate tracks with detections using IoU with prediction and quality scoring"""
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calculate IoU matrix with prediction-enhanced matching
        cost_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                # Use predicted position if available
                if hasattr(track, 'predicted_bbox') and track.predicted_bbox:
                    predicted_iou = self._calculate_iou(det.bbox, track.predicted_bbox)
                    current_iou = self._calculate_iou(det.bbox, track.bbox)
                    # Weighted combination of current and predicted IoU
                    iou = 0.7 * current_iou + 0.3 * predicted_iou
                else:
                    iou = self._calculate_iou(det.bbox, track.bbox)
                
                # Factor in track quality - prefer higher quality tracks
                if hasattr(track, 'quality_score'):
                    quality_bonus = track.quality_score * 0.1  # Small bonus for high-quality tracks
                    iou += quality_bonus
                
                cost_matrix[d, t] = 1 - iou  # Convert to cost (lower is better)
        
        matched_indices = []
        unmatched_dets = []
        unmatched_trks = []
        
        used_dets = set()
        used_trks = set()
        
        for d in range(len(detections)):
            for t in range(len(tracks)):
                if d in used_dets or t in used_trks:
                    continue
                iou = 1 - cost_matrix[d, t]
                if iou >= thresh:
                    matched_indices.append([d, t])
                    used_dets.add(d)
                    used_trks.add(t)
                    break
        
        for d in range(len(detections)):
            if d not in used_dets:
                unmatched_dets.append(d)
        
        for t in range(len(tracks)):
            if t not in used_trks:
                unmatched_trks.append(t)
        
        return matched_indices, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track(self, track: Track, detection: Detection, frame_width: int = 0, frame_height: int = 0):
        """Update track with new detection and robustness features"""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.association_failures = 0  # Reset failures on successful association
        
        track.mark_confirmed()
        
        x, y, w, h = detection.bbox
        center = (x + w // 2, y + h // 2)
        track.trajectory.append(center)
        
        # Update robustness features
        track.update_velocity()
        track.update_quality_score()
        track.update_stability_score()
        
        # Update prediction for next frame
        if frame_width > 0 and frame_height > 0:
            track.predicted_bbox = track.predict_next_position(frame_width, frame_height)
        
        if track.initial_position is None:
            track.initial_position = center
    
    def _initiate_track(self, detection: Detection) -> Track:
        """Initialize new track"""
        self.track_id_count += 1
        track = Track(
            track_id=self.track_id_count,
            bbox=detection.bbox,
            confidence=detection.confidence,
            hits=1,
            time_since_update=0,
            age=1
        )
        
        x, y, w, h = detection.bbox
        center = (x + w // 2, y + h // 2)
        track.trajectory.append(center)
        track.initial_position = center  # Set initial position
        
        return track
    
    def _remove_old_tracks(self, frame_width: int = 0, frame_height: int = 0):
        """Remove old tracks with border-aware logic"""
        for track in self.tracked_stracks[:]:
            track.time_since_update += 1
            
            # Adaptive buffer time based on track quality and reliability
            base_buffer_time = self.track_buffer if track.is_confirmed() else max(10, self.track_buffer // 3)
            
            # Quality-based buffer adjustment
            if hasattr(track, 'quality_score') and hasattr(track, 'is_reliable'):
                if track.is_reliable():
                    quality_multiplier = 1.5 + (track.quality_score * 0.5)  # 1.5x to 2.0x for reliable tracks
                else:
                    quality_multiplier = 0.7 + (track.quality_score * 0.3)  # 0.7x to 1.0x for unreliable tracks
                
                buffer_time = int(base_buffer_time * quality_multiplier)
            else:
                buffer_time = base_buffer_time
            
            # Additional multipliers for special cases
            if frame_width > 0 and frame_height > 0 and track.is_near_border(frame_width, frame_height):
                buffer_time = int(buffer_time * 5.0)  # 5.0x longer for border tracks (increased from 1.8x)
                
            if track.is_confirmed() and hasattr(track, 'permanent_face_id') and track.permanent_face_id is not None:
                buffer_time = int(buffer_time * 1.5)  # 1.5x longer for identified faces
            
            if track.time_since_update > buffer_time:
                track.state = "lost"
                self.tracked_stracks.remove(track)
                self.lost_stracks.append(track)
    
    def _try_reactivate_lost_track(self, detection: Detection, frame_width: int, frame_height: int) -> Optional[Track]:
        """Try to reactivate a recently lost track for border re-entries"""
        det_x, det_y, det_w, det_h = detection.bbox
        det_center = (det_x + det_w // 2, det_y + det_h // 2)
        
        # Check if detection is near border (potential re-entry)
        border_threshold = 100
        is_near_border = (det_x <= border_threshold or 
                         det_y <= border_threshold or 
                         det_x + det_w >= frame_width - border_threshold or 
                         det_y + det_h >= frame_height - border_threshold)
        
        if not is_near_border:
            return None
        
        # Look for recently lost tracks that could match this detection
        best_track = None
        best_distance = float('inf')
        max_reactivation_time = 90  # frames (increased from 30 to allow more time for reactivation)
        
        for lost_track in self.lost_stracks[:]:
            # Only consider tracks lost recently
            if lost_track.time_since_update > max_reactivation_time:
                continue
                
            # Calculate distance between detection and track's last known position
            if lost_track.trajectory:
                last_pos = lost_track.trajectory[-1]
                distance = ((det_center[0] - last_pos[0]) ** 2 + 
                           (det_center[1] - last_pos[1]) ** 2) ** 0.5
                
                # Only reactivate if detection is reasonably close to last position
                if distance < 200 and distance < best_distance:
                    best_distance = distance
                    best_track = lost_track
        
        if best_track is not None:
            # Reactivate the track
            self.lost_stracks.remove(best_track)
            best_track.bbox = detection.bbox
            best_track.confidence = detection.confidence
            best_track.time_since_update = 0
            best_track.hits += 1
            best_track.age += 1
            best_track.state = "confirmed"
            
            # Update trajectory
            x, y, w, h = detection.bbox
            center = (x + w // 2, y + h // 2)
            best_track.trajectory.append(center)
            
            print(f"Track {best_track.track_id}: Reactivated from border (distance: {best_distance:.1f}px)")
            return best_track
        
        return None


class BoTSORT:
    """BoT-SORT implementation for face tracking"""
    
    def __init__(self, frame_rate=30, track_thresh=0.3, track_buffer=60, match_thresh=0.4):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.track_id_count = 0
        
        # BoT-SORT specific parameters
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
    
    def update(self, detections: List[Detection], frame_width: int = 0, frame_height: int = 0) -> List[Track]:
        """Update tracker with new detections (BoT-SORT algorithm)"""
        self.frame_id += 1
        
        # Similar to ByteTrack but with additional appearance and motion models
        det_high = [d for d in detections if d.confidence >= self.track_thresh]
        det_low = [d for d in detections if d.confidence < self.track_thresh]
        
        tracks = []
        
        # Association with motion and appearance models
        matched, unmatched_dets, unmatched_trks = self._associate_with_motion(
            self.tracked_stracks, det_high, self.match_thresh
        )
        
        # Update matched tracks
        for m in matched:
            track = self.tracked_stracks[m[1]]
            det = det_high[m[0]]
            self._update_track_with_motion(track, det, frame_width, frame_height)
            tracks.append(track)
        
        # Re-associate with low confidence detections
        if det_low:
            unmatched_tracks = [self.tracked_stracks[i] for i in unmatched_trks]
            matched_low, unmatched_dets_low, unmatched_trks_low = self._associate_with_motion(
                unmatched_tracks, det_low, 0.5
            )
            
            for m in matched_low:
                track = unmatched_tracks[m[1]]
                det = det_low[m[0]]
                self._update_track_with_motion(track, det)
                tracks.append(track)
        
        # Try to re-associate with recently lost tracks before creating new ones
        final_unmatched_dets = []
        for i in unmatched_dets:
            det = det_high[i]
            if det.confidence >= self.track_thresh:
                # Check if this detection could be a recently lost track re-entering
                reactivated_track = self._try_reactivate_lost_track(det, frame_width, frame_height)
                if reactivated_track is not None:
                    tracks.append(reactivated_track)
                else:
                    final_unmatched_dets.append(i)
        
        # Initialize new tracks for remaining unmatched detections
        for i in final_unmatched_dets:
            det = det_high[i]
            new_track = self._initiate_track_with_motion(det)
            tracks.append(new_track)
        
        self.tracked_stracks = tracks
        self._remove_old_tracks(frame_width, frame_height)
        
        return self.tracked_stracks
    
    def _associate_with_motion(self, tracks: List[Track], detections: List[Detection], thresh: float):
        """Associate tracks with detections using motion model"""
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calculate costs using IoU and motion prediction
        cost_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                # Predict next position based on motion
                predicted_bbox = self._predict_bbox(track)
                iou_cost = 1 - self._calculate_iou(det.bbox, predicted_bbox)
                cost_matrix[d, t] = iou_cost
        
        # Simple greedy matching (in practice, you'd use Hungarian algorithm)
        matched_indices = []
        unmatched_dets = []
        unmatched_trks = []
        
        used_dets = set()
        used_trks = set()
        
        for d in range(len(detections)):
            best_match = -1
            best_cost = float('inf')
            for t in range(len(tracks)):
                if t in used_trks:
                    continue
                if cost_matrix[d, t] < best_cost and cost_matrix[d, t] < (1 - thresh):
                    best_cost = cost_matrix[d, t]
                    best_match = t
            
            if best_match != -1:
                matched_indices.append([d, best_match])
                used_dets.add(d)
                used_trks.add(best_match)
        
        for d in range(len(detections)):
            if d not in used_dets:
                unmatched_dets.append(d)
        
        for t in range(len(tracks)):
            if t not in used_trks:
                unmatched_trks.append(t)
        
        return matched_indices, unmatched_dets, unmatched_trks
    
    def _predict_bbox(self, track: Track) -> Tuple[int, int, int, int]:
        """Predict next bounding box based on motion model"""
        if len(track.trajectory) < 2:
            return track.bbox
        
        # Simple linear motion prediction
        last_pos = track.trajectory[-1]
        prev_pos = track.trajectory[-2]
        
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        
        x, y, w, h = track.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        predicted_center_x = center_x + dx
        predicted_center_y = center_y + dy
        
        predicted_x = predicted_center_x - w // 2
        predicted_y = predicted_center_y - h // 2
        
        return (predicted_x, predicted_y, w, h)
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track_with_motion(self, track: Track, detection: Detection, frame_width: int = 0, frame_height: int = 0):
        """Update track with motion model and robustness features"""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.association_failures = 0  # Reset failures on successful association
        
        # Mark as confirmed if enough hits
        track.mark_confirmed()
        
        # Update trajectory with motion smoothing
        x, y, w, h = detection.bbox
        center = (x + w // 2, y + h // 2)
        track.trajectory.append(center)
        
        # Update robustness features
        track.update_velocity()
        track.update_quality_score()
        track.update_stability_score()
        
        # Update prediction for next frame
        if frame_width > 0 and frame_height > 0:
            track.predicted_bbox = track.predict_next_position(frame_width, frame_height)
        
        if track.initial_position is None:
            track.initial_position = center
    
    def _initiate_track_with_motion(self, detection: Detection) -> Track:
        """Initialize new track with motion model"""
        self.track_id_count += 1
        track = Track(
            track_id=self.track_id_count,
            bbox=detection.bbox,
            confidence=detection.confidence,
            hits=1,
            time_since_update=0,
            age=1
        )
        
        x, y, w, h = detection.bbox
        center = (x + w // 2, y + h // 2)
        track.trajectory.append(center)
        track.initial_position = center  # Set initial position
        
        return track
    
    def _remove_old_tracks(self, frame_width: int = 0, frame_height: int = 0):
        """Remove old tracks with border-aware logic"""
        for track in self.tracked_stracks[:]:
            track.time_since_update += 1
            
            # Base buffer time - confirmed tracks get longer buffer
            buffer_time = self.track_buffer if track.is_confirmed() else max(10, self.track_buffer // 3)
            
            # Extend buffer time for tracks near borders
            if frame_width > 0 and frame_height > 0 and track.is_near_border(frame_width, frame_height):
                buffer_time = int(buffer_time * 6.0)  # 6.0x longer for border tracks (increased from 2.5x)
                
            # Even longer buffer for confirmed tracks with face ID
            if track.is_confirmed() and hasattr(track, 'permanent_face_id') and track.permanent_face_id is not None:
                buffer_time = int(buffer_time * 2.0)  # 2x longer for identified faces
            
            if track.time_since_update > buffer_time:
                track.state = "lost"
                self.tracked_stracks.remove(track)
                self.lost_stracks.append(track)
    
    def _try_reactivate_lost_track(self, detection: Detection, frame_width: int, frame_height: int) -> Optional[Track]:
        """Try to reactivate a recently lost track for border re-entries"""
        det_x, det_y, det_w, det_h = detection.bbox
        det_center = (det_x + det_w // 2, det_y + det_h // 2)
        
        # Check if detection is near border (potential re-entry)
        border_threshold = 100
        is_near_border = (det_x <= border_threshold or 
                         det_y <= border_threshold or 
                         det_x + det_w >= frame_width - border_threshold or 
                         det_y + det_h >= frame_height - border_threshold)
        
        if not is_near_border:
            return None
        
        # Look for recently lost tracks that could match this detection
        best_track = None
        best_distance = float('inf')
        max_reactivation_time = 90  # frames (increased from 30 to allow more time for reactivation)
        
        for lost_track in self.lost_stracks[:]:
            # Only consider tracks lost recently
            if lost_track.time_since_update > max_reactivation_time:
                continue
                
            # Calculate distance between detection and track's last known position
            if lost_track.trajectory:
                last_pos = lost_track.trajectory[-1]
                distance = ((det_center[0] - last_pos[0]) ** 2 + 
                           (det_center[1] - last_pos[1]) ** 2) ** 0.5
                
                # Only reactivate if detection is reasonably close to last position
                if distance < 200 and distance < best_distance:
                    best_distance = distance
                    best_track = lost_track
        
        if best_track is not None:
            # Reactivate the track
            self.lost_stracks.remove(best_track)
            best_track.bbox = detection.bbox
            best_track.confidence = detection.confidence
            best_track.time_since_update = 0
            best_track.hits += 1
            best_track.age += 1
            best_track.state = "confirmed"
            
            # Update trajectory
            x, y, w, h = detection.bbox
            center = (x + w // 2, y + h // 2)
            best_track.trajectory.append(center)
            
            print(f"Track {best_track.track_id}: Reactivated from border (distance: {best_distance:.1f}px)")
            return best_track
        
        return None


class RegionSelector:
    """Interactive region selection for video analysis"""
    
    def __init__(self):
        self.regions = []
        self.current_region = []
        self.drawing = False
        self.region_mode = False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for region selection"""
        if not self.region_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_region = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_region = [self.current_region[0], (x, y)]
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_region.append((x, y))
            if len(self.current_region) >= 2:
                self.regions.append(self.current_region[:2])
                print(f"Region {len(self.regions)} added: {self.current_region[:2]}")
            self.current_region = []
    
    def draw_regions(self, frame: np.ndarray) -> np.ndarray:
        """Draw regions on frame"""
        for i, region in enumerate(self.regions):
            if len(region) == 2:
                pt1, pt2 = region
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, f"Region {i+1}", 
                           (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.drawing and len(self.current_region) == 2:
            pt1, pt2 = self.current_region
            cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 2)
        
        return frame
    
    def point_in_regions(self, point: Tuple[int, int]) -> List[int]:
        """Check if point is in any region"""
        x, y = point
        regions_containing_point = []
        
        for i, region in enumerate(self.regions):
            if len(region) == 2:
                pt1, pt2 = region
                x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                
                if x1 <= x <= x2 and y1 <= y <= y2:
                    regions_containing_point.append(i + 1)
        
        return regions_containing_point


class FaceRecognitionService:
    """Face Recognition Service for Qt integration"""
    
    def __init__(self, confidence_threshold=0.5, verification_interval=30, 
                 detector_backend="opencv", recognition_model="SFace", tracker_type="bytetrack"):
        self.confidence_threshold = confidence_threshold
        self.verification_interval = verification_interval
        self.detector_backend = detector_backend
        self.recognition_model = recognition_model
        self.tracker_type = tracker_type.lower()
        
        # Initialize components
        print(f"🔧 Initializing FaceDetector with confidence threshold: {confidence_threshold}")
        self.detector = FaceDetector(confidence_threshold=confidence_threshold)
        print(f"✅ FaceDetector initialized successfully")
        self.verifier = FaceVerifier(
            recognition_model=recognition_model,
            detector_backend=detector_backend
        ) if DEEPFACE_AVAILABLE else None
        
        # Initialize advanced tracker
        if self.tracker_type == "bytetrack":
            self.tracker = ByteTracker()
            print(f"{Colors.GREEN}Using ByteTrack tracker{Colors.END}")
        elif self.tracker_type == "botsort":
            self.tracker = BoTSORT()
            print(f"{Colors.GREEN}Using BoTSORT tracker{Colors.END}")
        else:
            # Fallback to ByteTracker for unknown types
            self.tracker = ByteTracker()
            print(f"{Colors.YELLOW}Unknown tracker type '{tracker_type}', falling back to ByteTrack{Colors.END}")
        
        self.region_selector = RegionSelector()
        
        # State
        self.frame_count = 0
        self.embeddings_database = {}
        self.is_running = False
        self.current_frame = None
        self.current_tracks = []
        self.current_detections = []
        
        # Statistics
        self.total_detections = 0
        self.total_tracks = 0
        
        print("✅ Face Recognition Service initialized")
        print(f"🔧 Detection: Enhanced FaceDetector with YuNet support and adaptive threshold")
        print(f"🔧 Recognition: DeepFace {recognition_model} model" if self.verifier else "🔧 Recognition: DISABLED (DeepFace not available)")
        print(f"🔧 Tracker: {self.tracker_type.upper()} with robustness features")
        print(f"🔧 Features: Border tracking, quality scoring, region monitoring")
        print(f"🔧 Embeddings Database: {len(self.embeddings_database)} persons loaded")
    
    def load_embeddings_database(self, enrollment_service) -> bool:
        """Load embeddings from enrollment service"""
        print("🔄 Loading embeddings database...")
        try:
            self.embeddings_database = enrollment_service.get_all_person_embeddings()
            total_embeddings = sum(len(embeddings) for embeddings in self.embeddings_database.values())
            print(f"✅ Loaded {total_embeddings} embeddings for {len(self.embeddings_database)} persons")
            return True
        except Exception as e:
            print(f"❌ Error loading embeddings database: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for face recognition"""
        self.frame_count += 1
        self.current_frame = frame.copy()
        
        try:
            # Detect faces
            start_time = time.time()
            detections = self.detector.detect(frame)
            detection_time = time.time() - start_time
            
            # Debug: Log detection results every 60 frames
            if self.frame_count % 60 == 1:
                detector_type = "YuNet" if getattr(self.detector, 'use_yunet', False) else ("Haar" if getattr(self.detector, 'use_haar', False) else "DNN")
                print(f"🔍 Frame {self.frame_count}: {detector_type} detector found {len(detections)} faces")
            
            # Track faces
            start_time = time.time()
            h, w = frame.shape[:2]
            tracks = self.tracker.update(detections, w, h)
            tracking_time = time.time() - start_time
            
            # Verify faces
            verification_time = 0
            if self.verifier and self.embeddings_database:
                start_time = time.time()
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                if confirmed_tracks and self.frame_count % 30 == 1:  # Debug every 30 frames
                    print(f"🔍 Frame {self.frame_count}: Verifying {len(confirmed_tracks)} confirmed tracks against {len(self.embeddings_database)} enrolled persons")
                
                for track in tracks:
                    if track.is_confirmed():
                        self._verify_face(frame, track)
                verification_time = time.time() - start_time
            else:
                if self.frame_count % 60 == 1:  # Debug every 60 frames
                    print(f"⚠️ Frame {self.frame_count}: Face verification disabled - Verifier: {self.verifier is not None}, Database: {len(self.embeddings_database) if self.embeddings_database else 0} persons")
            
            # Update state
            self.current_detections = detections
            self.current_tracks = tracks
            self.total_detections += len(detections)
            self.total_tracks = len(tracks)
            
            return {
                'success': True,
                'frame_count': self.frame_count,
                'detections': detections,
                'tracks': tracks,
                'detection_time': detection_time,
                'tracking_time': tracking_time,
                'verification_time': verification_time,
                'total_detections': self.total_detections,
                'total_tracks': self.total_tracks
            }
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return {
                'success': False,
                'error': str(e),
                'frame_count': self.frame_count
            }
    
    def _verify_face(self, frame: np.ndarray, track: Track) -> None:
        """Verify face identity for a track"""
        if not self.verifier or not self.embeddings_database:
            print(f"⚠️ Track {track.track_id}: Verification skipped - Verifier: {self.verifier is not None}, Database: {len(self.embeddings_database) if self.embeddings_database else 0}")
            return
        
        # Skip verification if permanent face ID is already assigned
        if track.permanent_face_id is not None:
            return
        
        # Movement-based verification
        if track.initial_position is not None:
            x, y, w, h = track.bbox
            current_center = (x + w // 2, y + h // 2)
            initial_center = track.initial_position
            
            movement_distance = ((current_center[0] - initial_center[0]) ** 2 + 
                               (current_center[1] - initial_center[1]) ** 2) ** 0.5
            movement_threshold = frame.shape[0] * 0.1  # 10% of frame height
            
            if not track.has_moved_significantly:
                if movement_distance >= movement_threshold:
                    track.has_moved_significantly = True
                    print(f"🔍 Track {track.track_id}: Movement threshold reached ({movement_distance:.1f} >= {movement_threshold:.1f}), starting verification")
                else:
                    return  # Don't verify until significant movement
            
            # Verify every N frames after movement
            if self.frame_count - track.last_verification_frame < self.verification_interval:
                return
        
        print(f"🔍 Track {track.track_id}: Starting face verification process")
        
        # Extract face region
        x, y, w, h = track.bbox
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size > 0:
            try:
                print(f"🔍 Track {track.track_id}: Extracting face embedding from {face_img.shape} face image")
                embedding = self.verifier.get_face_embedding(face_img)
                if embedding is not None:
                    print(f"🔍 Track {track.track_id}: Face embedding extracted, verifying against database")
                    face_id, confidence = self.verifier.verify_embedding_against_database(
                        embedding, self.embeddings_database
                    )
                    print(f"🔍 Track {track.track_id}: Verification result: {face_id} (confidence: {confidence:.3f})")
                else:
                    print(f"⚠️ Track {track.track_id}: Failed to extract face embedding")
                    return
                
                # Implement voting system
                track.verification_count += 1
                track.last_verification_frame = self.frame_count
                
                if face_id != "Unknown":
                    # Add vote for this face ID
                    if face_id in track.face_id_votes:
                        track.face_id_votes[face_id] += 1
                    else:
                        track.face_id_votes[face_id] = 1
                    
                    # Check if any face ID has reached 5 votes
                    for voted_face_id, votes in track.face_id_votes.items():
                        if votes >= 5:
                            # Assign permanent face ID
                            track.permanent_face_id = voted_face_id
                            track.face_id = voted_face_id
                            track.face_confidence = confidence
                            print(f"✅ Track {track.track_id}: CONFIRMED as {voted_face_id} (after {votes} votes)")
                            return
                    
                    # Update current best guess
                    best_face_id = max(track.face_id_votes, key=track.face_id_votes.get)
                    best_votes = track.face_id_votes[best_face_id]
                    track.face_id = best_face_id
                    track.face_confidence = confidence
                    
                    print(f"🔄 Track {track.track_id}: Vote for {face_id} -> {best_face_id} ({best_votes}/5 votes)")
                else:
                    track.face_id = "Unknown"
                    track.face_confidence = 0.0
            except Exception as e:
                print(f"❌ Error during face verification for track {track.track_id}: {e}")
                track.last_verification_frame = self.frame_count
    
    def get_annotated_frame(self, frame: np.ndarray, show_detections: bool = True, show_tracks: bool = True, show_names: bool = True, show_regions: bool = False) -> np.ndarray:
        """Get frame with annotations drawn"""
        annotated_frame = frame.copy()
        
        # Debug: Print detection and tracking counts (only every 60 frames to reduce spam)
        if self.frame_count % 60 == 0:
            print(f"🔍 Frame {self.frame_count}: {len(self.current_detections)} detections, {len(self.current_tracks)} tracks")
        
        # Draw detections (yellow boxes) if enabled
        if show_detections:
            for det in self.current_detections:
                x, y, w, h = det.bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Yellow, thicker
                cv2.putText(annotated_frame, f"Det: {det.confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Thicker text
        
        # Draw tracks with face IDs (colored boxes) if enabled
        if show_tracks:
            for track in self.current_tracks:
                x, y, w, h = track.bbox
                color = self._get_track_color(track.track_id)
                
                line_thickness = 4 if track.is_confirmed() else 2  # Make all tracks more visible
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, line_thickness)
                
                # Debug track drawing (only every 60 frames) 
                if self.frame_count % 60 == 0:
                    print(f"   Track {track.track_id}: bbox=({x},{y},{w},{h}), color={color}")
                
                # Draw track ID and face ID (names) if enabled
                if show_names:
                    verification_started = track.has_moved_significantly or track.verification_count > 0
                    
                    if track.permanent_face_id is not None:
                        # Permanent face ID assigned
                        label = f"ID:{track.track_id} - {track.permanent_face_id} [CONFIRMED]"
                        label_color = (0, 255, 0)  # Green for confirmed faces
                    elif verification_started and track.face_id != "Unknown" and track.face_confidence > 0:
                        # Show voting progress
                        if track.face_id_votes and track.face_id in track.face_id_votes:
                            votes = track.face_id_votes[track.face_id]
                            label = f"ID:{track.track_id} - {track.face_id} ({votes}/5)"
                        else:
                            label = f"ID:{track.track_id} - {track.face_id} (0/5)"
                        label_color = (0, 255, 255)  # Yellow for voting in progress
                    elif verification_started:
                        # Show Unknown only after verification has started
                        label = f"ID:{track.track_id} - Unknown"
                        label_color = color
                    else:
                        # Before movement threshold - show only track ID
                        label = f"ID:{track.track_id}"
                        label_color = color
                    
                    cv2.putText(annotated_frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                
                # Draw border proximity indicator
                if track.is_near_border(annotated_frame.shape[1], annotated_frame.shape[0], 50):
                    # Show border indicator (orange triangle) for tracks near borders
                    cv2.circle(annotated_frame, (x + 10, y + 10), 4, (0, 165, 255), -1)  # Orange circle
                    cv2.putText(annotated_frame, "B", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw track quality indicator
                if hasattr(track, 'quality_score') and hasattr(track, 'is_reliable'):
                    if track.is_reliable():
                        # High quality track - solid green circle
                        cv2.circle(annotated_frame, (x + w - 10, y + 10), 4, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, "Q", (x + w - 15, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    elif track.quality_score < 0.5:
                        # Low quality track - red circle
                        cv2.circle(annotated_frame, (x + w - 10, y + 10), 4, (0, 0, 255), -1)
                        cv2.putText(annotated_frame, "!", (x + w - 13, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw movement verification indicator
                if track.last_verification_frame == self.frame_count:
                    # Show verification indicator (cyan circle) for tracks that were just verified
                    cv2.circle(annotated_frame, (x + w - 20, y + 10), 4, (255, 255, 0), -1)
                    cv2.putText(annotated_frame, "V", (x + w - 23, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw trajectory
                if len(track.trajectory) > 1:
                    points = list(track.trajectory)
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], color, 2)
                
                # Check if track is in any region and draw region info
                if show_regions and self.region_selector.regions:
                    center = (x + w // 2, y + h // 2)
                    regions = self.region_selector.point_in_regions(center)
                    if regions:
                        cv2.putText(annotated_frame, f"Regions: {regions}", 
                                   (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw regions if enabled
        if show_regions:
            annotated_frame = self.region_selector.draw_regions(annotated_frame)
        
        # If no detections or tracks, draw a message to show processing is working
        if len(self.current_detections) == 0 and len(self.current_tracks) == 0:
            cv2.putText(annotated_frame, f"Frame {self.frame_count}: No faces detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Always show frame counter and tracker info in top-right
        h, w = annotated_frame.shape[:2]
        cv2.putText(annotated_frame, f"Frame: {self.frame_count} | Tracker: {self.tracker_type.upper()}", 
                   (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show detection threshold info if adaptive
        if hasattr(self.detector, '_get_adaptive_threshold'):
            current_threshold = self.detector._get_adaptive_threshold()
            cv2.putText(annotated_frame, f"Detection Threshold: {current_threshold:.2f}", 
                       (w - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (128, 128, 128), # Gray
            (255, 192, 203), # Pink
        ]
        return colors[track_id % len(colors)]
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        confirmed_tracks = sum(1 for track in self.current_tracks if track.is_confirmed())
        identified_tracks = sum(1 for track in self.current_tracks if track.permanent_face_id is not None)
        reliable_tracks = sum(1 for track in self.current_tracks if hasattr(track, 'is_reliable') and track.is_reliable())
        
        # Detection threshold info
        current_threshold = self.detector._get_adaptive_threshold() if hasattr(self.detector, '_get_adaptive_threshold') else self.confidence_threshold
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.current_tracks),
            'confirmed_tracks': confirmed_tracks,
            'identified_tracks': identified_tracks,
            'reliable_tracks': reliable_tracks,
            'total_detections': self.total_detections,
            'total_embeddings': sum(len(embeddings) for embeddings in self.embeddings_database.values()),
            'enrolled_persons': len(self.embeddings_database),
            'verification_enabled': self.verifier is not None and bool(self.embeddings_database),
            'current_tracks': self.current_tracks,  # Include current tracks for GUI processing
            'tracker_type': self.tracker_type,
            'detection_threshold': current_threshold,
            'region_count': len(self.region_selector.regions),
            'features': {
                'border_tracking': True,
                'quality_scoring': True,
                'adaptive_threshold': True,
                'voting_system': True,
                'region_monitoring': len(self.region_selector.regions) > 0
            }
        }