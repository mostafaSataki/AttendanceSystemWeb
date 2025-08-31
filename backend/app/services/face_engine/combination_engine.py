#!/usr/bin/env python3
"""
Face Recognition Combination Engine
Supports two combinations as specified:
1. YuNet + SFace + YuNet alignment +  EDIFFIQA
2. DeepFace OpenCV + DeepFace SFace + DeepFace alignment + EDIFFIQA
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .yunet_detector import YuNetDetector
from .simple_detector import SimpleFaceDetector
from .sface_recognizer import SFaceRecognizer
from .face_aligner import FaceAligner
from .face_quality import FaceQualityAssessment

class CombinationType(Enum):
    YUNET_SFACE = "yunet_sface"


class FaceRecognitionCombinationEngine:
    """Face recognition engine supporting multiple combinations"""
    
    def __init__(self, combination_type: CombinationType, model_paths: Dict[str, str] = None):
        self.combination_type = combination_type
        self.model_paths = model_paths or {}
        
        # Initialize components based on combination type
        self._initialize_components()
        
        # Common components
        self.aligner = FaceAligner()
        # self.tracker = KalmanFaceTracker(max_disappeared=30, max_distance=100)  # Not needed for enrollment
        self.quality_assessor = FaceQualityAssessment(
            model_path=self.model_paths.get('quality', 'models/ediffiqa_tiny_jun2024.onnx')
        )
        
        # Recognition parameters
        self.recognition_threshold = 0.4
        self.quality_threshold = 0.3  # Lowered for better enrollment success
        
        # Aggressive performance optimizations
        self.max_faces_per_frame = 1  # Process only 1 face per frame for maximum speed
        self.skip_quality_for_tracking = True  # Skip quality assessment for face tracking
        self.enable_fast_mode = True  # Enable fast processing mode
        self.detection_confidence_boost = 0.1  # Boost confidence by 10% to help with strict thresholds
        
        print(f"SUCCESS: Face recognition engine initialized with {combination_type.value} combination")
    
    def _initialize_components(self):
        """Initialize components based on combination type"""
        if self.combination_type == CombinationType.YUNET_SFACE:
            # Combination 1: YuNet + OpenCV SFace (with fallback)
            try:
                self.detector = YuNetDetector(
                    model_path=self.model_paths.get('detector', 'models/face_detection_yunet_2023mar.onnx')
                )
                if not self.detector.is_available():
                    print("WARNING: YuNet not available, using simple detector")
                    self.detector = SimpleFaceDetector()
            except Exception as e:
                print(f"WARNING: YuNet failed ({e}), using simple detector")
                self.detector = SimpleFaceDetector()
            self.recognizer = SFaceRecognizer(
                model_path=self.model_paths.get('recognizer', 'models/face_recognition_sface_2021dec.onnx')
            )
            self.alignment_method = "yunet"

        else:
            raise ValueError(f"Unsupported combination type: {self.combination_type}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame"""
        if not self.detector.is_available():
            print("❌ Detector not available")
            return []
        
        detections = self.detector.detect_faces(frame)
        
        # Add combination info to detections
        for detection in detections:
            detection['combination'] = self.combination_type.value
            detection['alignment_method'] = self.alignment_method
        
        return detections
    
    def track_faces(self, detections: List[Dict]) -> List[Dict]:
        """Simple tracking - just return detections for enrollment (no complex tracking needed)"""
        return detections
    
    def align_face(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Align face using configured alignment method"""
        return self.aligner.auto_align(frame, detection, self.alignment_method)
    
    def assess_quality(self, face_roi: np.ndarray) -> float:
        """Assess face quality"""
        return self.quality_assessor.assess_quality(face_roi) or 0.5
    
    def generate_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Generate face embedding"""
        if not self.recognizer.is_available():
            print("❌ Recognizer not available")
            return None

        return self.recognizer.generate_embedding(face_roi)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return self.recognizer.compute_cosine_similarity(embedding1, embedding2)
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       gallery_embeddings: List[np.ndarray]) -> Optional[Tuple[int, float]]:
        """Find best matching embedding"""
        return self.recognizer.find_best_match(
            query_embedding, gallery_embeddings, self.recognition_threshold
        )
    
    def process_frame_for_recognition(self, frame: np.ndarray, 
                                    gallery_embeddings: List[np.ndarray] = None,
                                    gallery_names: List[str] = None) -> Dict[str, Any]:
        """Process frame for face recognition"""
        results = {
            'detections': [],
            'tracked_faces': [],
            'recognized_faces': [],
            'frame_info': {
                'combination': self.combination_type.value,
                'total_faces': 0,
                'recognized_count': 0
            }
        }
        
        try:
            # Detect faces
            detections = self.detect_faces(frame)
            results['detections'] = detections
            results['frame_info']['total_faces'] = len(detections)
            
            if not detections:
                return results
            
            # Limit number of faces processed for performance
            if len(detections) > self.max_faces_per_frame:
                # Keep the highest confidence faces
                detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)[:self.max_faces_per_frame]
                results['detections'] = detections
            
            # Track faces
            tracked_faces = self.track_faces(detections)
            results['tracked_faces'] = tracked_faces
            
            # Process each tracked face
            for face in tracked_faces:
                try:
                    # Align face
                    aligned_face = self.align_face(frame, face)
                    if aligned_face is None:
                        continue
                    
                    # Assess quality
                    quality_score = self.assess_quality(aligned_face)
                    face['quality_score'] = quality_score
                    
                    if quality_score < self.quality_threshold:
                        face['recognition_status'] = 'low_quality'
                        continue
                    
                    # Generate embedding
                    embedding = self.generate_embedding(aligned_face)
                    if embedding is None:
                        face['recognition_status'] = 'embedding_failed'
                        continue
                    
                    face['embedding'] = embedding
                    
                    # Match against gallery if provided
                    if gallery_embeddings and gallery_names:
                        match_result = self.find_best_match(embedding, gallery_embeddings)
                        
                        if match_result:
                            match_idx, similarity = match_result
                            face['recognized_name'] = gallery_names[match_idx]
                            face['recognition_confidence'] = similarity
                            face['recognition_status'] = 'recognized'
                            results['recognized_faces'].append(face)
                            results['frame_info']['recognized_count'] += 1
                        else:
                            face['recognition_status'] = 'unknown'
                    else:
                        face['recognition_status'] = 'no_gallery'
                
                except Exception as e:
                    print(f"❌ Error processing face: {e}")
                    face['recognition_status'] = 'processing_error'
            
            return results
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            results['error'] = str(e)
            return results
    
    def process_frame_for_enrollment(self, frame: np.ndarray, 
                                   required_poses: List[str] = None) -> Dict[str, Any]:
        """Process frame for face enrollment"""
        required_poses = required_poses or ['front', 'left', 'right', 'up', 'down']
        
        results = {
            'detections': [],
            'enrollment_candidates': [],
            'frame_info': {
                'combination': self.combination_type.value,
                'total_faces': 0,
                'good_quality_faces': 0
            }
        }
        
        try:
            # Detect faces
            detections = self.detect_faces(frame)
            results['detections'] = detections
            results['frame_info']['total_faces'] = len(detections)
            
            # For enrollment, focus on the best face only for speed
            if len(detections) > 1:
                # Keep only the highest confidence face for enrollment
                best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                # Boost confidence slightly to help with strict thresholds
                best_detection['confidence'] = min(1.0, best_detection['confidence'] + self.detection_confidence_boost)
                detections = [best_detection]
                results['detections'] = detections
            elif len(detections) == 1:
                # Boost confidence for single detection too
                detections[0]['confidence'] = min(1.0, detections[0]['confidence'] + self.detection_confidence_boost)
            
            # Process each detection for enrollment
            for detection in detections:
                try:
                    # Align face
                    aligned_face = self.align_face(frame, detection)
                    if aligned_face is None:
                        continue
                    
                    # Assess quality
                    quality_score = self.assess_quality(aligned_face)
                    
                    if quality_score >= self.quality_threshold:
                        results['frame_info']['good_quality_faces'] += 1
                        
                        # Generate embedding for enrollment
                        # embedding = self.generate_embedding(aligned_face)
                        
                        enrollment_candidate = {
                            'detection': detection,
                            'aligned_face': aligned_face,
                            'quality_score': quality_score,
                            # 'embedding': embedding,
                            # 'suitable_for_enrollment': embedding is not None
                        }
                        
                        results['enrollment_candidates'].append(enrollment_candidate)
                
                except Exception as e:
                    print(f"❌ Error processing enrollment candidate: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Enrollment frame processing error: {e}")
            results['error'] = str(e)
            return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            'combination_type': self.combination_type.value,
            'components': {
                'detector': type(self.detector).__name__,
                'recognizer': type(self.recognizer).__name__,
                'aligner': type(self.aligner).__name__,
                # 'tracker': type(self.tracker).__name__,  # Not needed for enrollment
                'quality_assessor': type(self.quality_assessor).__name__
            },
            'alignment_method': self.alignment_method,
            'thresholds': {
                'recognition': self.recognition_threshold,
                'quality': self.quality_threshold
            },
            'availability': {
                'detector': self.detector.is_available(),
                'recognizer': self.recognizer.is_available(),
                'quality_assessor': self.quality_assessor.is_available()
            }
        }
    
    # def reset_tracker(self):
    #     """Reset face tracker"""
    #     self.tracker.reset()
    
    def set_thresholds(self, recognition_threshold: float = None, quality_threshold: float = None):
        """Set recognition and quality thresholds"""
        if recognition_threshold is not None:
            self.recognition_threshold = recognition_threshold
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold