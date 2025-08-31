#!/usr/bin/env python3
"""
YuNet Face Detection from OpenCV Zoo
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class YuNetDetector:
    """YuNet face detector from OpenCV Zoo"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.4, nms_threshold: float = 0.3):
        self.model_path = model_path or "models/face_detection_yunet_2023mar.onnx"
        self.confidence_threshold = confidence_threshold  # Lowered for better detection rate
        self.nms_threshold = nms_threshold
        self.detector = None
        self.input_size = (320, 240)
        self.min_face_size = 20  # Allow smaller faces
        self.max_face_size = 600

        # Relaxed filtering
        self.enable_smart_filtering = False  # Keep disabled
        self.min_eye_distance = 8  # More lenient
        self.max_face_area_ratio = 0.8  # Allow larger faces
        
        self._initialize()
    
    def _initialize(self):
        """Initialize YuNet detector with fallback for older OpenCV"""
        try:
            if not os.path.exists(self.model_path):
                print(f"ERROR: YuNet model not found at: {self.model_path}")
                return self._initialize_fallback()
            
            # Try to use YuNet if available (OpenCV 4.8+)
            if hasattr(cv2, 'FaceDetectorYN'):
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=self.input_size,
                    score_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold
                )
                
                if self.detector is None:
                    print(f"ERROR: Failed to create YuNet detector")
                    return self._initialize_fallback()
                
                print(f"SUCCESS: YuNet detector initialized with confidence: {self.confidence_threshold}")
                self.use_fallback = False
                return True
            else:
                print(f"WARNING: OpenCV {cv2.__version__} doesn't support FaceDetectorYN, using fallback")
                return self._initialize_fallback()
            
        except Exception as e:
            print(f"ERROR: Failed to initialize YuNet detector: {e}")
            return self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback face detector using Haar cascades"""
        try:
            # Use Haar cascade as fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                print("ERROR: Failed to load Haar cascade detector")
                return False
            
            print("SUCCESS: Fallback Haar cascade detector initialized")
            self.use_fallback = True
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize fallback detector: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using YuNet or fallback detector"""
        if self.detector is None:
            return []
        
        if getattr(self, 'use_fallback', False):
            return self._detect_faces_fallback(frame)
        else:
            return self._detect_faces_yunet(frame)
    
    def _detect_faces_yunet(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using YuNet"""
        
        try:
            height, width = frame.shape[:2]
            
            # Set input size for the detector
            # Let the detector handle resizing internally by setting a fixed input size
            self.detector.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.detector.detect(frame)
            
            if faces is None:
                return []
            
            detections = []
            for face in faces:
                # YuNet returns: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence]
                x, y, w, h = map(int, face[:4])
                confidence = float(face[14])
                
                # Basic filtering
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                
                # Relaxed aspect ratio check
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                    continue
                
                # Extract landmarks
                landmarks = [[int(face[i]), int(face[i+1])] for i in range(4, 14, 2)]
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'landmarks': landmarks,
                    'detector': 'yunet'
                })
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"❌ YuNet detection error: {e}")
            return []
    
    def _detect_faces_fallback(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using Haar cascade fallback"""
        try:
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            for (x, y, w, h) in faces:
                # Convert to YuNet-compatible format
                detection = {
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,  # Fixed confidence for Haar
                    'landmarks': [],  # No landmarks in Haar
                    'x': x, 'y': y, 'w': w, 'h': h
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"ERROR: Haar cascade detection failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if detector is ready"""
        return self.detector is not None
    
    def align_face(self, frame: np.ndarray, landmarks: List[List[int]], target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """Align face using YuNet landmarks"""
        if len(landmarks) < 5:
            return None
        
        try:
            landmarks_np = np.array(landmarks, dtype=np.float32)
            
            # Define ideal landmark positions for aligned face
            ideal_landmarks = np.array([
                [38.2946, 51.6963],  # Right eye
                [73.5318, 51.5014],  # Left eye
                [56.0252, 71.7366],  # Nose tip
                [41.5493, 92.3655],  # Right mouth corner
                [70.7299, 92.2041]   # Left mouth corner
            ], dtype=np.float32)
            
            # Scale to target size
            if target_size != (112, 112):
                scale_x = target_size[0] / 112.0
                scale_y = target_size[1] / 112.0
                ideal_landmarks[:, 0] *= scale_x
                ideal_landmarks[:, 1] *= scale_y
            
            # Compute similarity transformation
            transformation_matrix = cv2.estimateAffinePartial2D(
                landmarks_np, ideal_landmarks, method=cv2.LMEDS
            )[0]
            
            if transformation_matrix is None:
                return None
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                frame, transformation_matrix, target_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            
            return aligned_face
            
        except Exception as e:
            print(f"❌ YuNet alignment error: {e}")
            return None