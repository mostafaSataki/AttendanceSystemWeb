#!/usr/bin/env python3
"""
Simple Face Detector using OpenCV Haar Cascades
Fallback for when YuNet is not available
"""

import cv2
import numpy as np
from typing import List

class SimpleFaceDetector:
    """Simple face detector using Haar cascades"""
    
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.detector = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Haar cascade detector"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                print("ERROR: Failed to load Haar cascade detector")
                return False
            
            print("SUCCESS: Simple face detector initialized")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize simple detector: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using Haar cascade"""
        if self.detector is None:
            return []
        
        try:
            # Convert to grayscale
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
                detection = {
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,  # Fixed confidence for Haar
                    'landmarks': [],
                    'x': x, 'y': y, 'w': w, 'h': h
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"ERROR: Face detection failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.detector is not None