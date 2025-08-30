#!/usr/bin/env python3
"""
Face Alignment System
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

class FaceAligner:
    """Universal face alignment system supporting multiple alignment methods"""
    
    def __init__(self):
        # Standard facial landmark positions for 112x112 face (DeepFace/ArcFace standard)
        self.ideal_landmarks_112 = np.array([
            [38.2946, 51.6963],  # Right eye
            [73.5318, 51.5014],  # Left eye
            [56.0252, 71.7366],  # Nose tip
            [41.5493, 92.3655],  # Right mouth corner
            [70.7299, 92.2041]   # Left mouth corner
        ], dtype=np.float32)
    
    def align_yunet(self, frame: np.ndarray, landmarks: List[List[int]], 
                    target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """Align face using YuNet landmarks (5-point alignment)"""
        if len(landmarks) < 5:
            print("⚠️ Insufficient YuNet landmarks for alignment")
            return None
        
        try:
            landmarks_np = np.array(landmarks, dtype=np.float32)
            
            # Scale ideal landmarks to target size
            ideal_landmarks = self._scale_landmarks(self.ideal_landmarks_112, target_size)
            
            # Compute similarity transformation
            transformation_matrix = self._get_similarity_transform(landmarks_np, ideal_landmarks)
            
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
    
    def align_deepface(self, frame: np.ndarray, landmarks: List[List[int]], 
                      target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """Align face using DeepFace-style alignment (eye-based)"""
        if len(landmarks) < 2:
            print("⚠️ Insufficient landmarks for DeepFace alignment")
            return None
        
        try:
            # Use eye landmarks (first two points)
            left_eye = np.array(landmarks[1], dtype=np.float32)   # Left eye in image
            right_eye = np.array(landmarks[0], dtype=np.float32)  # Right eye in image
            
            # Calculate angle between eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point between eyes
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            
            # Calculate scale based on eye distance
            eye_distance = np.linalg.norm(right_eye - left_eye)
            desired_eye_distance = target_size[0] * 0.35  # Standard eye distance ratio
            scale = desired_eye_distance / eye_distance if eye_distance > 0 else 1.0
            
            # Get rotation matrix with scaling
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
            
            # Adjust translation to center the face in target image
            tx = target_size[0] * 0.5 - eye_center[0] * scale
            ty = target_size[1] * 0.4 - eye_center[1] * scale  # Eyes slightly above center
            
            rotation_matrix[0, 2] += tx
            rotation_matrix[1, 2] += ty
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                frame, rotation_matrix, target_size,
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
            )
            
            return aligned_face
            
        except Exception as e:
            print(f"❌ DeepFace alignment error: {e}")
            return None
    
    def align_bbox_enhanced(self, frame: np.ndarray, bbox: List[int], 
                           target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """Enhanced bbox-based alignment with preprocessing"""
        try:
            x, y, w, h = bbox
            
            # Use asymmetric margins for better face capture
            margin = 0.15
            margin_x = int(w * margin)
            margin_y_top = int(h * (margin + 0.1))  # More margin above for forehead
            margin_y_bottom = int(h * margin)        # Less margin below
            
            # Calculate region with smart margins
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y_top)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y_bottom)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Apply preprocessing
            face_processed = self._enhance_face_quality(face_roi)
            
            # Resize to target size with high-quality interpolation
            face_resized = cv2.resize(face_processed, target_size, interpolation=cv2.INTER_CUBIC)
            
            return face_resized
            
        except Exception as e:
            print(f"❌ Enhanced bbox alignment error: {e}")
            return None
    
    def _scale_landmarks(self, landmarks: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Scale landmarks to target size"""
        if target_size == (112, 112):
            return landmarks.copy()
        
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] *= scale_x
        scaled_landmarks[:, 1] *= scale_y
        
        return scaled_landmarks
    
    def _get_similarity_transform(self, src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Compute similarity transformation matrix"""
        try:
            # Use only eye landmarks for more stable alignment
            src_eyes = src_landmarks[:2]  # First two landmarks (eyes)
            dst_eyes = dst_landmarks[:2]
            
            # Compute transformation matrix
            transformation_matrix, _ = cv2.estimateAffinePartial2D(
                src_eyes, dst_eyes, method=cv2.LMEDS
            )
            
            if transformation_matrix is None:
                # Fallback: use all available landmarks
                transformation_matrix, _ = cv2.estimateAffinePartial2D(
                    src_landmarks, dst_landmarks, method=cv2.LMEDS
                )
            
            return transformation_matrix
            
        except Exception as e:
            print(f"❌ Similarity transform error: {e}")
            return None
    
    def _enhance_face_quality(self, face_roi: np.ndarray) -> np.ndarray:
        """Enhance face quality with preprocessing"""
        try:
            # Apply CLAHE for better contrast
            if len(face_roi.shape) == 3:
                lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(lab[:,:,0])
                face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                face_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(face_roi)
            
            # Apply bilateral filter for noise reduction while preserving edges
            face_filtered = cv2.bilateralFilter(face_enhanced, 5, 50, 50)
            
            return face_filtered
            
        except Exception as e:
            print(f"❌ Face quality enhancement error: {e}")
            return face_roi
    
    def auto_align(self, frame: np.ndarray, detection: dict, 
                   method: str = "deepface", target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """Auto-align face using specified method"""
        try:
            bbox = detection.get('bbox', [])
            landmarks = detection.get('landmarks', [])
            
            if method == "yunet" and len(landmarks) >= 5:
                return self.align_yunet(frame, landmarks, target_size)
            elif method == "deepface" and len(landmarks) >= 2:
                return self.align_deepface(frame, landmarks, target_size)
            elif len(bbox) == 4:
                return self.align_bbox_enhanced(frame, bbox, target_size)
            else:
                print(f"⚠️ Insufficient data for {method} alignment")
                return None
                
        except Exception as e:
            print(f"❌ Auto-alignment error: {e}")
            return None