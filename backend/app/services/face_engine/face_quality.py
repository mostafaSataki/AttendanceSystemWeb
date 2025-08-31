#!/usr/bin/env python3
"""
Face Quality Assessment using EDIFFIQA from OpenCV Zoo
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Optional, Dict

class FaceQualityAssessment:
    """Face quality assessment using EDIFFIQA model"""
    
    def __init__(self, model_path: str = None, model_manager=None):
        self.model_path = model_path or "models/ediffiqa_tiny_jun2024.onnx"
        self.model_manager = model_manager
        self.session = None
        self.quality_threshold = 0.5
        self.input_size = (112, 112)
        self._initialize()
    
    def _initialize(self):
        """Initialize EDIFFIQA quality assessment model"""
        try:
            # Try to use model manager first
            if self.model_manager and hasattr(self.model_manager, 'quality_assessor') and self.model_manager.quality_assessor:
                self.session = self.model_manager.quality_assessor
                print("✅ Face quality assessor initialized from model manager")
                return True
            
            # Try to load model directly
            if os.path.exists(self.model_path):
                # Initialize ONNX Runtime session with options to suppress warnings
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Only show errors
                
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(
                    self.model_path, 
                    providers=providers,
                    sess_options=session_options
                )
                print(f"SUCCESS: Face quality assessor initialized successfully")
                return True
            
            print(f"WARNING: Quality assessment model not found, using fallback methods")
            return False
            
        except Exception as e:
            print(f"ERROR: Failed to initialize quality assessor: {e}")
            return False
    
    def assess_quality(self, face_roi: np.ndarray) -> Optional[float]:
        """Assess face image quality using EDIFFIQA model"""
        if self.session is None:
            return self._fallback_quality_assessment(face_roi)
        
        try:
            # Preprocess face for EDIFFIQA quality assessment
            face_processed = self._preprocess_for_quality(face_roi)
            
            if face_processed is None:
                return self._fallback_quality_assessment(face_roi)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: face_processed})
            
            # Get quality score
            quality_score = float(output[0][0])
            
            # EDIFFIQA outputs quality score, normalize if needed
            if quality_score > 1.0:
                quality_score = quality_score / 100.0  # Sometimes in 0-100 range
            
            # Ensure score is in valid range [0, 1]
            quality_score = max(0.0, min(1.0, quality_score))
            return quality_score
            
        except Exception as e:
            print(f"❌ EDIFFIQA quality assessment error: {e}")
            return self._fallback_quality_assessment(face_roi)
    
    def _preprocess_for_quality(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face for EDIFFIQA quality assessment"""
        try:
            # Resize to model input size
            if face_roi.shape[:2] != self.input_size:
                face_resized = cv2.resize(face_roi, self.input_size, interpolation=cv2.INTER_LINEAR)
            else:
                face_resized = face_roi.copy()
            
            # Ensure BGR format
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_bgr = face_resized
            else:
                face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
            
            # Normalize to [0, 1] range
            face_normalized = face_bgr.astype(np.float32) / 255.0
            
            # Convert to CHW format and add batch dimension
            face_chw = np.transpose(face_normalized, (2, 0, 1))  # HWC -> CHW
            face_batch = np.expand_dims(face_chw, axis=0)  # Add batch dimension
            
            return face_batch
            
        except Exception as e:
            print(f"❌ Quality preprocessing error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if quality assessor is ready"""
        return self.session is not None
    
    def _fallback_quality_assessment(self, face_roi: np.ndarray) -> float:
        """Simple quality assessment without AI model"""
        try:
            # Basic quality checks
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Check brightness (not too dark/bright)
            mean_brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            brightness_score = max(0.0, min(1.0, brightness_score))
            
            # Check contrast
            contrast = np.std(gray) / 255.0
            contrast_score = min(contrast * 2, 1.0)
            
            # Check sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # Combined score
            combined_score = (brightness_score * 0.3 + contrast_score * 0.3 + sharpness_score * 0.4)
            return max(0.3, min(0.9, combined_score))  # Keep in reasonable range
            
        except:
            return 0.6
    
    def is_good_quality(self, face_roi: np.ndarray) -> bool:
        """Check if face meets quality threshold"""
        quality_score = self.assess_quality(face_roi)
        if quality_score is None:
            return False
        return quality_score >= self.quality_threshold
    
    def get_quality_metrics(self, face_roi: np.ndarray) -> dict:
        """Get comprehensive quality metrics"""
        try:
            # Basic quality checks
            metrics = {
                'blur_score': self._calculate_blur(face_roi),
                'brightness_score': self._calculate_brightness(face_roi),
                'contrast_score': self._calculate_contrast(face_roi),
                'sharpness_score': self._calculate_sharpness(face_roi)
            }
            
            # AI-based quality score
            ai_quality = self.assess_quality(face_roi)
            if ai_quality is not None:
                metrics['ai_quality_score'] = ai_quality
            
            # Overall quality
            basic_scores = [v for k, v in metrics.items() if k != 'ai_quality_score']
            metrics['overall_quality'] = np.mean(basic_scores)
            
            return metrics
            
        except Exception as e:
            print(f"Quality metrics error: {e}")
            return {}
    
    def _calculate_blur(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 1.0)  # Normalize
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate brightness score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        # Optimal brightness is around 0.4-0.7
        if 0.4 <= mean_brightness <= 0.7:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(mean_brightness - 0.55) * 2)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 255.0
        return min(contrast * 2, 1.0)  # Normalize
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate sharpness score using gradient magnitude"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        return min(sharpness / 100.0, 1.0)  # Normalize