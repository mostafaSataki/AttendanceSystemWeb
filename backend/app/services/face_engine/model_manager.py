import os
import cv2
import numpy as np
import requests
from typing import Dict, Optional, Tuple
from pathlib import Path
from ...core.model_config import ModelConfig, MODEL_URLS

class ModelManager:
    def __init__(self):
        self.models_path = ModelConfig.MODELS_DIR
        self.models_path.mkdir(exist_ok=True)
        
        # Model URLs from centralized config
        self.model_urls = MODEL_URLS
        
        # Current active models
        self.active_models = {
            'face_detection': None,
            'face_recognition': None,
            'face_quality': None,
            'head_pose': None
        }
        
        # Model instances
        self.detector = None
        self.recognizer = None
        self.quality_assessor = None
        self.head_pose_estimator = None
    
    def download_model(self, category: str, model_name: str) -> bool:
        """Download model if not exists"""
        try:
            model_path = self.models_path / category / f"{model_name}.onnx"
            model_path.parent.mkdir(exist_ok=True)
            
            if model_path.exists():
                return True
                
            url = self.model_urls[category][model_name]
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            return False
    
    def load_detection_model(self, model_name: str) -> bool:
        """Load face detection model"""
        try:
            # Check if model_name already has .onnx extension
            if not model_name.endswith('.onnx'):
                model_filename = f"{model_name}.onnx"
            else:
                model_filename = model_name
            
            model_path = self.models_path / model_filename
            if not model_path.exists():
                print(f"Face detection model not found: {model_path}")
                return False
                
            self.detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                input_size=(320, 240),
                score_threshold=0.6,
                nms_threshold=0.3
            )
            self.active_models['face_detection'] = model_name
            return True
        except Exception as e:
            print(f"Failed to load detection model: {e}")
            return False
    
    def load_recognition_model(self, model_name: str) -> bool:
        """Load face recognition model"""
        try:
            model_path = self.models_path / 'face_recognition' / f"{model_name}.onnx"
            if not model_path.exists():
                return False
                
            self.recognizer = cv2.FaceRecognizerSF.create(
                str(model_path),
                ""
            )
            self.active_models['face_recognition'] = model_name
            return True
        except Exception as e:
            print(f"Failed to load recognition model: {e}")
            return False
    
    def load_quality_model(self, model_name: str) -> bool:
        """Load face quality assessment model"""
        try:
            # Check if model_name already has .onnx extension
            if not model_name.endswith('.onnx'):
                model_filename = f"{model_name}.onnx"
            else:
                model_filename = model_name
            
            model_path = self.models_path / model_filename
            if not model_path.exists():
                print(f"Face quality model not found: {model_path}")
                return False
                
            # Load ONNX model for quality assessment
            import onnxruntime as ort
            self.quality_assessor = ort.InferenceSession(str(model_path))
            self.active_models['face_quality'] = model_name
            return True
        except Exception as e:
            print(f"Failed to load quality model: {e}")
            return False
    
    def load_head_pose_model(self, model_name: str = None) -> bool:
        """Load head pose estimation model"""
        try:
            if model_name is None:
                model_name = ModelConfig.DEFAULT_HEAD_POSE_MODEL.replace('.onnx', '')
            
            model_filename = f"{model_name}.onnx" if not model_name.endswith('.onnx') else model_name
            model_path = self.models_path / model_filename
            
            if not model_path.exists():
                print(f"Head pose model not found: {model_path}")
                return False
                
            # Load ONNX model for head pose estimation
            import onnxruntime as ort
            self.head_pose_estimator = ort.InferenceSession(str(model_path))
            self.active_models['head_pose'] = model_name
            print(f"Head pose model loaded: {model_filename}")
            return True
        except Exception as e:
            print(f"Failed to load head pose model: {e}")
            return False
    
    def get_available_models(self) -> Dict:
        """Get list of available models"""
        available = {}
        for category, models in self.model_urls.items():
            available[category] = []
            for model_name in models.keys():
                model_path = self.models_path / category / f"{model_name}.onnx"
                available[category].append({
                    'name': model_name,
                    'downloaded': model_path.exists(),
                    'active': self.active_models[category] == model_name
                })
        
        # Add head pose model check using centralized config
        head_pose_model = self.models_path / ModelConfig.HEAD_POSE_MODEL_B66
        available['head_pose'] = [{
            'name': ModelConfig.HEAD_POSE_MODEL_B66.replace('.onnx', ''),
            'downloaded': head_pose_model.exists(),
            'active': self.active_models['head_pose'] == ModelConfig.HEAD_POSE_MODEL_B66.replace('.onnx', '')
        }]
        
        return available