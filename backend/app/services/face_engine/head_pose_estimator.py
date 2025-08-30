import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import os
from pathlib import Path
from ...core.model_config import ModelConfig
from ...models.enrollment import PoseState

class HeadPoseEstimator:
    """Head pose estimation using ONNX model"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model_filename = ModelConfig.HEAD_POSE_MODEL_B66
        self.YAW_THRESHOLD = 25.0
        self.PITCH_THRESHOLD = 20.0
        self.model_img_size = 224
        self.session = None
        self.input_name = None
        
        self._init_session()
    
    def _init_session(self):
        """Initialize ONNX Runtime session"""
        try:
            full_model_path = ModelConfig.get_model_path(self.model_filename)
            
            if full_model_path.exists():
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(str(full_model_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                print(f"Head pose model loaded successfully: {self.model_filename}")
            else:
                print(f"Head pose model not found at {full_model_path}")
                self.session = None
            
        except Exception as e:
            print(f"Error loading head pose model: {e}")
            self.session = None
    
    def determine_head_pose_state_multi(self, roll: float, yaw: float, pitch: float) -> List[PoseState]:
        """Determine multiple head pose states from angles"""
        states = []
        ANGLE_THRESHOLD = 10.0  # Lowered from 13.0 to match landmark-based model
        
        if pitch < -ANGLE_THRESHOLD:
            states.append(PoseState.DOWN)
        elif pitch > ANGLE_THRESHOLD:
            states.append(PoseState.UP)
        
        if yaw < -ANGLE_THRESHOLD:
            states.append(PoseState.LEFT)
        elif yaw > ANGLE_THRESHOLD:
            states.append(PoseState.RIGHT)
        
        if not states:
            states.append(PoseState.FRONT)
        
        return states
    
    def extract_head_pose(self, face_roi: np.ndarray) -> Tuple[List[float], bool]:
        """Extract head pose from face region"""
        headpose = []
        success = False
        
        if self.session is not None and face_roi is not None and face_roi.size > 0:
            try:
                face_inputs = self._preprocess(face_roi)
                
                if face_inputs is not None:
                    input_shape = [1, 3, self.model_img_size, self.model_img_size]
                    face_inputs = face_inputs.reshape(input_shape).astype(np.float32)
                    
                    outputs = self.session.run(None, {self.input_name: face_inputs})
                    
                    roll = float(outputs[0][0])
                    yaw = float(outputs[1][0])
                    pitch = float(outputs[2][0])
                    
                    headpose = [roll, yaw, pitch]
                    success = True
            except Exception as e:
                print(f"Head pose estimation error: {e}")
        
        if not success:
            # Return neutral pose if estimation fails
            headpose = [0.0, 0.0, 0.0]
        
        return headpose, success
    
    def get_pose_states(self, face_roi: np.ndarray) -> List[PoseState]:
        """Get pose states from face region"""
        headpose, success = self.extract_head_pose(face_roi)
        
        if success and len(headpose) >= 3:
            roll, yaw, pitch = headpose
            return self.determine_head_pose_state_multi(roll, yaw, pitch)
        else:
            return [PoseState.FRONT]  # Default to front pose
    
    def _preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for model input"""
        if img is None or img.size == 0:
            return None
        
        try:
            # Resize to model input size
            resized_img = cv2.resize(img, (self.model_img_size, self.model_img_size))
            
            # Convert to RGB if needed
            if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized_img = resized_img.astype(np.float32) / 255.0
            
            # Normalize with ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            for i in range(3):
                normalized_img[:, :, i] = (normalized_img[:, :, i] - mean[i]) / std[i]
            
            # Convert to CHW format
            img_data = np.transpose(normalized_img, (2, 0, 1))
            return img_data
            
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return None
    
    def is_model_available(self) -> bool:
        """Check if head pose model is available"""
        return self.session is not None