#!/usr/bin/env python3
"""
SFace Recognition from OpenCV Zoo
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Optional, List, Tuple

class SFaceRecognizer:
    """SFace face recognizer from OpenCV Zoo"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/face_recognition_sface_2021dec.onnx"
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.embedding_size = 128  # SFace embedding size
        self._initialize()
    
    def _initialize(self):
        """Initialize SFace recognizer"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ SFace model not found at: {self.model_path}")
                return False
            
            # Initialize ONNX Runtime session with options to suppress warnings
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # Only show errors
            
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Get input information
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"✅ SFace recognizer initialized successfully")
            print(f"   Input shape: {self.input_shape}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize SFace recognizer: {e}")
            return False
    
    def generate_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Generate face embedding using SFace"""
        if self.session is None:
            return None
        
        try:
            # Preprocess face for SFace model
            face_processed = self._preprocess_face(face_roi)
            
            if face_processed is None:
                return None
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: face_processed})
            embedding = outputs[0][0]  # Get first output, first batch
            
            # L2 normalize embedding
            embedding = self._l2_normalize(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"❌ SFace embedding generation error: {e}")
            return None
    
    def _preprocess_face(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face for SFace model"""
        try:
            # SFace expects 112x112 BGR image
            target_size = (112, 112)
            
            if face_roi.shape[:2] != target_size:
                face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
            else:
                face_resized = face_roi.copy()
            
            # Apply preprocessing similar to original SFace training
            # 1. Ensure BGR format
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_bgr = face_resized
            else:
                face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
            
            # 2. Normalize to [0, 1] range
            face_normalized = face_bgr.astype(np.float32) / 255.0
            
            # 3. Apply standard normalization (ImageNet-like)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_normalized = (face_normalized - mean) / std
            
            # 4. Convert to CHW format and add batch dimension
            face_chw = np.transpose(face_normalized, (2, 0, 1))  # HWC -> CHW
            face_batch = np.expand_dims(face_chw, axis=0)  # Add batch dimension
            
            return face_batch
            
        except Exception as e:
            print(f"❌ Face preprocessing error: {e}")
            return None
    
    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            # Ensure embeddings are L2 normalized
            emb1_norm = self._l2_normalize(embedding1)
            emb2_norm = self._l2_normalize(embedding2)
            
            # Cosine similarity (dot product of normalized vectors)
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Clamp to [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Similarity computation error: {e}")
            return 0.0
    
    def is_available(self) -> bool:
        """Check if recognizer is ready"""
        return self.session is not None
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       gallery_embeddings: List[np.ndarray], 
                       threshold: float = 0.4) -> Optional[Tuple[int, float]]:
        """Find best matching embedding from gallery"""
        if not gallery_embeddings or query_embedding is None:
            return None
        
        best_similarity = -1.0
        best_index = None
        
        for i, gallery_embedding in enumerate(gallery_embeddings):
            if gallery_embedding is None:
                continue
            
            similarity = self.compute_cosine_similarity(query_embedding, gallery_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_index = i
        
        if best_index is not None:
            return (best_index, best_similarity)
        
        return None