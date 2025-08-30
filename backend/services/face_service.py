import cv2
import face_recognition
import numpy as np
import pickle
import os
from typing import Tuple, Optional

class FaceService:
    def __init__(self):
        self.tolerance = 0.6
        
    def process_face(self, image_path: str) -> Tuple[bytes, int]:
        """
        Process an image file and return face encoding and confidence score
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                raise ValueError("No face found in the image")
            
            if len(face_locations) > 1:
                raise ValueError("Multiple faces found in the image")
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) == 0:
                raise ValueError("Could not generate face encoding")
            
            # Calculate confidence score (simplified)
            face_encoding = face_encodings[0]
            confidence_score = self._calculate_confidence_score(image, face_locations[0])
            
            # Convert to bytes for storage
            encoding_bytes = pickle.dumps(face_encoding)
            
            return encoding_bytes, confidence_score
            
        except Exception as e:
            raise Exception(f"Face processing failed: {str(e)}")
    
    def _calculate_confidence_score(self, image: np.ndarray, face_location: tuple) -> int:
        """
        Calculate a confidence score based on face detection quality
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract face region
            top, right, bottom, left = face_location
            face_image = cv_image[top:bottom, left:right]
            
            # Calculate basic metrics
            face_size = bottom - top
            image_size = min(image.shape[:2])
            size_ratio = face_size / image_size
            
            # Calculate blur detection (Laplacian variance)
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Normalize scores and calculate final confidence
            size_score = min(size_ratio * 100, 100)  # Max 100 for size
            blur_normalized = min(blur_score / 100, 100)  # Normalize blur score
            
            # Combined confidence score
            confidence = int((size_score + blur_normalized) / 2)
            confidence = max(0, min(100, confidence))  # Ensure 0-100 range
            
            return confidence
            
        except Exception:
            return 75  # Default confidence if calculation fails
    
    def save_face_encoding(self, face_encoding: bytes, file_path: str):
        """
        Save face encoding to file
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(face_encoding)
        except Exception as e:
            raise Exception(f"Failed to save face encoding: {str(e)}")
    
    def load_face_encoding(self, file_path: str) -> bytes:
        """
        Load face encoding from file
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to load face encoding: {str(e)}")
    
    def compare_faces(self, known_encoding_bytes: bytes, unknown_encoding_bytes: bytes) -> float:
        """
        Compare two face encodings and return similarity score
        """
        try:
            known_encoding = pickle.loads(known_encoding_bytes)
            unknown_encoding = pickle.loads(unknown_encoding_bytes)
            
            # Calculate face distance
            face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            
            # Convert to similarity score (0-100)
            similarity = (1 - face_distance) * 100
            return max(0, min(100, similarity))
            
        except Exception as e:
            raise Exception(f"Face comparison failed: {str(e)}")
    
    def recognize_face(self, unknown_encoding_bytes: bytes, known_encodings: list) -> Optional[dict]:
        """
        Recognize face from a list of known encodings
        """
        try:
            unknown_encoding = pickle.loads(unknown_encoding_bytes)
            known_encodings_list = [pickle.loads(enc) for enc in known_encodings]
            
            # Find best match
            face_distances = face_recognition.face_distance(known_encodings_list, unknown_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] <= self.tolerance:
                similarity = (1 - face_distances[best_match_index]) * 100
                return {
                    "index": best_match_index,
                    "similarity": similarity,
                    "distance": face_distances[best_match_index]
                }
            
            return None
            
        except Exception as e:
            raise Exception(f"Face recognition failed: {str(e)}")
    
    def detect_faces_in_image(self, image_path: str) -> list:
        """
        Detect all faces in an image and return their locations
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            return face_locations
        except Exception as e:
            raise Exception(f"Face detection failed: {str(e)}")