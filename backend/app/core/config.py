"""
Configuration settings for the application
"""

import os
from pathlib import Path
from typing import Optional

class Settings:
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Attendance System API"
    VERSION: str = "1.0.0"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./attendance_system.db")
    
    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    
    # File Storage
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    FACE_IMAGES_DIR: Path = BASE_DIR / "face_images"
    ENROLLMENTS_DIR: Path = BASE_DIR / "enrollments"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Create directories if they don't exist
    def __post_init__(self):
        for directory in [self.UPLOAD_DIR, self.FACE_IMAGES_DIR, self.ENROLLMENTS_DIR, self.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # DeepFace Settings
    DEEPFACE_MODEL: str = "VGG-Face"  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
    DEEPFACE_BACKEND: str = "opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface
    DEEPFACE_ENFORCE_DETECTION: bool = False
    
    # Camera Settings
    DEFAULT_CAMERA_INDEX: int = 0
    CAMERA_RESOLUTION: tuple = (640, 480)
    CAMERA_FPS: int = 30

settings = Settings()

# Create required directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FACE_IMAGES_DIR, exist_ok=True)
os.makedirs(settings.ENROLLMENTS_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)