from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import numpy as np

class EnrollmentStatus(str, Enum):
    WAITING = "waiting"
    CAPTURING = "capturing"
    COMPLETED = "completed"
    FAILED = "failed"

class PoseState(str, Enum):
    FRONT = "FRONT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"

class EnrollmentProgress(BaseModel):
    """Current enrollment progress"""
    status: EnrollmentStatus
    current_target: PoseState
    captured_poses: List[PoseState]
    progress: float = Field(..., ge=0, le=1, description="Progress as a ratio 0-1")
    message: str
    quality_score: float = Field(..., ge=0, le=1)
    pose_detected: List[PoseState]
    face_region: Optional[Tuple[int, int, int, int]] = None
    countdown: float = Field(default=0.0, ge=0)

class EnrollmentImageData(BaseModel):
    """Metadata for an enrollment image"""
    pose_state: PoseState
    quality_score: float = Field(..., ge=0, le=1)
    timestamp: datetime
    face_region: Tuple[int, int, int, int]
    filename: str

class EnrollmentMetadata(BaseModel):
    """Complete enrollment metadata"""
    user_id: str
    enrollment_time: datetime
    poses: Dict[PoseState, EnrollmentImageData]
    enrollment_path: str

class EnrollmentStartRequest(BaseModel):
    """Request to start enrollment process"""
    user_id: str
    quality_threshold: float = Field(default=0.6, ge=0, le=1)

class EnrollmentCompleteRequest(BaseModel):
    """Request to save completed enrollment"""
    user_id: str
    save_path: Optional[str] = None

class EnrollmentConfigResponse(BaseModel):
    """Enrollment system configuration"""
    required_poses: List[PoseState]
    quality_threshold: float
    pose_hold_time: float
    min_face_size: int
    models_available: bool

class FaceQualityMetrics(BaseModel):
    """Detailed face quality metrics"""
    blur_score: float = Field(..., ge=0, le=1)
    brightness_score: float = Field(..., ge=0, le=1)
    contrast_score: float = Field(..., ge=0, le=1)
    sharpness_score: float = Field(..., ge=0, le=1)
    ai_quality_score: Optional[float] = Field(None, ge=0, le=1)
    overall_quality: float = Field(..., ge=0, le=1)

class EnrollmentSessionStatus(BaseModel):
    """Current enrollment session status"""
    session_id: str
    user_id: Optional[str] = None
    status: EnrollmentStatus
    started_at: datetime
    updated_at: datetime
    progress: EnrollmentProgress
    quality_metrics: Optional[FaceQualityMetrics] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    status_code: int = 400