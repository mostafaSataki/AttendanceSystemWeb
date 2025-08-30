"""
Model configuration constants for the face enrollment system
"""

from pathlib import Path

class ModelConfig:
    """Centralized model configuration"""
    
    # Base models directory
    MODELS_DIR = Path("models")
    
    # Face Detection Models
    FACE_DETECTION_YUNET_2023MAR = "face_detection_yunet_2023mar.onnx"
    FACE_DETECTION_YUNET_2023MAR_INT8 = "face_detection_yunet_2023mar_int8.onnx"
    FACE_DETECTION_YUNET_2023MAR_INT8BQ = "face_detection_yunet_2023mar_int8bq.onnx"
    
    # Face Recognition Models
    FACE_RECOGNITION_SFACE_2021DEC = "face_recognition_sface_2021dec.onnx"
    FACE_RECOGNITION_SFACE_2021DEC_INT8 = "face_recognition_sface_2021dec_int8.onnx"
    FACE_RECOGNITION_SFACE_2021DEC_INT8BQ = "face_recognition_sface_2021dec_int8bq.onnx"
    
    # Face Quality Assessment Models
    FACE_QUALITY_EDIFFIQA_TINY = "ediffiqa_tiny_jun2024.onnx"
    FACE_QUALITY_EDIFFIQA_2022OCT = "face_image_quality_assessment_ediffiqa_2022oct.onnx"
    
    # Head Pose Estimation Models
    HEAD_POSE_MODEL_B66 = "model-b66.onnx"
    
    # Default models for enrollment system
    DEFAULT_FACE_DETECTION_MODEL = FACE_DETECTION_YUNET_2023MAR
    DEFAULT_FACE_QUALITY_MODEL = FACE_QUALITY_EDIFFIQA_TINY
    DEFAULT_HEAD_POSE_MODEL = HEAD_POSE_MODEL_B66
    DEFAULT_FACE_RECOGNITION_MODEL = FACE_RECOGNITION_SFACE_2021DEC
    
    # Required models for enrollment
    ENROLLMENT_REQUIRED_MODELS = [
        DEFAULT_FACE_DETECTION_MODEL,
        DEFAULT_FACE_QUALITY_MODEL,
        DEFAULT_HEAD_POSE_MODEL
    ]
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get full path for a model file"""
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def check_model_exists(cls, model_name: str) -> bool:
        """Check if a model file exists"""
        return cls.get_model_path(model_name).exists()
    
    @classmethod
    def get_missing_enrollment_models(cls) -> list:
        """Get list of missing models required for enrollment"""
        missing = []
        for model in cls.ENROLLMENT_REQUIRED_MODELS:
            if not cls.check_model_exists(model):
                missing.append(model)
        return missing
    
    @classmethod
    def validate_enrollment_models(cls) -> bool:
        """Check if all required enrollment models are available"""
        return len(cls.get_missing_enrollment_models()) == 0

# Model URLs for downloading
MODEL_URLS = {
    'face_detection': {
        'yunet_2023mar': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'yunet_2023mar_int8': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx',
        'yunet_2023mar_int8bq': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8bq.onnx'
    },
    'face_recognition': {
        'sface_2021dec': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        'sface_2021dec_int8': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx',
        'sface_2021dec_int8bq': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx'
    },
    'face_quality': {
        'ediffiqa_tiny': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx'
    }
}