# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Server
- **Recommended**: `run_server.bat` (Windows) or `python run_server.py`
- **Alternative**: `python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload`
- **Server URL**: http://localhost:8001  
- **API Documentation**: http://localhost:8001/docs

### Dependencies
- **Install**: `pip install -r requirements.txt`
- **Face recognition deps**: `pip install -r requirements_face_recognition.txt` (if needed)
- **Qt deps**: `pip install -r requirements_qt.txt` (if needed)

### Testing
- **Basic test**: `python test_database.py`
- No comprehensive test suite exists yet

### Database Management
- **Clear fake data**: `clear_fake_data.bat` (Windows) or `clear_fake_data.py`
- **Database file**: `attendance_system.db` (SQLite)

## Architecture Overview

### Core Structure
This is a **FastAPI-based face recognition attendance system** with the following key components:

1. **Face Recognition Engine** (`app/services/face_engine/`):
   - Multiple detection/recognition combinations (YuNet + SFace, DeepFace variants)
   - ONNX model support for face detection (`yunet_*.onnx`) and recognition (`sface_*.onnx`)
   - Face quality assessment using EDIFFIQA
   - Real-time processing with threading support

2. **API Layer** (`app/api/endpoints/`):
   - `people.py` - CRUD operations for person management
   - `enrollment.py` - Face enrollment workflow
   - `recognition.py` - Face recognition operations
   - `streaming.py` - RTSP streaming with analytics
   - `poses.py` - Head pose estimation

3. **Services Layer** (`app/services/`):
   - `face_recognition_service.py` - Main face recognition orchestrator
   - `rtsp_manager.py` - RTSP stream management
   - `analytics_integration.py` - Analytics pipeline integration
   - `backend_enrollment_service.py` - Enrollment process management

### Key Design Patterns

- **Lifespan Management**: FastAPI lifespan context manager initializes face recognition service on startup
- **State Management**: Face service attached to `app.state` for global access
- **Modular Face Engine**: Pluggable combinations of detection/recognition models
- **Thread-Safe Processing**: Queue-based processing for real-time recognition
- **CORS Configuration**: Configured for frontend at localhost:3000

### Model Files Structure
- Detection models: `models/face_detection/yunet_*.onnx`
- Recognition models: `models/face_recognition/sface_*.onnx`
- Quality assessment: `models/ediffiqa_tiny_jun2024.onnx`
- Model variants include optimized and quantized versions (`int8`, `int8bq`)

### Configuration
- Settings in `app/core/config.py` with environment variable support
- Face detection confidence: 0.5 (default)
- Face recognition threshold: 0.6 (default)
- Supports multiple DeepFace backends and models

### API Endpoints Pattern
- RESTful design with consistent response formats
- File upload support for face enrollment
- **Multi-pose enrollment**: `POST /api/enrollment/multi-pose` returns multiple face images (FRONT, LEFT, RIGHT, UP, DOWN)
- **Pose confirmation**: `POST /api/enrollment/confirm-enrollment` saves collected poses after user review
- Real-time streaming endpoints for RTSP integration
- Health check endpoint with face service status

### Face Enrollment Workflow
1. **Start Multi-Pose Collection**: Call `POST /api/enrollment/multi-pose` with source config
2. **Receive Multiple Images**: API returns base64-encoded images for each collected pose
3. **User Review**: Frontend displays all collected poses for user confirmation
4. **Save Enrollment**: Call `POST /api/enrollment/confirm-enrollment` to save confirmed poses

## Important Notes

- **Model Dependencies**: Face recognition requires ONNX models in the `models/` directory
- **Thread Safety**: Face recognition service uses threading for real-time processing
- **Database**: Uses SQLite by default, configurable via DATABASE_URL
- **CORS**: Frontend expected at http://localhost:3000
- **File Storage**: Face images and encodings stored in respective upload directories