#!/usr/bin/env python3
"""
FastAPI Backend for Attendance System Web Application
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.api.endpoints import enrollment_video as enrollment, recognition, people, poses
# from app.api.endpoints import streaming  # Temporarily disabled due to jwt dependency
from app.core.config import settings
from app.services.face_recognition_service import FaceRecognitionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup  
    print("Starting Video Enrollment Service...")
    # Use simple face service for video enrollment
    app.state.face_service = None
    
    yield
    
    # Shutdown
    print("Shutting down Face Recognition Service...")
    if hasattr(app.state, 'face_service') and app.state.face_service:
        await app.state.face_service.cleanup()


app = FastAPI(
    title="Attendance System API",
    description="Face Recognition Attendance System Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(enrollment.router, prefix="/api/enrollment", tags=["enrollment"])
app.include_router(recognition.router, prefix="/api/recognition", tags=["recognition"])
app.include_router(people.router, prefix="/api/people", tags=["people"])
app.include_router(poses.router, prefix="/api/poses", tags=["poses"])
# app.include_router(streaming.router, prefix="/api/stream", tags=["streaming"])  # Temporarily disabled


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Attendance System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face-recognition-api",
        "face_service_status": hasattr(app.state, 'face_service') and app.state.face_service.is_initialized
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )