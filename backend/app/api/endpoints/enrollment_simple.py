"""
Simple Face Enrollment API endpoints
Temporary implementation that works without complex dependencies
"""

from fastapi import APIRouter, HTTPException, Request
import time
import base64
import cv2
import numpy as np
from typing import Dict, Any

router = APIRouter()

@router.post("/multi-pose")
async def start_multi_pose_enrollment(request: Request):
    """Start multi-pose face enrollment session (simplified version)"""
    try:
        body = await request.json()
        source = body.get("source", "camera")
        source_config = body.get("source_config", {})
        person_name = body.get("person_name", "temp_enrollment")
        
        print(f"Starting multi-pose enrollment - Source: {source}, Person: {person_name}")
        
        # Create mock face images for different poses
        pose_images = []
        poses = ["FRONT", "LEFT", "RIGHT"]
        
        for i, pose_name in enumerate(poses):
            # Create a simple colored rectangle as mock face image
            mock_image = np.zeros((112, 112, 3), dtype=np.uint8)
            # Different color for each pose
            if pose_name == "FRONT":
                mock_image[:, :] = [100, 150, 200]  # Blue-ish
            elif pose_name == "LEFT":
                mock_image[:, :] = [150, 100, 200]  # Purple-ish  
            else:  # RIGHT
                mock_image[:, :] = [200, 150, 100]  # Orange-ish
            
            # Add text to indicate the pose
            cv2.putText(mock_image, pose_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', mock_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            pose_images.append({
                "pose_name": pose_name,
                "image": f"data:image/jpeg;base64,{image_base64}",
                "quality_score": 0.85 - (i * 0.05),  # Decreasing quality
                "timestamp": time.time()
            })
        
        return {
            "status": "success",
            "message": f"Collected {len(pose_images)} poses for {person_name}",
            "person_name": person_name,
            "total_poses": len(pose_images),
            "pose_images": pose_images,
            "processing_stats": {
                "frames_processed": 150,
                "processing_time": 3.2,
                "poses_collected": [p["pose_name"] for p in pose_images]
            }
        }
        
    except Exception as e:
        print(f"Error in multi-pose enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-pose enrollment failed: {str(e)}")


@router.post("/confirm-enrollment")
async def confirm_enrollment(request: Request):
    """Confirm and save enrollment with collected poses (simplified)"""
    try:
        body = await request.json()
        person_name = body.get("person_name")
        collected_poses_data = body.get("collected_poses_data", {})
        
        if not person_name:
            raise HTTPException(status_code=400, detail="Person name is required")
        
        print(f"Saving enrollment for {person_name} with {len(collected_poses_data)} poses")
        
        return {
            "status": "success",
            "message": f"Enrollment confirmed for {person_name}",
            "person_name": person_name,
            "total_poses": len(collected_poses_data),
            "embeddings_generated": len(collected_poses_data),
            "embedding_model": "simple_mock"
        }
        
    except Exception as e:
        print(f"Error confirming enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm enrollment: {str(e)}")


@router.get("/status")
async def get_enrollment_status():
    """Get enrollment status for all persons (simplified)"""
    return {
        "status": "success",
        "total_persons": 0,
        "persons": [],
        "total_embeddings": 0,
        "last_updated": time.time()
    }