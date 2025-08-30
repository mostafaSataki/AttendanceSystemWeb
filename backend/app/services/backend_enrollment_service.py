#!/usr/bin/env python3
"""
Backend Enrollment Service - Pure business logic without UI dependencies
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("⚠️ DeepFace not available. Embeddings will be generated using existing face engine.")
    DEEPFACE_AVAILABLE = False

from .face_engine import FaceRecognitionCombinationEngine, CombinationType
from .face_engine.head_pose_estimator import HeadPoseEstimator, PoseState
from .face_engine.model_manager import ModelManager

class BackendEnrollmentService:
    """Pure backend enrollment service without any display dependencies"""
    
    def __init__(self, combination_type: CombinationType = CombinationType.YUNET_SFACE):
        self.base_path = Path("face_images")
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize face recognition engine
        self.face_engine = FaceRecognitionCombinationEngine(combination_type)
        
        # Initialize model manager and head pose estimator
        self.model_manager = ModelManager()
        self.head_pose_estimator = HeadPoseEstimator(self.model_manager)
        
        # Enrollment configuration (aligned with working CLI settings)
        self.pose_stability_frames = 2  # Reduced to 2 for easier collection (matches CLI)
        self.quality_threshold = 0.4  # Match CLI quality threshold (was 0.3)
        
        # Set face engine quality threshold to match CLI
        self.face_engine.set_thresholds(quality_threshold=self.quality_threshold)
        self.max_frames_per_pose = 200  # Reduced max frames
        self.enrollment_timeout = 90.0  # Match CLI timeout (90s)
        self.frame_skip = 3  # Process every 3rd frame (matches CLI)
        self.max_processed_frames = 600  # Match CLI max frames
        
        # Load enrollment database
        self.enrollment_database = self._load_enrollment_database()
        
        # DeepFace configuration for embeddings
        self.detector_backend = "opencv"
        self.recognition_model = "SFace"
        
        print(f"✅ Backend enrollment service initialized")
        if DEEPFACE_AVAILABLE:
            print(f"✅ DeepFace available for embedding generation (model: {self.recognition_model})")
        else:
            print(f"⚠️ Will use existing face engine for embedding generation")

    def _load_enrollment_database(self) -> Dict[str, Any]:
        """Load enrollment database"""
        db_path = self.base_path / "enrollment_database.json"
        try:
            if db_path.exists():
                with open(db_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load enrollment database: {e}")
        
        return {
            'version': '1.0',
            'persons': {},
            'last_updated': datetime.now().isoformat()
        }

    def _save_enrollment_database(self):
        """Save enrollment database"""
        db_path = self.base_path / "enrollment_database.json"
        try:
            self.enrollment_database['last_updated'] = datetime.now().isoformat()
            with open(db_path, 'w') as f:
                json.dump(self.enrollment_database, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving enrollment database: {e}")

    def process_frame_for_enrollment(self, frame: np.ndarray, enrollment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single frame for enrollment - pure backend logic"""
        result = {
            'success': False,
            'candidates': [],
            'pose_detected': None,
            'frame_info': {
                'frame_number': enrollment_state.get('frame_count', 0),
                'pose_stability': {},
                'collected_poses': list(enrollment_state.get('collected_poses', {}).keys())
            },
            'enrollment_complete': False,
            'should_show_review': False,
            'debug_info': {}
        }
        
        try:
            # Process frame with face engine
            detection_result = self.face_engine.process_frame_for_enrollment(frame)
            
            if not detection_result.get('enrollment_candidates'):
                result['debug_info']['no_candidates'] = True
                return result
            
            # Get best candidate with improved selection criteria
            best_candidate = self._select_best_candidate(detection_result['enrollment_candidates'], frame.shape)
            
            result['candidates'] = [best_candidate]
            result['success'] = True
            
            # Estimate head pose using aligned face
            aligned_face = best_candidate.get('aligned_face')
            if aligned_face is not None:
                pose_states = self.head_pose_estimator.get_pose_states(aligned_face)
                if pose_states:
                    current_pose = pose_states[0]  # Take first detected pose
                    result['pose_detected'] = current_pose.value
                    result['debug_info']['detected_poses'] = [p.value for p in pose_states]
                else:
                    current_pose = PoseState.FRONT  # Default pose
                    result['pose_detected'] = current_pose.value
            else:
                current_pose = PoseState.FRONT  # Default pose
                result['pose_detected'] = current_pose.value
            
            # Update pose stability
            if current_pose not in enrollment_state['pose_stability']:
                enrollment_state['pose_stability'][current_pose] = 0
            
            enrollment_state['pose_stability'][current_pose] += 1
            
            # Reset stability for other poses
            for pose in PoseState:
                if pose != current_pose:
                    enrollment_state['pose_stability'][pose] = 0
            
            # Check if pose is stable enough for collection
            required_stability = 1 if current_pose == PoseState.RIGHT else self.pose_stability_frames
            if (enrollment_state['pose_stability'][current_pose] >= required_stability and
                current_pose not in enrollment_state['collected_poses']):
                
                # Collect this pose
                enrollment_state['collected_poses'][current_pose] = {
                    'aligned_face': best_candidate['aligned_face'],
                    'quality_score': best_candidate['quality_score'],
                    'frame_number': enrollment_state['frame_count'],
                    'timestamp': datetime.now().isoformat()
                }
                
                result['debug_info']['pose_collected'] = current_pose.value
            
            # Update frame info
            result['frame_info']['pose_stability'] = {
                pose.value: stability for pose, stability in enrollment_state['pose_stability'].items()
            }
            
            # Check if enrollment is complete (all 5 poses) or sufficient (at least 3 poses including FRONT)
            required_poses = {PoseState.FRONT, PoseState.LEFT, PoseState.RIGHT, PoseState.UP, PoseState.DOWN}
            collected_poses = set(enrollment_state['collected_poses'].keys())
            
            # Full enrollment (all 5 poses)
            result['enrollment_complete'] = required_poses.issubset(collected_poses)
            
            # Minimum viable enrollment (at least 3 poses with FRONT required)
            minimum_poses = 3
            has_front_pose = PoseState.FRONT in collected_poses
            result['enrollment_sufficient'] = len(collected_poses) >= minimum_poses and has_front_pose
            
            # Should show review only when enrollment is complete or sufficient (not just any poses)
            result['should_show_review'] = result['enrollment_complete'] or result['enrollment_sufficient']
            
            return result
            
        except Exception as e:
            result['debug_info']['error'] = str(e)
            print(f"❌ Frame processing error: {e}")
            return result

    def initialize_enrollment_state(self, person_name: str) -> Dict[str, Any]:
        """Initialize enrollment state for a person"""
        return {
            'person_name': person_name,
            'frame_count': 0,
            'start_time': time.time(),
            'pose_stability': {},
            'collected_poses': {},
            'cancelled': False,
            'completed': False,
            'last_activity_time': time.time()
        }

    def save_enrollment_data(self, person_name: str, collected_poses: Dict) -> Dict[str, Any]:
        """Save enrollment data to disk with embeddings"""
        try:
            print(f"🔧 Saving enrollment for {person_name}, poses: {list(collected_poses.keys())}")
            print(f"🔧 Pose key types: {[type(k).__name__ for k in collected_poses.keys()]}")
            
            person_folder = self.base_path / person_name
            person_folder.mkdir(exist_ok=True)
            
            # Save each pose image and generate embeddings
            saved_poses = {}
            embeddings = {}
            
            for pose_name, pose_data in collected_poses.items():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{person_name}_{pose_name}_{timestamp}.jpg"
                file_path = person_folder / filename
                
                # Save aligned face image
                cv2.imwrite(str(file_path), pose_data['aligned_face'])
                print(f"💾 Saved pose image: {filename}")
                
                # Generate embedding for this pose
                print(f"🧠 Generating embedding for {pose_name} pose...")
                embedding = self.get_face_embedding(str(file_path))
                
                saved_poses[pose_name] = {
                    'filename': filename,
                    'quality_score': pose_data['quality_score'],
                    'frame_number': pose_data['frame_number'],
                    'timestamp': pose_data.get('timestamp', datetime.now().isoformat()),
                    'has_embedding': embedding is not None
                }
                
                # Store embedding separately (convert to list for JSON serialization)
                if embedding is not None:
                    embeddings[pose_name] = embedding.tolist()
                    print(f"✅ Generated embedding for {pose_name} (size: {embedding.shape})")
                else:
                    print(f"⚠️ Failed to generate embedding for {pose_name}")
            
            # Update enrollment database with embeddings
            self.enrollment_database['persons'][person_name] = {
                'poses': saved_poses,
                'embeddings': embeddings,  # Store embeddings in database
                'enrolled_date': datetime.now().isoformat(),
                'total_poses': len(saved_poses),
                'embedding_model': self.recognition_model if DEEPFACE_AVAILABLE else 'face_engine',
                'detector_backend': self.detector_backend if DEEPFACE_AVAILABLE else 'yunet'
            }
            
            self._save_enrollment_database()
            
            return {
                'success': True,
                'person_name': person_name,
                'saved_poses': saved_poses,
                'total_poses': len(saved_poses),
                'embeddings_generated': len(embeddings),
                'embedding_model': self.recognition_model if DEEPFACE_AVAILABLE else 'face_engine'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'person_name': person_name
            }

    def enroll_from_video_backend(self, video_path: str, person_name: str, 
                                 progress_callback: Optional[Callable] = None,
                                 frame_callback: Optional[Callable] = None,
                                 auto_save: bool = True) -> Dict[str, Any]:
        """Backend video enrollment without any display logic"""
        if not os.path.exists(video_path):
            return {'success': False, 'error': 'Video file not found'}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video file'}
        
        # Initialize enrollment state
        enrollment_state = self.initialize_enrollment_state(person_name)
        
        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        result = {
            'success': False,
            'person_name': person_name,
            'total_frames_processed': 0,
            'poses_collected': [],
            'processing_time': 0,
            'cancelled': False
        }
        
        start_time = time.time()
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Video ended - if we have collected poses, mark for review
                    if enrollment_state.get('collected_poses'):
                        enrollment_state['video_ended'] = True
                    break
                
                frame_counter += 1
                
                # Skip frames for performance (process every nth frame)
                if frame_counter % self.frame_skip != 0:
                    continue
                
                # Stop processing after max frames to prevent hanging
                if frame_counter > self.max_processed_frames:
                    print(f"⏰ Stopped processing after {self.max_processed_frames} frames for performance")
                    break
                
                enrollment_state['frame_count'] = frame_counter
                
                # Process frame
                frame_result = self.process_frame_for_enrollment(frame, enrollment_state)
                
                # Call frame callback if provided (for frontend to handle display)
                if frame_callback:
                    should_continue = frame_callback(
                        frame=frame,
                        frame_result=frame_result,
                        enrollment_state=enrollment_state
                    )
                    if not should_continue:
                        enrollment_state['cancelled'] = True
                        break
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        current_frame=enrollment_state['frame_count'],
                        total_frames=total_frames,
                        collected_poses=len(enrollment_state['collected_poses']),
                        current_pose=frame_result.get('pose_detected')
                    )
                
                # Check if enrollment is complete
                if frame_result.get('enrollment_complete'):
                    enrollment_state['completed'] = True
                    break
                
                # Check timeout
                if time.time() - start_time > self.enrollment_timeout:
                    break
            
            result['total_frames_processed'] = enrollment_state['frame_count']
            result['processing_time'] = time.time() - start_time
            result['cancelled'] = enrollment_state.get('cancelled', False)
            
            # After processing, check the results
            collected_poses = enrollment_state.get('collected_poses', {})
            
            # If not cancelled and we have poses, it's a success (for the GUI to handle)
            if not enrollment_state.get('cancelled') and collected_poses:
                result['success'] = True
                result['person_name'] = person_name
                result['poses_collected'] = list(collected_poses.keys())
                result['collected_poses_data'] = collected_poses
                result['video_ended'] = enrollment_state.get('video_ended', False)
                # Only show review when video ended, enrollment completed, or sufficient poses collected
                result['should_show_review'] = (
                    enrollment_state.get('video_ended', False) or  # Video reached end
                    enrollment_state.get('completed', False) or    # All poses collected
                    len(collected_poses) >= 3  # Sufficient poses (will be checked by GUI logic)
                )
                
                if auto_save:
                    save_result = self.save_enrollment_data(person_name, collected_poses)
                    result.update(save_result)
            
            elif enrollment_state.get('cancelled'):
                result['error'] = 'Enrollment was cancelled by the user.'
                
            else: # Not cancelled, but no poses
                result['error'] = 'Enrollment finished, but no poses were collected.'
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
            
        finally:
            cap.release()

    def enroll_from_camera_stream(self, camera_source, person_name: str, 
                                 progress_callback: Optional[Callable] = None,
                                 frame_callback: Optional[Callable] = None,
                                 auto_save: bool = True,
                                 max_enrollment_time: float = 120.0) -> Dict[str, Any]:
        """Backend camera stream enrollment for IP cameras and webcams"""
        
        # Open camera/stream
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            return {'success': False, 'error': f'Could not open camera/stream: {camera_source}'}
        
        # Try to set some reasonable defaults for cameras
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass  # Ignore if camera doesn't support these properties
        
        # Initialize enrollment state
        enrollment_state = self.initialize_enrollment_state(person_name)
        
        result = {
            'success': False,
            'person_name': person_name,
            'total_frames_processed': 0,
            'poses_collected': [],
            'processing_time': 0,
            'cancelled': False
        }
        
        start_time = time.time()
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # For cameras, ret=False might mean temporary issue, try a few times
                    cap.release()
                    time.sleep(0.1)  # Wait 100ms
                    cap = cv2.VideoCapture(camera_source)
                    if not cap.isOpened():
                        break
                    continue
                
                frame_counter += 1
                
                # Skip frames for performance (process every nth frame)
                if frame_counter % self.frame_skip != 0:
                    continue
                
                # Stop processing after max frames to prevent hanging
                if frame_counter > self.max_processed_frames:
                    print(f"⏰ Stopped processing after {self.max_processed_frames} frames for performance")
                    break
                
                enrollment_state['frame_count'] = frame_counter
                
                # Process frame
                frame_result = self.process_frame_for_enrollment(frame, enrollment_state)
                
                # Call frame callback if provided (for frontend to handle display)
                if frame_callback:
                    should_continue = frame_callback(
                        frame=frame,
                        frame_result=frame_result,
                        enrollment_state=enrollment_state
                    )
                    if not should_continue:
                        enrollment_state['cancelled'] = True
                        break
                
                # Call progress callback if provided (for streaming, total_frames is unknown)
                if progress_callback:
                    progress_callback(
                        current_frame=enrollment_state['frame_count'],
                        total_frames=-1,  # Unknown for streaming
                        collected_poses=len(enrollment_state['collected_poses']),
                        current_pose=frame_result.get('pose_detected')
                    )
                
                # Check if enrollment is complete
                if frame_result.get('enrollment_complete'):
                    enrollment_state['completed'] = True
                    break
                
                # Check timeout
                if time.time() - start_time > max_enrollment_time:
                    print(f"⏰ Camera enrollment timeout after {max_enrollment_time}s")
                    break
            
            result['total_frames_processed'] = enrollment_state['frame_count']
            result['processing_time'] = time.time() - start_time
            result['cancelled'] = enrollment_state.get('cancelled', False)
            
            # After processing, check the results
            collected_poses = enrollment_state.get('collected_poses', {})
            
            # If not cancelled and we have poses, it's a success (for the GUI to handle)
            if not enrollment_state.get('cancelled') and collected_poses:
                result['success'] = True
                result['person_name'] = person_name
                result['poses_collected'] = list(collected_poses.keys())
                result['collected_poses_data'] = collected_poses
                result['video_ended'] = enrollment_state.get('video_ended', False)
                # Only show review when video ended, enrollment completed, or sufficient poses collected
                result['should_show_review'] = (
                    enrollment_state.get('video_ended', False) or  # Video reached end
                    enrollment_state.get('completed', False) or    # All poses collected
                    len(collected_poses) >= 3  # Sufficient poses (will be checked by GUI logic)
                )
                
                if auto_save:
                    save_result = self.save_enrollment_data(person_name, collected_poses)
                    result.update(save_result)
            
            elif enrollment_state.get('cancelled'):
                result['error'] = 'Enrollment was cancelled by the user.'
                
            else: # Not cancelled, but no poses
                result['error'] = 'Enrollment finished, but no poses were collected.'
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
            
        finally:
            cap.release()

    def save_pending_enrollment(self, person_name: str, collected_poses_data: Dict) -> Dict[str, Any]:
        """Save pending enrollment data that was collected but not yet saved"""
        try:
            # Convert PoseState enum keys to string keys if needed
            converted_poses = self._convert_pose_keys_to_strings(collected_poses_data)
            save_result = self.save_enrollment_data(person_name, converted_poses)
            # Reload the enrollment database to ensure fresh data
            self.enrollment_database = self._load_enrollment_database()
            return save_result
        except Exception as e:
            return {
                'success': False,
                'person_name': person_name,
                'error': str(e)
            }
    
    def _convert_pose_keys_to_strings(self, collected_poses: Dict) -> Dict[str, Any]:
        """Convert PoseState enum keys to string keys"""
        converted = {}
        for pose_key, pose_data in collected_poses.items():
            # Handle both PoseState enum and string keys
            if hasattr(pose_key, 'value'):
                string_key = pose_key.value
            else:
                string_key = str(pose_key)
            converted[string_key] = pose_data
        return converted

    def _select_best_candidate(self, candidates: List[Dict], frame_shape: tuple) -> Dict:
        """Select best candidate prioritizing center position, larger faces, and filtering background faces"""
        if not candidates:
            return {}
        
        if len(candidates) == 1:
            candidate = candidates[0]
            # Still apply basic filtering even for single candidate
            if self._is_likely_background_face(candidate, frame_shape):
                return {}  # Reject background face
            return candidate
        
        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        # Filter out likely background faces first
        filtered_candidates = []
        for candidate in candidates:
            if not self._is_likely_background_face(candidate, frame_shape):
                filtered_candidates.append(candidate)
        
        # If all faces filtered out, return empty (no good candidate)
        if not filtered_candidates:
            return {}
        
        # If only one good candidate after filtering, return it
        if len(filtered_candidates) == 1:
            return filtered_candidates[0]
        
        def candidate_score(candidate):
            quality_score = candidate.get('quality_score', 0)
            # Try both possible bbox locations  
            bbox = (candidate.get('detection', {}).get('bbox') or 
                    candidate.get('bbox') or 
                    [0, 0, 50, 50])
            
            # Extract bbox coordinates
            x, y, w, h = bbox
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Calculate distance from center (normalized)
            center_distance = ((face_center_x - frame_center_x) ** 2 + 
                             (face_center_y - frame_center_y) ** 2) ** 0.5
            max_distance = (frame_width ** 2 + frame_height ** 2) ** 0.5
            center_score = 1.0 - (center_distance / max_distance)  # Higher score for center faces
            
            # Calculate face size score (normalized) - bigger bonus for enrollment
            face_area = w * h
            frame_area = frame_width * frame_height
            size_ratio = face_area / frame_area
            size_score = min(size_ratio * 2, 1.0)  # Double bonus for larger faces, cap at 1.0
            
            # Bonus for faces in the center third of the frame
            center_third_x = frame_width / 3
            center_third_y = frame_height / 3
            in_center_third = (center_third_x <= face_center_x <= 2 * center_third_x and
                              center_third_y <= face_center_y <= 2 * center_third_y)
            center_third_bonus = 0.2 if in_center_third else 0
            
            # Combined score: 40% quality, 35% center position, 20% size, 5% center third bonus
            combined_score = (0.40 * quality_score + 
                            0.35 * center_score + 
                            0.20 * size_score + 
                            center_third_bonus)
            
            return combined_score
        
        # Select candidate with highest combined score
        best_candidate = max(filtered_candidates, key=candidate_score)
        return best_candidate
    
    def _is_likely_background_face(self, candidate: Dict, frame_shape: tuple) -> bool:
        """Determine if a face is likely in the background and should be filtered out"""
        # Try both possible bbox locations
        bbox = (candidate.get('detection', {}).get('bbox') or 
                candidate.get('bbox') or 
                [0, 0, 50, 50])
        quality_score = candidate.get('quality_score', 0)
        
        x, y, w, h = bbox
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate face size as percentage of frame
        face_area = w * h
        frame_area = frame_width * frame_height
        size_ratio = face_area / frame_area
        
        # Calculate position relative to frame edges
        center_x = x + w // 2
        center_y = y + h // 2
        edge_distance_x = min(center_x, frame_width - center_x) / frame_width
        edge_distance_y = min(center_y, frame_height - center_y) / frame_height
        min_edge_distance = min(edge_distance_x, edge_distance_y)
        
        # Background face indicators:
        # 1. Very small faces (< 2% of frame area)
        # 2. Very low quality faces (< 0.2 quality)
        # 3. Faces very close to edges with small size
        # 4. Faces in extreme corners
        
        is_too_small = size_ratio < 0.02  # Less than 2% of frame
        is_very_low_quality = quality_score < 0.2
        is_edge_small_face = min_edge_distance < 0.1 and size_ratio < 0.05  # Close to edge and small
        
        # Check if face is in corner (within 15% of edge in both dimensions)
        corner_threshold = 0.15
        is_in_corner = ((center_x < corner_threshold * frame_width or center_x > (1 - corner_threshold) * frame_width) and
                       (center_y < corner_threshold * frame_height or center_y > (1 - corner_threshold) * frame_height))
        
        is_background = is_too_small or is_very_low_quality or is_edge_small_face or is_in_corner
        
        # Debug logging for background detection
        if is_background:
            reasons = []
            if is_too_small: reasons.append(f"too_small({size_ratio:.3f})")
            if is_very_low_quality: reasons.append(f"low_quality({quality_score:.3f})")
            if is_edge_small_face: reasons.append(f"edge_small({min_edge_distance:.3f},{size_ratio:.3f})")
            if is_in_corner: reasons.append("in_corner")
            print(f"🚫 Background face filtered: {', '.join(reasons)}")
        
        return is_background

    def get_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Get face embedding for a single image using DeepFace.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Face embedding as numpy array or None if failed
        """
        if not DEEPFACE_AVAILABLE:
            print(f"⚠️ DeepFace not available, using face engine embedding")
            return self._get_embedding_from_face_engine(image_path)
        
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=self.recognition_model,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"❌ Error extracting DeepFace embedding from {image_path}: {e}")
            # Fallback to face engine
            return self._get_embedding_from_face_engine(image_path)
    
    def _get_embedding_from_face_engine(self, image_path: str) -> Optional[np.ndarray]:
        """Fallback method to get embedding using the existing face engine"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # The image should already be aligned face from enrollment, so just generate embedding
            embedding = self.face_engine.generate_embedding(image)
            return embedding
        except Exception as e:
            print(f"❌ Error extracting face engine embedding from {image_path}: {e}")
            return None

    def get_person_embeddings(self, person_name: str) -> Dict[str, np.ndarray]:
        """Get all embeddings for a person"""
        person_data = self.enrollment_database.get('persons', {}).get(person_name)
        if not person_data or 'embeddings' not in person_data:
            return {}
        
        embeddings = {}
        for pose_name, embedding_list in person_data['embeddings'].items():
            embeddings[pose_name] = np.array(embedding_list)
        
        return embeddings
    
    def get_all_person_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get embeddings for all enrolled persons"""
        all_embeddings = {}
        for person_name in self.enrollment_database.get('persons', {}):
            embeddings = self.get_person_embeddings(person_name)
            if embeddings:
                all_embeddings[person_name] = embeddings
        return all_embeddings

    def generate_missing_embeddings(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Generate embeddings for all poses that don't have embeddings yet"""
        try:
            persons_data = self.enrollment_database.get('persons', {})
            total_persons = len(persons_data)
            total_poses_processed = 0
            embeddings_generated = 0
            errors = []
            
            if progress_callback:
                progress_callback(0, total_persons, "Starting embedding generation...")
            
            for person_idx, (person_name, person_data) in enumerate(persons_data.items()):
                if progress_callback:
                    progress_callback(person_idx, total_persons, f"Processing {person_name}...")
                
                poses = person_data.get('poses', {})
                existing_embeddings = person_data.get('embeddings', {})
                person_folder = self.base_path / person_name
                
                # Check which poses need embeddings
                poses_needing_embeddings = []
                for pose_name, pose_data in poses.items():
                    if pose_name not in existing_embeddings:
                        poses_needing_embeddings.append((pose_name, pose_data))
                
                if not poses_needing_embeddings:
                    print(f"✅ {person_name}: All poses already have embeddings")
                    continue
                
                print(f"🧠 {person_name}: Generating {len(poses_needing_embeddings)} missing embeddings...")
                
                # Generate missing embeddings
                new_embeddings = {}
                for pose_name, pose_data in poses_needing_embeddings:
                    total_poses_processed += 1
                    filename = pose_data.get('filename')
                    if not filename:
                        errors.append(f"{person_name}/{pose_name}: No filename found")
                        continue
                    
                    image_path = person_folder / filename
                    if not image_path.exists():
                        errors.append(f"{person_name}/{pose_name}: Image file not found: {filename}")
                        continue
                    
                    # Generate embedding
                    embedding = self.get_face_embedding(str(image_path))
                    if embedding is not None:
                        new_embeddings[pose_name] = embedding.tolist()
                        embeddings_generated += 1
                        print(f"✅ Generated embedding for {person_name}/{pose_name}")
                        
                        # Update pose data to indicate it has embedding
                        poses[pose_name]['has_embedding'] = True
                    else:
                        errors.append(f"{person_name}/{pose_name}: Failed to generate embedding")
                        poses[pose_name]['has_embedding'] = False
                
                # Update database if we generated any embeddings for this person
                if new_embeddings:
                    # Merge with existing embeddings
                    all_embeddings = existing_embeddings.copy()
                    all_embeddings.update(new_embeddings)
                    
                    # Update person data
                    persons_data[person_name]['embeddings'] = all_embeddings
                    persons_data[person_name]['poses'] = poses
                    persons_data[person_name]['embedding_model'] = self.recognition_model if DEEPFACE_AVAILABLE else 'face_engine'
                    persons_data[person_name]['detector_backend'] = self.detector_backend if DEEPFACE_AVAILABLE else 'yunet'
                    
                    print(f"💾 Updated {person_name}: {len(new_embeddings)} new embeddings")
            
            # Save updated database
            if embeddings_generated > 0:
                self._save_enrollment_database()
                print(f"✅ Database updated with {embeddings_generated} new embeddings")
            
            if progress_callback:
                progress_callback(total_persons, total_persons, "Completed!")
            
            return {
                'success': True,
                'total_persons': total_persons,
                'total_poses_processed': total_poses_processed,
                'embeddings_generated': embeddings_generated,
                'errors': errors,
                'embedding_model': self.recognition_model if DEEPFACE_AVAILABLE else 'face_engine'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'embeddings_generated': embeddings_generated,
                'errors': errors
            }

    def count_missing_embeddings(self) -> Dict[str, Any]:
        """Count how many poses don't have embeddings"""
        persons_data = self.enrollment_database.get('persons', {})
        total_poses = 0
        poses_with_embeddings = 0
        missing_embeddings = 0
        persons_needing_embeddings = []
        
        for person_name, person_data in persons_data.items():
            poses = person_data.get('poses', {})
            existing_embeddings = person_data.get('embeddings', {})
            
            person_total_poses = len(poses)
            person_embeddings = len(existing_embeddings)
            person_missing = person_total_poses - person_embeddings
            
            total_poses += person_total_poses
            poses_with_embeddings += person_embeddings
            
            if person_missing > 0:
                missing_embeddings += person_missing
                persons_needing_embeddings.append({
                    'name': person_name,
                    'total_poses': person_total_poses,
                    'existing_embeddings': person_embeddings,
                    'missing_embeddings': person_missing
                })
        
        return {
            'total_persons': len(persons_data),
            'total_poses': total_poses,
            'poses_with_embeddings': poses_with_embeddings,
            'missing_embeddings': missing_embeddings,
            'persons_needing_embeddings': persons_needing_embeddings,
            'all_embeddings_complete': missing_embeddings == 0
        }

    def get_enrollment_status(self, person_name: str = None) -> Dict[str, Any]:
        """Get enrollment status with embedding information"""
        if person_name:
            person_data = self.enrollment_database.get('persons', {}).get(person_name)
            if person_data:
                embeddings = person_data.get('embeddings', {})
                return {
                    'person_name': person_name,
                    'enrolled': True,
                    'poses': person_data.get('poses', {}),
                    'enrolled_date': person_data.get('enrolled_date'),
                    'total_poses': person_data.get('total_poses', 0),
                    'embeddings_count': len(embeddings),
                    'embedding_model': person_data.get('embedding_model', 'unknown'),
                    'detector_backend': person_data.get('detector_backend', 'unknown'),
                    'poses_with_embeddings': list(embeddings.keys())
                }
            else:
                return {
                    'person_name': person_name,
                    'enrolled': False
                }
        else:
            persons_data = self.enrollment_database.get('persons', {})
            total_embeddings = sum(len(p.get('embeddings', {})) for p in persons_data.values())
            return {
                'total_persons': len(persons_data),
                'persons': list(persons_data.keys()),
                'total_embeddings': total_embeddings,
                'last_updated': self.enrollment_database.get('last_updated')
            }

    def delete_enrollment(self, person_name: str) -> Dict[str, Any]:
        """Delete enrollment data for a person"""
        try:
            # Remove from database
            if person_name in self.enrollment_database.get('persons', {}):
                del self.enrollment_database['persons'][person_name]
                self._save_enrollment_database()
            
            # Remove folder
            person_folder = self.base_path / person_name
            if person_folder.exists():
                import shutil
                shutil.rmtree(person_folder)
            
            return {
                'success': True,
                'person_name': person_name,
                'message': f'Enrollment data for {person_name} deleted successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'person_name': person_name,
                'error': str(e)
            }

    def start_video_enrollment(self, session_id: str, video_path: str) -> Dict[str, Any]:
        """Initialize video enrollment session"""
        try:
            # Store session info (in a real implementation, use proper session management)
            session_info = {
                'session_id': session_id,
                'type': 'video',
                'video_path': video_path,
                'started_at': datetime.now().isoformat()
            }
            
            # For now, return success to indicate session is ready
            # In a full implementation, you'd start the video processing here
            print(f"✅ Video enrollment session initialized: {session_id}")
            print(f"📹 Video path: {video_path}")
            
            return {
                'success': True,
                'session_id': session_id,
                'message': f'Video enrollment initialized with {video_path}',
                'video_path': video_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }

    def start_camera_enrollment(self, session_id: str, camera_index: int = 0, rtsp_url: str = None) -> Dict[str, Any]:
        """Initialize camera enrollment session"""
        try:
            # Store session info
            session_info = {
                'session_id': session_id,
                'type': 'camera',
                'camera_index': camera_index,
                'rtsp_url': rtsp_url,
                'started_at': datetime.now().isoformat()
            }
            
            # For now, return success to indicate session is ready
            print(f"✅ Camera enrollment session initialized: {session_id}")
            if rtsp_url:
                print(f"📷 RTSP URL: {rtsp_url}")
            else:
                print(f"📷 Camera index: {camera_index}")
            
            return {
                'success': True,
                'session_id': session_id,
                'message': f'Camera enrollment initialized',
                'camera_index': camera_index,
                'rtsp_url': rtsp_url
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }