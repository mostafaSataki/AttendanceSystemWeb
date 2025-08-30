# 📹 RTSP Streaming with Analytics Integration

A complete RTSP streaming solution with single-ingest, multi-consumer architecture for face recognition analytics and on-demand client playback.

## 🎯 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RTSP Camera   │────│  FFmpeg Relay    │────│  Analytics      │
│   (H.264)       │    │  (Single Ingest) │    │  Service        │
└─────────────────┘    └──────────────────┘    │  (Face Recognition) │
                                │               └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │   Shared Memory │
                       │   Buffer Pool   │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │  On-Demand HLS  │
                       │  Generator      │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │  FastAPI + JWT  │
                       │  Token Auth     │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │   Next.js UI    │
                       │   HLS Player    │
                       └─────────────────┘
```

## ✨ Key Features

### 🔥 Single Ingest, Multi Consumer
- **One RTSP connection per camera** - eliminates duplicate network load
- **Shared frame buffer** - analytics and playback consume from same source
- **Thread-safe frame distribution** - concurrent access without conflicts

### 🎬 On-Demand HLS Playback  
- **Low-Latency HLS with fMP4** - 2-second segments for minimal delay
- **JWT-based security** - signed tokens with expiration
- **Automatic cleanup** - sessions terminate when unused
- **Browser-native support** - works in Safari + hls.js for others

### 🤖 Real-Time Analytics
- **Frame skipping** - processes every 3rd frame for performance  
- **Face recognition integration** - detects and identifies faces
- **Performance statistics** - tracks processing times and accuracy
- **Non-blocking processing** - doesn't impact video stream quality

### 🔒 Security & Access Control
- **JWT tokens** - short-lived session authentication
- **Session management** - automatic expiry and cleanup
- **Protected endpoints** - all HLS content requires valid tokens
- **CORS enabled** - supports cross-origin requests

## 🛠️ Installation & Setup

### Backend Dependencies
```bash
cd backend
pip install PyJWT==2.8.0
# Other dependencies already in requirements.txt
```

### Frontend Dependencies  
```bash
cd AttendanceSystemWeb
npm install hls.js
# Already added to package.json
```

### System Requirements
- **FFmpeg** - must be installed and available in PATH
- **OpenCV** - for video processing and analytics
- **Python 3.8+** - for backend services
- **Node.js 18+** - for frontend application

## 🚀 API Usage

### 1. Add RTSP Camera
```bash
curl -X POST http://localhost:8000/api/stream/cameras/add \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "camera_001",
    "rtsp_url": "rtsp://username:password@192.168.1.100:554/stream",  
    "name": "Main Entrance",
    "enable_analytics": true
  }'
```

### 2. Create Playback Session
```bash
curl -X POST http://localhost:8000/api/stream/playback/create-session \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "camera_001",
    "user_id": "web_user"
  }'
```

Response:
```json
{
  "success": true,
  "session_id": "abc123",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "hls_url": "/api/stream/hls/abc123/playlist.m3u8",
  "expires_at": "2024-01-01T15:30:00Z"
}
```

### 3. Start HLS Generation
```bash
curl -X POST http://localhost:8000/api/stream/playback/abc123/start
```

### 4. Access HLS Stream
```
http://localhost:8000/api/stream/hls/abc123/playlist.m3u8
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## 🎮 Frontend Integration

### React HLS Player Component
```tsx
import HLSPlayer from '@/components/HLSPlayer';

<HLSPlayer
  cameraId="camera_001"
  cameraName="Main Entrance"
  onError={(error) => console.error(error)}
  onSessionEnd={() => console.log('Session ended')}
/>
```

### Camera Management
```tsx
import CameraManager from '@/components/CameraManager';

// Complete camera management interface
<CameraManager />
```

## 📊 Analytics Integration

### Real-Time Face Detection
- Processes frames at 10 FPS (every 3rd frame from 30 FPS stream)
- Detects faces using OpenCV/YOLO
- Tracks face recognition statistics
- Non-blocking analytics pipeline

### Performance Monitoring
```bash
# Get analytics statistics
curl http://localhost:8000/api/stream/analytics/stats?camera_id=camera_001
```

Response:
```json
{
  "success": true,
  "camera_id": "camera_001", 
  "stats": {
    "frames_processed": 1250,
    "faces_detected": 45,
    "faces_recognized": 32,
    "uptime_seconds": 3600,
    "avg_processing_time": 0.085
  }
}
```

## 🔧 Configuration Options

### RTSP Manager Settings
```python
# In rtsp_manager.py
self.session_duration = timedelta(minutes=30)  # JWT expiry
self.jwt_secret = secrets.token_urlsafe(32)    # Auth secret
```

### HLS Settings  
```python
# FFmpeg HLS generation parameters
'-hls_time', '2',              # 2-second segments
'-hls_list_size', '5',         # Keep 5 segments  
'-hls_segment_type', 'fmp4',   # fMP4 for low latency
```

### Analytics Settings
```python
# In analytics_integration.py  
self.process_every_n_frames = 3  # Skip frames for performance
```

## 🛡️ Security Considerations

### JWT Token Security
- **Short expiration** - 30-minute sessions by default
- **Signed tokens** - tamper-proof with secret key
- **Session validation** - server-side session tracking

### RTSP URL Protection
- **Credentials in URLs** - store securely, don't log
- **Network security** - use VPN/private networks when possible
- **Camera firmware** - keep updated for security patches

## 📈 Performance & Scalability

### Single Camera Performance
- **RTSP ingest**: ~5 MB/s for 1080p H.264
- **Analytics processing**: ~85ms per frame
- **HLS generation**: ~10 MB/s during active streaming
- **Memory usage**: ~200MB per active camera

### Multi-Camera Scaling
- **Linear CPU scaling** - each camera adds ~1 CPU core
- **Network bandwidth** - RTSP ingest + HLS output per camera
- **Storage requirements** - temporary HLS segments (~100MB per session)

### Recommended Limits
- **Development**: 1-2 cameras
- **Production**: 4-8 cameras per server (depends on hardware)
- **Enterprise**: Load balancer with multiple servers

## 🔍 Troubleshooting

### Common Issues

#### FFmpeg Not Found
```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

#### RTSP Connection Failed
```python
# Check RTSP URL format
rtsp://username:password@ip:port/path

# Test with FFmpeg directly
ffmpeg -i "rtsp://admin:password@192.168.1.100:554/stream" -f null -
```

#### HLS Playback Issues
- **Check JWT token** - must be valid and not expired
- **Verify CORS** - frontend domain must be allowed
- **Browser support** - Safari native, others need hls.js

#### Analytics Performance
- **Reduce frame rate** - increase `process_every_n_frames`
- **Lower resolution** - downscale frames before processing
- **GPU acceleration** - use CUDA if available

### Debug Commands

#### Check Camera Status
```bash
curl http://localhost:8000/api/stream/cameras/status
```

#### Monitor Active Sessions  
```bash
curl http://localhost:8000/api/stream/playback/sessions
```

#### Health Check
```bash
curl http://localhost:8000/api/stream/health
```

## 🎯 Next Steps & Improvements

### Planned Features
- [ ] **WebRTC support** - for even lower latency
- [ ] **Multi-bitrate HLS** - adaptive streaming quality
- [ ] **Recording capability** - save streams to disk
- [ ] **Motion detection** - trigger recording on movement
- [ ] **Push notifications** - alert on face recognition events

### Performance Optimizations
- [ ] **GPU decoding** - hardware-accelerated video processing
- [ ] **NVIDIA codec support** - NVENC/NVDEC integration
- [ ] **Edge computing** - run analytics closer to cameras
- [ ] **CDN integration** - distribute HLS globally

---

## 📞 Support & Contributing

For issues and feature requests, please create an issue in the repository.

**Built with ❤️ for real-time video analytics**