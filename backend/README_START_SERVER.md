# How to Start the Backend Server

## Quick Start

### Option 1: Windows Batch File (Recommended)
```bash
# Double-click or run from command prompt
run_server.bat
```

### Option 2: Python Script
```bash
# From backend directory
python run_server.py
```

### Option 3: Direct uvicorn command
```bash
# From backend directory
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Prerequisites

1. **Python 3.8+** must be installed
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Server Information

- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Port**: 8000 (default)
- **Auto-reload**: Enabled for development

## Endpoints Available

- `GET /api/people` - Get all people
- `POST /api/people` - Create new person
- `PUT /api/people/{id}` - Update person
- `DELETE /api/people/{id}` - Delete person
- `POST /api/people/{id}/toggle-status` - Toggle person status
- `POST /api/enrollment/start` - Start face enrollment
- `POST /api/enrollment/stop` - Stop face enrollment
- `POST /api/recognition/start` - Start face recognition
- `POST /api/recognition/stop` - Stop face recognition

## Troubleshooting

### Error: "Failed to fetch"
- Make sure the backend server is running on port 8000
- Check if there are any error messages in the server console
- Verify the frontend is configured to connect to `http://localhost:8000`

### Error: "Module not found"
```bash
# Install missing packages
pip install -r requirements.txt
```

### Error: "Port already in use"
```bash
# Kill process using port 8000
taskkill /f /im python.exe
# Or use a different port
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Next Steps

1. **Start the backend server** using one of the methods above
2. **Start the frontend** (in another terminal):
   ```bash
   cd D:\source\AttendanceSystemWeb
   npm run dev
   ```
3. **Open your browser** to http://localhost:3000

The frontend will now connect to the backend API and you can add/manage people without errors!