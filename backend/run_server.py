#!/usr/bin/env python3
"""
Start FastAPI Backend Server for Attendance System
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting FastAPI Backend Server for Attendance System...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    try:
        # Start uvicorn server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install fastapi uvicorn python-multipart")

if __name__ == "__main__":
    main()