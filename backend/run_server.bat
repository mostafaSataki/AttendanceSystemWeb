@echo off
echo Starting FastAPI Backend Server for Attendance System...
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if uvicorn is installed
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Error: uvicorn is not installed
    echo Installing uvicorn...
    pip install uvicorn
)

REM Check if fastapi is installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Error: fastapi is not installed
    echo Installing required packages...
    pip install fastapi uvicorn python-multipart
)

REM Start the server
echo Starting server on http://localhost:8000
echo API documentation will be available at http://localhost:8000/docs
echo.

cd /d "%~dp0"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause