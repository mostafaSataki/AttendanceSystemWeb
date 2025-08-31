@echo off
echo Killing process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo Port 8000 has been cleared.
pause