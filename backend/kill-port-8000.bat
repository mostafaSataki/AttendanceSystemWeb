@echo off
echo Killing process on port 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo Port 8001 has been cleared.
pause