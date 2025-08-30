@echo off
echo Killing process on port 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo Port 3000 has been cleared.
pause