@echo off
REM ====================================================================
REM Clear Fake Data Script for Attendance System Backend
REM ====================================================================
REM This script removes all fake/test data from the system including:
REM - SQLite database files
REM - Face images and enrollments
REM - Upload directories
REM - JSON database files
REM - Log files and temporary data
REM ====================================================================

echo.
echo ====================================================================
echo               ATTENDANCE SYSTEM - CLEAR FAKE DATA
echo ====================================================================
echo.
echo WARNING: This will permanently delete all test/fake data!
echo.
set /p confirm="Are you sure you want to continue? (Y/N): "

if /i "%confirm%" neq "Y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo Starting data cleanup...
echo.

REM Remove SQLite database files
echo [1/8] Removing SQLite database files...
if exist "attendance_system.db" (
    del /q "attendance_system.db"
    echo   - Deleted attendance_system.db
)
if exist "people_database.json" (
    del /q "people_database.json"
    echo   - Deleted people_database.json
)
if exist "*.db" (
    del /q "*.db"
    echo   - Deleted additional .db files
)
if exist "*.sqlite" (
    del /q "*.sqlite"
    echo   - Deleted .sqlite files
)

REM Remove face images directory
echo [2/8] Removing face images...
if exist "face_images" (
    rmdir /s /q "face_images"
    echo   - Deleted face_images directory
)

REM Remove enrollments directory
echo [3/8] Removing enrollment data...
if exist "enrollments" (
    rmdir /s /q "enrollments"
    echo   - Deleted enrollments directory
)

REM Remove uploads directory
echo [4/8] Removing uploaded files...
if exist "uploads" (
    rmdir /s /q "uploads"
    echo   - Deleted uploads directory
)

REM Remove JSON database files
echo [5/8] Removing JSON database files...
if exist "enrollment_database.json" (
    del /q "enrollment_database.json"
    echo   - Deleted enrollment_database.json
)
if exist "people_database.json" (
    del /q "people_database.json"
    echo   - Deleted people_database.json
)
if exist "*.json" (
    for %%f in (*.json) do (
        if not "%%f"=="package.json" if not "%%f"=="package-lock.json" if not "%%f"=="requirements.json" (
            del /q "%%f"
            echo   - Deleted %%f
        )
    )
)

REM Remove log files
echo [6/8] Removing log files...
if exist "*.log" (
    del /q "*.log"
    echo   - Deleted log files
)

REM Remove temporary files and cache
echo [7/8] Removing temporary files...
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo   - Deleted __pycache__ directory
)
if exist "app\__pycache__" (
    rmdir /s /q "app\__pycache__"
    echo   - Deleted app\__pycache__ directory
)
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
if exist "*.pyc" (
    del /s /q "*.pyc"
    echo   - Deleted .pyc files
)
if exist "*.tmp" (
    del /q "*.tmp"
    echo   - Deleted temporary files
)

REM Recreate necessary empty directories
echo [8/8] Recreating empty directories...
mkdir "face_images" 2>nul
mkdir "enrollments" 2>nul
mkdir "uploads" 2>nul
echo   - Recreated empty directories

echo.
echo ====================================================================
echo                        CLEANUP COMPLETED
echo ====================================================================
echo.
echo The following data has been cleared:
echo   ✓ Database files (SQLite)
echo   ✓ Face images and pose data
echo   ✓ Enrollment records
echo   ✓ Uploaded files
echo   ✓ JSON databases
echo   ✓ Log files
echo   ✓ Temporary files and cache
echo   ✓ Python cache files
echo.
echo Empty directories have been recreated for fresh start.
echo.
echo System is now ready for new data!
echo ====================================================================
echo.
pause