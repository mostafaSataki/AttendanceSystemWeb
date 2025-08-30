#!/bin/bash

# ====================================================================
# Clear Fake Data Script for Attendance System Backend
# ====================================================================
# This script removes all fake/test data from the system including:
# - SQLite database files
# - Face images and enrollments
# - Upload directories
# - JSON database files
# - Log files and temporary data
# ====================================================================

echo ""
echo "===================================================================="
echo "               ATTENDANCE SYSTEM - CLEAR FAKE DATA"
echo "===================================================================="
echo ""
echo "WARNING: This will permanently delete all test/fake data!"
echo ""
read -p "Are you sure you want to continue? (Y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Starting data cleanup..."
echo ""

# Remove SQLite database files
echo "[1/8] Removing SQLite database files..."
if [ -f "attendance_system.db" ]; then
    rm -f "attendance_system.db"
    echo "   - Deleted attendance_system.db"
fi
if [ -f "people_database.json" ]; then
    rm -f "people_database.json"
    echo "   - Deleted people_database.json"
fi
if ls *.db 1> /dev/null 2>&1; then
    rm -f *.db
    echo "   - Deleted additional .db files"
fi
if ls *.sqlite 1> /dev/null 2>&1; then
    rm -f *.sqlite
    echo "   - Deleted .sqlite files"
fi

# Remove face images directory
echo "[2/8] Removing face images..."
if [ -d "face_images" ]; then
    rm -rf "face_images"
    echo "   - Deleted face_images directory"
fi

# Remove enrollments directory
echo "[3/8] Removing enrollment data..."
if [ -d "enrollments" ]; then
    rm -rf "enrollments"
    echo "   - Deleted enrollments directory"
fi

# Remove uploads directory
echo "[4/8] Removing uploaded files..."
if [ -d "uploads" ]; then
    rm -rf "uploads"
    echo "   - Deleted uploads directory"
fi

# Remove JSON database files
echo "[5/8] Removing JSON database files..."
if [ -f "enrollment_database.json" ]; then
    rm -f "enrollment_database.json"
    echo "   - Deleted enrollment_database.json"
fi
if [ -f "people_database.json" ]; then
    rm -f "people_database.json"
    echo "   - Deleted people_database.json"
fi

# Remove other JSON files (except package files)
for json_file in *.json; do
    if [ -f "$json_file" ]; then
        case "$json_file" in
            "package.json"|"package-lock.json"|"requirements.json")
                # Skip these files
                ;;
            *)
                rm -f "$json_file"
                echo "   - Deleted $json_file"
                ;;
        esac
    fi
done

# Remove log files
echo "[6/8] Removing log files..."
if ls *.log 1> /dev/null 2>&1; then
    rm -f *.log
    echo "   - Deleted log files"
fi

# Remove temporary files and cache
echo "[7/8] Removing temporary files..."
if [ -d "__pycache__" ]; then
    rm -rf "__pycache__"
    echo "   - Deleted __pycache__ directory"
fi
if [ -d "app/__pycache__" ]; then
    rm -rf "app/__pycache__"
    echo "   - Deleted app/__pycache__ directory"
fi

# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
if find . -name "*.pyc" -delete 2>/dev/null; then
    echo "   - Deleted .pyc files"
fi

# Remove .tmp files
if ls *.tmp 1> /dev/null 2>&1; then
    rm -f *.tmp
    echo "   - Deleted temporary files"
fi

# Recreate necessary empty directories
echo "[8/8] Recreating empty directories..."
mkdir -p "face_images" "enrollments" "uploads"
echo "   - Recreated empty directories"

echo ""
echo "===================================================================="
echo "                        CLEANUP COMPLETED"
echo "===================================================================="
echo ""
echo "The following data has been cleared:"
echo "   ✓ Database files (SQLite)"
echo "   ✓ Face images and pose data"
echo "   ✓ Enrollment records"
echo "   ✓ Uploaded files"
echo "   ✓ JSON databases"
echo "   ✓ Log files"
echo "   ✓ Temporary files and cache"
echo "   ✓ Python cache files"
echo ""
echo "Empty directories have been recreated for fresh start."
echo ""
echo "System is now ready for new data!"
echo "===================================================================="
echo ""