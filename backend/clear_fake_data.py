#!/usr/bin/env python3
"""
Clear Fake Data Script for Attendance System Backend

This script removes all fake/test data from the system including:
- SQLite database files
- Face images and enrollments
- Upload directories  
- JSON database files
- Log files and temporary data
"""

import os
import shutil
import glob
from pathlib import Path


def print_header():
    """Print script header"""
    print("\n" + "="*70)
    print("               ATTENDANCE SYSTEM - CLEAR FAKE DATA")
    print("="*70)


def confirm_deletion():
    """Ask user for confirmation"""
    print("\nWARNING: This will permanently delete all test/fake data!")
    response = input("\nAre you sure you want to continue? (Y/N): ").strip().upper()
    return response == 'Y'


def safe_remove_file(file_path):
    """Safely remove a file if it exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   - Deleted {file_path}")
            return True
    except Exception as e:
        print(f"   - Failed to delete {file_path}: {e}")
    return False


def safe_remove_dir(dir_path):
    """Safely remove a directory if it exists"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   - Deleted {dir_path} directory")
            return True
    except Exception as e:
        print(f"   - Failed to delete {dir_path}: {e}")
    return False


def safe_create_dir(dir_path):
    """Safely create a directory"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"   - Failed to create {dir_path}: {e}")
    return False


def main():
    """Main cleanup function"""
    print_header()
    
    if not confirm_deletion():
        print("Operation cancelled.")
        return
    
    print("\nStarting data cleanup...\n")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Step 1: Remove SQLite database files
    print("[1/8] Removing SQLite database files...")
    safe_remove_file("attendance_system.db")
    safe_remove_file("people_database.json")  # Legacy JSON file
    db_files = glob.glob("*.db") + glob.glob("*.sqlite")
    for db_file in db_files:
        safe_remove_file(db_file)
    
    # Step 2: Remove face images directory
    print("[2/8] Removing face images...")
    safe_remove_dir("face_images")
    
    # Step 3: Remove enrollments directory  
    print("[3/8] Removing enrollment data...")
    safe_remove_dir("enrollments")
    
    # Step 4: Remove uploads directory
    print("[4/8] Removing uploaded files...")
    safe_remove_dir("uploads")
    
    # Step 5: Remove JSON database files (except package files)
    print("[5/8] Removing JSON database files...")
    # Specific database files
    safe_remove_file("enrollment_database.json")
    safe_remove_file("people_database.json")
    
    # Other JSON files (except package files)
    json_files = glob.glob("*.json")
    excluded_files = {"package.json", "package-lock.json", "requirements.json"}
    for json_file in json_files:
        if os.path.basename(json_file) not in excluded_files:
            safe_remove_file(json_file)
    
    # Step 6: Remove log files
    print("[6/8] Removing log files...")
    log_files = glob.glob("*.log")
    for log_file in log_files:
        safe_remove_file(log_file)
    
    # Step 7: Remove temporary files and cache
    print("[7/8] Removing temporary files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs[:]:  # Create a copy of the list to modify during iteration
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                safe_remove_dir(cache_path)
                dirs.remove(dir_name)  # Don't recurse into deleted directory
    
    # Remove .pyc files
    pyc_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc"):
                pyc_files.append(os.path.join(root, file))
    
    for pyc_file in pyc_files:
        safe_remove_file(pyc_file)
    
    # Remove .tmp files
    tmp_files = glob.glob("*.tmp")
    for tmp_file in tmp_files:
        safe_remove_file(tmp_file)
    
    # Step 8: Recreate necessary empty directories
    print("[8/8] Recreating empty directories...")
    directories_to_create = ["face_images", "enrollments", "uploads"]
    for directory in directories_to_create:
        safe_create_dir(directory)
    print("   - Recreated empty directories")
    
    # Print completion message
    print("\n" + "="*70)
    print("                        CLEANUP COMPLETED")
    print("="*70)
    print("\nThe following data has been cleared:")
    print("   ✓ Database files (SQLite)")
    print("   ✓ Face images and pose data")
    print("   ✓ Enrollment records")
    print("   ✓ Uploaded files")
    print("   ✓ JSON databases")
    print("   ✓ Log files")
    print("   ✓ Temporary files and cache")
    print("   ✓ Python cache files")
    print("\nEmpty directories have been recreated for fresh start.")
    print("\nSystem is now ready for new data!")
    print("="*70)


if __name__ == "__main__":
    main()