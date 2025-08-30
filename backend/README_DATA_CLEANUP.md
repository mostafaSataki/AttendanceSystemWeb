# Data Cleanup Scripts

This directory contains scripts to clear fake/test data from the Attendance System backend.

## Available Scripts

### 1. Windows Batch File
```bash
clear_fake_data.bat
```
- Double-click to run on Windows
- Interactive confirmation prompt
- Clears all test data and recreates empty directories

### 2. Python Script (Cross-platform)
```bash
python clear_fake_data.py
```
- Works on Windows, Linux, and macOS
- Interactive confirmation prompt
- Same functionality as batch file

## What Gets Cleaned

The scripts remove the following data:

### Database Files
- `*.db` - SQLite database files
- `*.sqlite` - SQLite database files
- `enrollment_database.json` - JSON-based enrollment database

### Image and Upload Directories
- `face_images/` - All stored face images
- `enrollments/` - All enrollment data
- `uploads/` - All uploaded files

### Temporary Files
- `*.log` - Log files
- `*.tmp` - Temporary files
- `__pycache__/` - Python cache directories
- `*.pyc` - Python compiled files

## What Gets Preserved

The scripts preserve:
- Model files in `models/` directory
- Source code files (`.py`, `.js`, etc.)
- Configuration files
- `package.json` and `package-lock.json`

## What Gets Recreated

After cleanup, the scripts recreate empty directories:
- `face_images/` (empty)
- `enrollments/` (empty)
- `uploads/` (empty)

## Usage Instructions

### Windows
1. Open the `backend` directory
2. Double-click `clear_fake_data.bat`
3. Confirm when prompted with 'Y'
4. Wait for completion

### Any Platform
1. Open terminal in `backend` directory
2. Run: `python clear_fake_data.py`
3. Confirm when prompted with 'Y'
4. Wait for completion

## Safety Features

- **Interactive confirmation** - Both scripts ask for confirmation before deletion
- **Selective deletion** - Only removes data files, preserves source code
- **Error handling** - Continues even if some files can't be deleted
- **Directory recreation** - Ensures required directories exist after cleanup

## When to Use

Use these scripts when you want to:
- Clear test data after development
- Reset the system to a clean state
- Remove fake enrollment data
- Start fresh with real data
- Clean up after testing sessions

## Warning

⚠️ **These scripts permanently delete data!** Make sure you have backups of any important data before running the cleanup scripts.