#!/usr/bin/env python3
"""
Test database functionality
"""

import json
from pathlib import Path

# Test the same functions as in people.py
DB_FILE = Path("people_database.json")

def load_people_db():
    """Load people database from JSON file"""
    if DB_FILE.exists():
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading people database: {e}")
    return []

def save_people_db(people_data):
    """Save people database to JSON file"""
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(people_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Saved database to {DB_FILE}")
    except Exception as e:
        print(f"❌ Error saving people database: {e}")

if __name__ == "__main__":
    print("Testing database functionality...")
    
    # Test loading
    people = load_people_db()
    print(f"Loaded {len(people)} people from database")
    
    # Test saving
    test_data = [
        {
            "id": 1,
            "first_name": "Test",
            "last_name": "Person",
            "personnel_code": "TEST001",
            "department": "Testing",
            "is_active": True
        }
    ]
    
    save_people_db(test_data)
    
    # Test loading again
    people = load_people_db()
    print(f"After save: loaded {len(people)} people from database")
    
    print("Database test completed!")