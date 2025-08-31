#!/usr/bin/env python3
"""
SQLite database setup and operations for Attendance System
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import os

DB_FILE = Path("attendance_system.db")

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Create people table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                personnel_code TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                position TEXT,
                email TEXT,
                phone TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create enrollments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                face_encoding_path TEXT,
                face_image_path TEXT,
                confidence_score REAL,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE
            )
        ''')
        
        # Create recognition_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_location TEXT,
                image_path TEXT,
                FOREIGN KEY (person_id) REFERENCES people (id)
            )
        ''')
        
        conn.commit()
        print("SUCCESS: Database initialized successfully")
        
    except Exception as e:
        print(f"ERROR: Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def dict_from_row(row):
    """Convert SQLite row to dictionary"""
    return {key: row[key] for key in row.keys()}

class PeopleDB:
    """Database operations for people management"""
    
    @staticmethod
    def get_all(skip: int = 0, limit: int = 100, search: Optional[str] = None, 
                status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all people with optional filtering"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM people WHERE 1=1"
            params = []
            
            # Add search filter
            if search:
                query += " AND (first_name LIKE ? OR last_name LIKE ? OR personnel_code LIKE ? OR department LIKE ?)"
                search_param = f"%{search}%"
                params.extend([search_param, search_param, search_param, search_param])
            
            # Add status filter
            if status == "active":
                query += " AND is_active = 1"
            elif status == "inactive":
                query += " AND is_active = 0"
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            people = []
            for row in rows:
                person = dict_from_row(row)
                # Get enrollments for this person
                person['enrollments'] = PeopleDB.get_enrollments(person['id'])
                people.append(person)
            
            return people
            
        except Exception as e:
            print(f"❌ Error getting people: {e}")
            return []
        finally:
            conn.close()
    
    @staticmethod
    def get_by_id(person_id: int) -> Optional[Dict[str, Any]]:
        """Get person by ID"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM people WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            
            if row:
                person = dict_from_row(row)
                person['enrollments'] = PeopleDB.get_enrollments(person_id)
                return person
            return None
            
        except Exception as e:
            print(f"❌ Error getting person: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def create(person_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new person"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO people (first_name, last_name, personnel_code, department, position, email, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_data['first_name'],
                person_data['last_name'],
                person_data['personnel_code'],
                person_data['department'],
                person_data.get('position'),
                person_data.get('email'),
                person_data.get('phone')
            ))
            
            person_id = cursor.lastrowid
            conn.commit()
            
            # Return the created person
            return PeopleDB.get_by_id(person_id)
            
        except sqlite3.IntegrityError as e:
            print(f"❌ Integrity error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error creating person: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def update(person_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update person"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Build dynamic update query
            fields = []
            values = []
            
            for key, value in update_data.items():
                if key != 'id':  # Don't update ID
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            if not fields:
                return PeopleDB.get_by_id(person_id)
            
            values.append(person_id)
            query = f"UPDATE people SET {', '.join(fields)} WHERE id = ?"
            
            cursor.execute(query, values)
            conn.commit()
            
            if cursor.rowcount > 0:
                return PeopleDB.get_by_id(person_id)
            return None
            
        except sqlite3.IntegrityError as e:
            print(f"❌ Integrity error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error updating person: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def delete(person_id: int) -> bool:
        """Delete person"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM people WHERE id = ?", (person_id,))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"❌ Error deleting person: {e}")
            return False
        finally:
            conn.close()
    
    @staticmethod
    def toggle_status(person_id: int) -> Optional[Dict[str, Any]]:
        """Toggle person's active status"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("UPDATE people SET is_active = NOT is_active WHERE id = ?", (person_id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                return PeopleDB.get_by_id(person_id)
            return None
            
        except Exception as e:
            print(f"❌ Error toggling status: {e}")
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_enrollments(person_id: int) -> List[Dict[str, Any]]:
        """Get enrollments for a person"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM enrollments WHERE person_id = ? AND is_active = 1", (person_id,))
            rows = cursor.fetchall()
            return [dict_from_row(row) for row in rows]
            
        except Exception as e:
            print(f"❌ Error getting enrollments: {e}")
            return []
        finally:
            conn.close()

# Initialize database on module import
if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print("Database setup complete!")
else:
    # Initialize database when module is imported
    init_database()