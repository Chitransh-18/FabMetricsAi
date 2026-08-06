"""
SQLite Database Integration for Remote Profile Access & Inspection History.
Stores user accounts, batch wafer scan records, defect bounding box coordinates, and yield metrics.
"""

from __future__ import annotations
import os
import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).parent.parent / "fabmetrics.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db() -> None:
    """Initialize SQLite tables for users and inspection history."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Cleanroom Engineer',
            created_at TEXT NOT NULL
        )
    """)

    # Inspection Records Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            defects_count INTEGER DEFAULT 0,
            bounding_boxes_json TEXT,
            image_b64 TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Create Default Admin & Cleanroom Accounts if missing
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_pass = hash_password("password123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", "admin@fabmetrics.ai", admin_pass, "Lead Semiconductor Engineer", time.strftime("%Y-%m-%d %H:%M:%S"))
        )

    cursor.execute("SELECT id FROM users WHERE username = 'engineer'")
    if not cursor.fetchone():
        eng_pass = hash_password("cleanroom2026")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("engineer", "cleanroom@fabmetrics.ai", eng_pass, "Cleanroom Yield Analyst", time.strftime("%Y-%m-%d %H:%M:%S"))
        )

    conn.commit()
    conn.close()

def authenticate_user(username: str, password_raw: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials and return profile info."""
    conn = get_connection()
    cursor = conn.cursor()
    p_hash = hash_password(password_raw)

    cursor.execute(
        "SELECT id, username, email, role, created_at FROM users WHERE (username = ? OR email = ?) AND password_hash = ?",
        (username, username, p_hash)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None

def save_inspection_record(
    user_id: int,
    username: str,
    filename: str,
    predicted_class: str,
    confidence: float,
    defects_count: int = 0,
    bounding_boxes: Optional[List[Dict]] = None,
    image_b64: str = ""
) -> int:
    """Save an inspection scan result to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    bb_json = json.dumps(bounding_boxes or [])

    cursor.execute(
        """
        INSERT INTO inspection_records 
        (user_id, username, filename, predicted_class, confidence, defects_count, bounding_boxes_json, image_b64, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            username,
            filename,
            predicted_class,
            float(confidence),
            defects_count,
            bb_json,
            image_b64,
            time.strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def get_user_inspections(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch past inspection history from database."""
    conn = get_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute(
            """
            SELECT id, user_id, username, filename, predicted_class, confidence, defects_count, bounding_boxes_json, timestamp 
            FROM inspection_records WHERE user_id = ? ORDER BY id DESC LIMIT ?
            """,
            (user_id, limit)
        )
    else:
        cursor.execute(
            """
            SELECT id, user_id, username, filename, predicted_class, confidence, defects_count, bounding_boxes_json, timestamp 
            FROM inspection_records ORDER BY id DESC LIMIT ?
            """,
            (limit,)
        )

    rows = cursor.fetchall()
    conn.close()

    return results

def get_inspection_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Fetch single inspection record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, user_id, username, filename, predicted_class, confidence, defects_count, bounding_boxes_json, image_b64, timestamp 
        FROM inspection_records WHERE id = ?
        """,
        (record_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        item = dict(row)
        item["bounding_boxes"] = json.loads(item["bounding_boxes_json"] or "[]")
        del item["bounding_boxes_json"]
        return item
    return None


def get_database_analytics() -> Dict[str, Any]:
    """Calculate aggregate yield statistics from database inspection records."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_scans FROM inspection_records")
    total_scans = cursor.fetchone()["total_scans"]

    cursor.execute("SELECT COUNT(*) as non_defective FROM inspection_records WHERE predicted_class = 'none'")
    non_defective = cursor.fetchone()["non_defective"]

    cursor.execute("SELECT predicted_class, COUNT(*) as count FROM inspection_records GROUP BY predicted_class")
    class_distribution = {row["predicted_class"]: row["count"] for row in cursor.fetchall()}

    conn.close()

    yield_rate = round((non_defective / total_scans * 100), 2) if total_scans > 0 else 100.0

    return {
        "total_scans": total_scans,
        "clean_wafers": non_defective,
        "defective_wafers": total_scans - non_defective,
        "yield_rate_percent": yield_rate,
        "class_distribution": class_distribution
    }

# Initialize Database on Import
init_db()
