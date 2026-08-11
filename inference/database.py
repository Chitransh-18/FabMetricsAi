"""
SQLite Database Integration for Remote Profile Access & Inspection History.
Stores user accounts, user sessions, batch wafer scan records, defect bounding box coordinates, and yield metrics.
"""

from __future__ import annotations
import os
import json
import sqlite3
import hashlib
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

DB_PATH = Path(__file__).parent.parent / "fabmetrics.db"

def get_connection() -> sqlite3.Connection:
    """Get SQLite database connection with WAL mode and busy timeout enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"

def verify_password(password_raw: str, stored_hash: str) -> bool:
    """Verify raw password against stored salted PBKDF2 or legacy hash."""
    try:
        if "$" in stored_hash:
            salt, key_hex = stored_hash.split("$", 1)
            check = hashlib.pbkdf2_hmac("sha256", password_raw.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
            return secrets.compare_digest(check, key_hex)
        else:
            # Backward compatibility for plain SHA-256 legacy hash
            legacy_hash = hashlib.sha256(password_raw.encode("utf-8")).hexdigest()
            return secrets.compare_digest(legacy_hash, stored_hash)
    except Exception:
        return False

def init_db() -> None:
    """Initialize SQLite tables for users, sessions, and inspection history."""
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

    # User Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
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
    cursor.execute("SELECT id, password_hash FROM users WHERE username = 'admin'")
    row_admin = cursor.fetchone()
    if not row_admin:
        admin_pass = hash_password("password123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", "admin@fabmetrics.ai", admin_pass, "Lead Semiconductor Engineer", time.strftime("%Y-%m-%d %H:%M:%S"))
        )
    elif "$" not in row_admin["password_hash"]:
        # Migrate legacy hash to salted PBKDF2
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'admin'", (hash_password("password123"),))

    cursor.execute("SELECT id, password_hash FROM users WHERE username = 'engineer'")
    row_eng = cursor.fetchone()
    if not row_eng:
        eng_pass = hash_password("cleanroom2026")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("engineer", "cleanroom@fabmetrics.ai", eng_pass, "Cleanroom Yield Analyst", time.strftime("%Y-%m-%d %H:%M:%S"))
        )
    elif "$" not in row_eng["password_hash"]:
        # Migrate legacy hash to salted PBKDF2
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'engineer'", (hash_password("cleanroom2026"),))

    conn.commit()
    conn.close()

def authenticate_user(username: str, password_raw: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials and return profile info."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, email, password_hash, role, created_at FROM users WHERE username = ? OR email = ?",
        (username, username)
    )
    row = cursor.fetchone()
    conn.close()

    if row and verify_password(password_raw, row["password_hash"]):
        user_dict = dict(row)
        del user_dict["password_hash"]
        return user_dict
    return None

def create_session(user_id: int, duration_seconds: int = 86400 * 7) -> str:
    """Create a persistent session token for authenticated user."""
    token = secrets.token_hex(32)
    now = time.time()
    expires = now + duration_seconds
    conn = get_connection()
    conn.execute(
        "INSERT INTO user_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now, expires)
    )
    conn.commit()
    conn.close()
    return token

def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Fetch user profile by session token if active."""
    if not token:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    now = time.time()
    cursor.execute(
        """
        SELECT u.id, u.username, u.email, u.role, u.created_at 
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ? AND s.expires_at > ?
        """,
        (token, now)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

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

    results = []
    for r in rows:
        item = dict(r)
        item["bounding_boxes"] = json.loads(item["bounding_boxes_json"] or "[]")
        del item["bounding_boxes_json"]
        results.append(item)

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
