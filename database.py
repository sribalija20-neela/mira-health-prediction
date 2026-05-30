"""
database.py
-----------
Handles all SQLite database operations for the Health Prediction Application.
Provides functions for creating, reading, updating, and deleting patient records.
"""

import sqlite3
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/health_prediction.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database() -> None:
    """
    Create the patients table if it does not already exist.
    Called once on application startup.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS patients (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name       TEXT    NOT NULL,
        dob             DATE    NOT NULL,
        email           TEXT    NOT NULL UNIQUE,
        glucose         REAL    NOT NULL,
        haemoglobin     REAL    NOT NULL,
        cholesterol     REAL    NOT NULL,
        remarks         TEXT    DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with get_connection() as conn:
            conn.execute(sql)
            conn.commit()
        logger.info("Database initialised successfully.")
    except sqlite3.Error as e:
        logger.error("Failed to initialise database: %s", e)
        raise


# ─── CREATE ──────────────────────────────────────────────────────────────────

def create_patient(
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str = "",
) -> int:
    """
    Insert a new patient record. Returns the new patient's auto-generated ID.
    Raises sqlite3.IntegrityError if the email already exists.
    """
    sql = """
    INSERT INTO patients (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql, (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks))
            conn.commit()
            logger.info("Patient created with ID %d.", cursor.lastrowid)
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.warning("Duplicate email: %s", email)
        raise
    except sqlite3.Error as e:
        logger.error("Error creating patient: %s", e)
        raise


# ─── READ ─────────────────────────────────────────────────────────────────────

def get_all_patients() -> list:
    """Retrieve all patient records ordered by creation date (newest first)."""
    sql = "SELECT * FROM patients ORDER BY created_at DESC"
    try:
        with get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Error fetching patients: %s", e)
        return []


def get_patient_by_id(patient_id: int) -> Optional[dict]:
    """Retrieve a single patient record by ID."""
    sql = "SELECT * FROM patients WHERE id = ?"
    try:
        with get_connection() as conn:
            row = conn.execute(sql, (patient_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error("Error fetching patient %d: %s", patient_id, e)
        return None


def search_patients(query: str) -> list:
    """Search patients by name or email (case-insensitive partial match)."""
    sql = """
    SELECT * FROM patients
    WHERE LOWER(full_name) LIKE ? OR LOWER(email) LIKE ?
    ORDER BY created_at DESC
    """
    pattern = f"%{query.lower()}%"
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, (pattern, pattern)).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Error searching patients: %s", e)
        return []


# ─── UPDATE ───────────────────────────────────────────────────────────────────

def update_patient(
    patient_id: int,
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str = "",
) -> bool:
    """Update an existing patient record. Returns True on success."""
    sql = """
    UPDATE patients
    SET full_name=?, dob=?, email=?, glucose=?, haemoglobin=?, cholesterol=?, remarks=?
    WHERE id=?
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql, (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, patient_id))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info("Patient %d updated.", patient_id)
            return success
    except sqlite3.Error as e:
        logger.error("Error updating patient %d: %s", patient_id, e)
        return False


def update_remarks(patient_id: int, remarks: str) -> bool:
    """Update only the remarks field (called after AI prediction)."""
    sql = "UPDATE patients SET remarks=? WHERE id=?"
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql, (remarks, patient_id))
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Error updating remarks for patient %d: %s", patient_id, e)
        return False


# ─── DELETE ───────────────────────────────────────────────────────────────────

def delete_patient(patient_id: int) -> bool:
    """Delete a patient record by ID. Returns True on success."""
    sql = "DELETE FROM patients WHERE id=?"
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql, (patient_id,))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info("Patient %d deleted.", patient_id)
            return success
    except sqlite3.Error as e:
        logger.error("Error deleting patient %d: %s", patient_id, e)
        return False
