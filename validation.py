"""
validation.py
-------------
Reusable input validation functions for the Health Prediction Application.
All functions return a tuple of (is_valid: bool, error_message: str).
"""

import re
import logging
from datetime import date, datetime
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_full_name(name: str) -> Tuple[bool, str]:
    """
    Validate that the full name is not empty and contains only valid characters.
    Minimum 2 characters, letters, spaces, hyphens, and apostrophes allowed.
    """
    name = name.strip()
    if not name:
        return False, "Full name cannot be empty."
    if len(name) < 2:
        return False, "Full name must be at least 2 characters long."
    if not re.match(r"^[A-Za-z\s'\-]+$", name):
        return False, "Full name may only contain letters, spaces, hyphens, and apostrophes."
    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format using a standard RFC 5322-inspired regex.
    """
    email = email.strip()
    if not email:
        return False, "Email address cannot be empty."
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g., name@example.com)."
    return True, ""


def validate_dob(dob_input) -> Tuple[bool, str]:
    """
    Validate date of birth:
    - Must not be empty / None
    - Cannot be a future date
    - Patient must be at least 1 year old (no same-day DOB)
    - Accepts a datetime.date object or a string in YYYY-MM-DD format
    """
    if dob_input is None:
        return False, "Date of birth cannot be empty."

    # Normalise to a date object
    if isinstance(dob_input, datetime):
        dob = dob_input.date()
    elif isinstance(dob_input, date):
        dob = dob_input
    else:
        try:
            dob = datetime.strptime(str(dob_input), "%Y-%m-%d").date()
        except ValueError:
            return False, "Invalid date format. Please use YYYY-MM-DD."

    today = date.today()
    if dob > today:
        return False, "Date of birth cannot be a future date."
    if dob == today:
        return False, "Date of birth cannot be today's date."

    age_years = (today - dob).days / 365.25
    if age_years > 120:
        return False, "Date of birth seems too far in the past. Please check the value."

    return True, ""


def validate_blood_value(
    value,
    field_name: str,
    min_val: float = 0.0,
    max_val: float = 10000.0,
) -> Tuple[bool, str]:
    """
    Validate that a blood test value is numeric and within a physiologically plausible range.

    Args:
        value:      The raw input to validate.
        field_name: Human-readable name for error messages (e.g., "Glucose").
        min_val:    Minimum acceptable value (default 0).
        max_val:    Maximum acceptable value (default 10 000).
    """
    if value is None or str(value).strip() == "":
        return False, f"{field_name} cannot be empty."

    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return False, f"{field_name} must be a numeric value."

    if numeric < min_val:
        return False, f"{field_name} must be ≥ {min_val}."
    if numeric > max_val:
        return False, f"{field_name} must be ≤ {max_val}."

    return True, ""


def validate_glucose(value) -> Tuple[bool, str]:
    """Glucose (mg/dL): 0 – 1000."""
    return validate_blood_value(value, "Glucose", min_val=0.0, max_val=1000.0)


def validate_haemoglobin(value) -> Tuple[bool, str]:
    """Haemoglobin (g/dL): 0 – 25."""
    return validate_blood_value(value, "Haemoglobin", min_val=0.0, max_val=25.0)


def validate_cholesterol(value) -> Tuple[bool, str]:
    """Cholesterol (mg/dL): 0 – 1000."""
    return validate_blood_value(value, "Cholesterol", min_val=0.0, max_val=1000.0)


def validate_patient_form(
    full_name: str,
    dob,
    email: str,
    glucose,
    haemoglobin,
    cholesterol,
) -> Tuple[bool, list]:
    """
    Run all validations for the patient registration/edit form.

    Returns:
        (all_valid: bool, errors: list[str])
        errors is an empty list when all_valid is True.
    """
    errors = []

    checks = [
        validate_full_name(full_name),
        validate_dob(dob),
        validate_email(email),
        validate_glucose(glucose),
        validate_haemoglobin(haemoglobin),
        validate_cholesterol(cholesterol),
    ]

    for is_valid, message in checks:
        if not is_valid:
            errors.append(message)

    return len(errors) == 0, errors
