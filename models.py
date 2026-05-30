"""
models.py
---------
Data models for the Health Prediction Application.
Uses Python dataclasses for clean, type-annotated data structures.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class Patient:
    """
    Represents a patient record as stored in the database.

    All blood test values use standard clinical units:
        glucose      → mg/dL
        haemoglobin  → g/dL
        cholesterol  → mg/dL
    """

    full_name: str
    dob: date
    email: str
    glucose: float
    haemoglobin: float
    cholesterol: float
    remarks: str = ""
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def age(self) -> int:
        """Calculate the patient's current age in years."""
        today = date.today()
        dob = self.dob if isinstance(self.dob, date) else datetime.strptime(str(self.dob), "%Y-%m-%d").date()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    @property
    def dob_str(self) -> str:
        """Return DOB as a formatted string (DD MMM YYYY)."""
        dob = self.dob if isinstance(self.dob, date) else datetime.strptime(str(self.dob), "%Y-%m-%d").date()
        return dob.strftime("%d %b %Y")

    @property
    def has_remarks(self) -> bool:
        """True if AI-generated remarks have been assigned."""
        return bool(self.remarks and self.remarks.strip())

    # ── Factory method ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        """
        Construct a Patient instance from a database row dict.
        Handles type coercion (e.g., string → float for blood values).
        """
        dob_raw = data.get("dob")
        if isinstance(dob_raw, str):
            dob = datetime.strptime(dob_raw, "%Y-%m-%d").date()
        elif isinstance(dob_raw, date):
            dob = dob_raw
        else:
            dob = dob_raw  # Leave as-is; validation will catch bad values

        return cls(
            id=data.get("id"),
            full_name=str(data.get("full_name", "")).strip(),
            dob=dob,
            email=str(data.get("email", "")).strip(),
            glucose=float(data.get("glucose", 0.0)),
            haemoglobin=float(data.get("haemoglobin", 0.0)),
            cholesterol=float(data.get("cholesterol", 0.0)),
            remarks=str(data.get("remarks", "")),
            created_at=data.get("created_at"),
        )

    def to_dict(self) -> dict:
        """Serialise the Patient to a plain dict (suitable for database writes)."""
        return {
            "id": self.id,
            "full_name": self.full_name,
            "dob": self.dob.strftime("%Y-%m-%d") if isinstance(self.dob, date) else str(self.dob),
            "email": self.email,
            "glucose": self.glucose,
            "haemoglobin": self.haemoglobin,
            "cholesterol": self.cholesterol,
            "remarks": self.remarks,
            "created_at": str(self.created_at) if self.created_at else None,
        }


# ─── Risk Level Helper ────────────────────────────────────────────────────────

def get_risk_level(glucose: float, haemoglobin: float, cholesterol: float) -> str:
    """
    Derive a simple risk label (Low / Moderate / High) from blood test values.
    Used for UI badge colouring in the dashboard table.
    """
    high_risk = (
        glucose > 125
        or haemoglobin < 12.0
        or cholesterol > 239
    )
    moderate_risk = (
        100 < glucose <= 125
        or 12.0 <= haemoglobin < 13.5
        or 200 < cholesterol <= 239
    )

    if high_risk:
        return "High"
    if moderate_risk:
        return "Moderate"
    return "Low"
