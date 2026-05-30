"""
prediction.py
-------------
AI-powered health prediction module for the Health Prediction Application.

Primary method  : Anthropic Claude API (claude-haiku-3-5 — fast and cost-effective).
Fallback method : Local rule-based engine (runs offline, no API key required).

The AI model receives the patient's blood test values and returns a concise,
clinically-styled health remark suitable for display in the Remarks field.
"""

import os
import logging
from typing import Tuple

import anthropic

logger = logging.getLogger(__name__)


# ─── REFERENCE RANGES (used by rule-based fallback) ──────────────────────────
# All values in standard clinical units:
#   Glucose      → mg/dL
#   Haemoglobin  → g/dL
#   Cholesterol  → mg/dL

GLUCOSE_NORMAL_MIN = 70.0
GLUCOSE_NORMAL_MAX = 100.0   # Fasting
GLUCOSE_PREDIABETES_MAX = 125.0

HAEMOGLOBIN_MALE_MIN = 13.5
HAEMOGLOBIN_FEMALE_MIN = 12.0
HAEMOGLOBIN_MAX = 17.5

CHOLESTEROL_OPTIMAL_MAX = 200.0
CHOLESTEROL_BORDERLINE_MAX = 239.0


# ─── AI PREDICTION (Claude API) ──────────────────────────────────────────────

def predict_with_ai(
    full_name: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
) -> Tuple[bool, str]:
    """
    Call the Anthropic Claude API to generate a health remark.

    Returns:
        (success: bool, remark: str)
        On failure, success is False and remark contains an error message or
        the rule-based fallback result.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — using rule-based fallback.")
        return False, predict_rule_based(glucose, haemoglobin, cholesterol)

    prompt = f"""You are a clinical health analyst assistant. A patient has submitted the following blood test results:

- Patient Name  : {full_name}
- Glucose       : {glucose} mg/dL
- Haemoglobin   : {haemoglobin} g/dL
- Cholesterol   : {cholesterol} mg/dL

Based on standard clinical reference ranges, provide a concise health remark (3–5 sentences) that:
1. Identifies any values outside normal ranges and what they may indicate.
2. Highlights any values within healthy ranges positively.
3. Recommends a sensible next step (e.g., consult a physician, lifestyle changes).

Important: This is an informational summary only, not a medical diagnosis. Keep the language professional yet accessible. Do not use bullet points — write in clear prose."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        remark = message.content[0].text.strip()
        logger.info("AI prediction generated successfully for patient: %s", full_name)
        return True, remark

    except anthropic.AuthenticationError:
        logger.error("Invalid Anthropic API key.")
        fallback = predict_rule_based(glucose, haemoglobin, cholesterol)
        return False, f"[API key invalid — rule-based result] {fallback}"

    except anthropic.RateLimitError:
        logger.warning("Anthropic rate limit reached — using fallback.")
        fallback = predict_rule_based(glucose, haemoglobin, cholesterol)
        return False, f"[Rate limit reached — rule-based result] {fallback}"

    except Exception as e:
        logger.error("Unexpected error calling Anthropic API: %s", e)
        fallback = predict_rule_based(glucose, haemoglobin, cholesterol)
        return False, f"[API unavailable — rule-based result] {fallback}"


# ─── RULE-BASED FALLBACK ──────────────────────────────────────────────────────

def predict_rule_based(glucose: float, haemoglobin: float, cholesterol: float) -> str:
    """
    Local rule-based health prediction engine.
    Evaluates each blood marker against clinical reference ranges and
    composes a human-readable health remark.
    """
    remarks = []
    risk_flags = []

    # ── Glucose ──
    if glucose < GLUCOSE_NORMAL_MIN:
        risk_flags.append("low blood glucose (hypoglycaemia risk)")
        remarks.append(
            f"Glucose level of {glucose} mg/dL is below the normal fasting range "
            f"({GLUCOSE_NORMAL_MIN}–{GLUCOSE_NORMAL_MAX} mg/dL), which may indicate hypoglycaemia."
        )
    elif glucose <= GLUCOSE_NORMAL_MAX:
        remarks.append(f"Glucose level of {glucose} mg/dL is within the healthy fasting range.")
    elif glucose <= GLUCOSE_PREDIABETES_MAX:
        risk_flags.append("elevated glucose (pre-diabetes risk)")
        remarks.append(
            f"Glucose level of {glucose} mg/dL falls in the pre-diabetic range "
            f"({GLUCOSE_NORMAL_MAX + 1}–{GLUCOSE_PREDIABETES_MAX} mg/dL). Lifestyle modifications are advisable."
        )
    else:
        risk_flags.append("high blood glucose (diabetes risk)")
        remarks.append(
            f"Glucose level of {glucose} mg/dL is significantly elevated, suggesting a potential risk of diabetes. "
            f"Immediate consultation with a healthcare provider is strongly recommended."
        )

    # ── Haemoglobin ──
    if haemoglobin < HAEMOGLOBIN_FEMALE_MIN:
        risk_flags.append("low haemoglobin (anaemia risk)")
        remarks.append(
            f"Haemoglobin level of {haemoglobin} g/dL is below the normal threshold, "
            f"indicating a possible risk of anaemia. Further investigation is advised."
        )
    elif haemoglobin > HAEMOGLOBIN_MAX:
        risk_flags.append("elevated haemoglobin (polycythaemia risk)")
        remarks.append(
            f"Haemoglobin level of {haemoglobin} g/dL is above the normal range, "
            f"which may warrant further evaluation for polycythaemia."
        )
    else:
        remarks.append(f"Haemoglobin level of {haemoglobin} g/dL appears within an acceptable range.")

    # ── Cholesterol ──
    if cholesterol <= CHOLESTEROL_OPTIMAL_MAX:
        remarks.append(f"Total cholesterol of {cholesterol} mg/dL is in the desirable range.")
    elif cholesterol <= CHOLESTEROL_BORDERLINE_MAX:
        risk_flags.append("borderline-high cholesterol")
        remarks.append(
            f"Total cholesterol of {cholesterol} mg/dL is borderline high. "
            f"Dietary adjustments and increased physical activity are recommended."
        )
    else:
        risk_flags.append("high cholesterol (cardiovascular risk)")
        remarks.append(
            f"Total cholesterol of {cholesterol} mg/dL is high, indicating an elevated risk of "
            f"cardiovascular disease. A physician consultation and potential treatment plan are advised."
        )

    # ── Summary ──
    if risk_flags:
        summary = (
            f"Risk indicators identified: {', '.join(risk_flags)}. "
            "Please consult a qualified healthcare professional for a comprehensive evaluation and personalised guidance."
        )
    else:
        summary = (
            "All blood test values appear within normal clinical reference ranges. "
            "Maintain a balanced diet, regular exercise, and schedule routine check-ups to preserve good health."
        )

    return " ".join(remarks) + " " + summary
