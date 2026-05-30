"""
app.py
------
Main entry point for the MIRA Health Prediction Application.
Built with Streamlit — run with: streamlit run app.py
"""

import sqlite3
import os
import logging
from datetime import date, datetime

import streamlit as st
import pandas as pd

import database as db
import validation as val
from models import Patient, get_risk_level
from prediction import predict_with_ai

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRA — Health Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global Typography & Palette ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
      color: #e0f2f1;
  }
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
      color: #b2dfdb !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
      color: #ffffff !important;
  }

  /* ── Hero Banner ── */
  .hero-banner {
      background: linear-gradient(135deg, #00695c 0%, #004d40 50%, #0d47a1 100%);
      border-radius: 16px;
      padding: 28px 36px;
      margin-bottom: 24px;
      color: white;
  }
  .hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; }
  .hero-banner p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }

  /* ── Metric Cards ── */
  .metric-card {
      background: white;
      border-radius: 12px;
      padding: 20px 24px;
      border-left: 5px solid #00897b;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      margin-bottom: 16px;
  }
  .metric-card h2 { font-size: 2rem; font-weight: 700; color: #00695c; margin: 0; }
  .metric-card p  { font-size: 0.85rem; color: #757575; margin: 0; }

  /* ── Risk Badges ── */
  .badge-low      { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-moderate { background:#fff8e1; color:#f57f17; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-high     { background:#ffebee; color:#c62828; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

  /* ── Form Container ── */
  .form-card {
      background: #f9fafb;
      border: 1px solid #e0e0e0;
      border-radius: 14px;
      padding: 28px 32px;
      margin-bottom: 20px;
  }

  /* ── Section Headers ── */
  .section-header {
      font-size: 1.05rem;
      font-weight: 600;
      color: #37474f;
      border-bottom: 2px solid #b2dfdb;
      padding-bottom: 6px;
      margin-bottom: 16px;
  }

  /* ── Remarks Box ── */
  .remarks-box {
      background: #e8f5e9;
      border-left: 4px solid #43a047;
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 0.9rem;
      color: #1b5e20;
      line-height: 1.6;
  }

  /* ── Action Buttons ── */
  div.stButton > button {
      border-radius: 8px;
      font-weight: 500;
      transition: all 0.2s ease;
  }

  /* Hide default Streamlit watermark */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Initialise DB ────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
db.initialise_database()


# ─── Session State Defaults ───────────────────────────────────────────────────
for key, default in {
    "edit_patient_id": None,
    "view_patient_id": None,
    "confirm_delete_id": None,
    "search_query": "",
    "active_section": "Dashboard",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MIRA")
    st.markdown("**Medical Intelligence**  \nRobotic Automation")
    st.divider()

    nav = st.radio(
        "Navigate",
        ["📊 Dashboard", "➕ Add Patient", "📋 Patient Records", "ℹ️ About"],
        index=0,
    )
    section = nav.split(" ", 1)[1]  # Strip emoji

    st.divider()
    st.markdown("**AI Engine**")
    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    if api_key_set:
        st.success("Claude API ✅ Connected")
    else:
        st.warning("Rule-based fallback active")
        st.caption("Set ANTHROPIC_API_KEY in .env to enable AI remarks.")


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def show_success(msg: str):
    st.success(f"✅ {msg}")

def show_error(msg: str):
    st.error(f"❌ {msg}")

def show_warning(msg: str):
    st.warning(f"⚠️ {msg}")

def risk_badge(level: str) -> str:
    cls = f"badge-{level.lower()}"
    return f'<span class="{cls}">{level}</span>'


def render_patient_form(
    prefix: str,
    defaults: dict = None,
) -> dict:
    """
    Render the patient data entry form and return field values.
    prefix  : unique key prefix to avoid widget ID collisions (e.g. 'add' or 'edit')
    defaults: dict of pre-filled values for edit mode
    """
    d = defaults or {}

    st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full Name *", value=d.get("full_name", ""), key=f"{prefix}_name")
    with c2:
        email = st.text_input("Email Address *", value=d.get("email", ""), key=f"{prefix}_email")

    dob_default = None
    if d.get("dob"):
        try:
            dob_default = datetime.strptime(str(d["dob"]), "%Y-%m-%d").date()
        except Exception:
            dob_default = None

    dob = st.date_input(
        "Date of Birth *",
        value=dob_default,
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        key=f"{prefix}_dob",
    )

    st.markdown('<div class="section-header" style="margin-top:20px">🩸 Blood Test Results</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        glucose = st.number_input(
            "Glucose (mg/dL) *", min_value=0.0, max_value=1000.0,
            value=float(d.get("glucose", 0.0)), step=0.1, format="%.1f", key=f"{prefix}_glucose",
        )
    with b2:
        haemoglobin = st.number_input(
            "Haemoglobin (g/dL) *", min_value=0.0, max_value=25.0,
            value=float(d.get("haemoglobin", 0.0)), step=0.1, format="%.1f", key=f"{prefix}_hgb",
        )
    with b3:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL) *", min_value=0.0, max_value=1000.0,
            value=float(d.get("cholesterol", 0.0)), step=0.1, format="%.1f", key=f"{prefix}_chol",
        )

    return {
        "full_name": full_name,
        "dob": dob,
        "email": email,
        "glucose": glucose,
        "haemoglobin": haemoglobin,
        "cholesterol": cholesterol,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if "Dashboard" in section:

    st.markdown("""
    <div class="hero-banner">
        <h1>🏥 MIRA Health Prediction</h1>
        <p>Medical Intelligence Robotic Automation — Patient Blood Test Analytics & AI Health Insights</p>
    </div>
    """, unsafe_allow_html=True)

    all_patients = db.get_all_patients()
    total = len(all_patients)
    with_remarks = sum(1 for p in all_patients if p.get("remarks", "").strip())
    high_risk = sum(
        1 for p in all_patients
        if get_risk_level(p["glucose"], p["haemoglobin"], p["cholesterol"]) == "High"
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, val_, label in [
        (m1, total, "Total Patients"),
        (m2, with_remarks, "AI Remarks Generated"),
        (m3, high_risk, "High Risk Patients"),
        (m4, total - high_risk, "Low / Moderate Risk"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{val_}</h2>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    if total == 0:
        st.info("No patient records yet. Use **➕ Add Patient** to get started.")
    else:
        st.markdown("### 📋 Recent Patients")
        recent = all_patients[:5]
        for p in recent:
            risk = get_risk_level(p["glucose"], p["haemoglobin"], p["cholesterol"])
            st.markdown(
                f"**{p['full_name']}** &nbsp;·&nbsp; {p['email']} &nbsp;·&nbsp; "
                f"Glucose: {p['glucose']} · Hgb: {p['haemoglobin']} · Chol: {p['cholesterol']} &nbsp; "
                + risk_badge(risk),
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: ADD PATIENT
# ═══════════════════════════════════════════════════════════════════════════════

elif "Add Patient" in section:

    st.markdown("## ➕ Register New Patient")
    st.caption("All fields marked * are required.")

    with st.form("add_patient_form", clear_on_submit=True):
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        form_data = render_patient_form("add")
        st.markdown('</div>', unsafe_allow_html=True)

        col_save, col_ai, _ = st.columns([1, 1, 3])
        with col_save:
            submit = st.form_submit_button("💾 Save Patient", use_container_width=True, type="primary")
        with col_ai:
            submit_ai = st.form_submit_button("🤖 Save + Generate AI Remarks", use_container_width=True)

    if submit or submit_ai:
        is_valid, errors = val.validate_patient_form(
            form_data["full_name"],
            form_data["dob"],
            form_data["email"],
            form_data["glucose"],
            form_data["haemoglobin"],
            form_data["cholesterol"],
        )
        if not is_valid:
            for err in errors:
                show_error(err)
        else:
            try:
                patient_id = db.create_patient(
                    full_name=form_data["full_name"],
                    dob=str(form_data["dob"]),
                    email=form_data["email"],
                    glucose=form_data["glucose"],
                    haemoglobin=form_data["haemoglobin"],
                    cholesterol=form_data["cholesterol"],
                )
                if submit_ai:
                    with st.spinner("🤖 Generating AI health remarks…"):
                        _, remarks = predict_with_ai(
                            form_data["full_name"],
                            form_data["glucose"],
                            form_data["haemoglobin"],
                            form_data["cholesterol"],
                        )
                    db.update_remarks(patient_id, remarks)
                    show_success(f"Patient saved and AI remarks generated for {form_data['full_name']}!")
                    st.markdown(f'<div class="remarks-box">{remarks}</div>', unsafe_allow_html=True)
                else:
                    show_success(f"Patient '{form_data['full_name']}' registered successfully!")
            except sqlite3.IntegrityError:
                show_error("A patient with this email address already exists.")
            except Exception as e:
                show_error(f"Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: PATIENT RECORDS (Read / Update / Delete)
# ═══════════════════════════════════════════════════════════════════════════════

elif "Patient Records" in section:

    st.markdown("## 📋 Patient Records")

    # ── Search Bar ──
    search_col, _ = st.columns([2, 3])
    with search_col:
        search_query = st.text_input("🔍 Search by name or email", value=st.session_state.search_query, placeholder="e.g. John or john@example.com")
    st.session_state.search_query = search_query

    if search_query.strip():
        patients = db.search_patients(search_query.strip())
        st.caption(f"{len(patients)} result(s) for {search_query})
    else:
        patients = db.get_all_patients()

    if not patients:
        st.info("No patient records found.")
    else:
        # ── Table Header ──
        hdr = st.columns([3, 2, 2, 1.5, 1.5, 1.5, 1.5, 3, 3])
        for col, label in zip(hdr, ["Name", "Email", "DOB", "Glucose", "Hgb", "Chol", "Risk", "Remarks", "Actions"]):
            col.markdown(f"**{label}**")
        st.divider()

        for p in patients:
            risk = get_risk_level(p["glucose"], p["haemoglobin"], p["cholesterol"])
            row = st.columns([3, 2, 2, 1.5, 1.5, 1.5, 1.5, 3, 3])

            row[0].write(p["full_name"])
            row[1].write(p["email"])
            row[2].write(p["dob"])
            row[3].write(f"{p['glucose']:.1f}")
            row[4].write(f"{p['haemoglobin']:.1f}")
            row[5].write(f"{p['cholesterol']:.1f}")
            row[6].markdown(risk_badge(risk), unsafe_allow_html=True)

            remarks_preview = (p.get("remarks") or "—")
            if len(remarks_preview) > 60:
                remarks_preview = remarks_preview[:57] + "…"
            row[7].caption(remarks_preview)

            # ── Action Buttons ──
            with row[8]:
                btn_cols = st.columns(3)
                if btn_cols[0].button("✏️", key=f"edit_{p['id']}", help="Edit"):
                    st.session_state.edit_patient_id = p["id"]
                if btn_cols[1].button("🗑️", key=f"del_{p['id']}", help="Delete"):
                    st.session_state.confirm_delete_id = p["id"]
                if btn_cols[2].button("🤖", key=f"ai_{p['id']}", help="Generate AI Remarks"):
                    with st.spinner("Generating AI remarks…"):
                        _, remarks = predict_with_ai(
                            p["full_name"], p["glucose"], p["haemoglobin"], p["cholesterol"]
                        )
                    db.update_remarks(p["id"], remarks)
                    show_success(f"Remarks updated for {p['full_name']}")
                    st.rerun()

        st.divider()

        # ── Confirm Delete Modal ──
        if st.session_state.confirm_delete_id:
            pid = st.session_state.confirm_delete_id
            patient = db.get_patient_by_id(pid)
            if patient:
                st.warning(f"⚠️ Are you sure you want to delete **{patient['full_name']}**? This cannot be undone.")
                dc1, dc2, _ = st.columns([1, 1, 4])
                if dc1.button("✅ Confirm Delete", type="primary"):
                    db.delete_patient(pid)
                    st.session_state.confirm_delete_id = None
                    show_success(f"Patient '{patient['full_name']}' deleted.")
                    st.rerun()
                if dc2.button("❌ Cancel"):
                    st.session_state.confirm_delete_id = None
                    st.rerun()

        # ── Edit Patient Panel ──
        if st.session_state.edit_patient_id:
            pid = st.session_state.edit_patient_id
            patient = db.get_patient_by_id(pid)
            if patient:
                st.divider()
                st.markdown(f"### ✏️ Edit Patient — {patient['full_name']}")

                with st.form(f"edit_form_{pid}"):
                    form_data = render_patient_form("edit", defaults=patient)
                    ec1, ec2, ec3 = st.columns([1, 1, 3])
                    with ec1:
                        save_edit = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)
                    with ec2:
                        cancel_edit = st.form_submit_button("✖ Cancel", use_container_width=True)

                if save_edit:
                    is_valid, errors = val.validate_patient_form(
                        form_data["full_name"],
                        form_data["dob"],
                        form_data["email"],
                        form_data["glucose"],
                        form_data["haemoglobin"],
                        form_data["cholesterol"],
                    )
                    if not is_valid:
                        for err in errors:
                            show_error(err)
                    else:
                        db.update_patient(
                            pid,
                            full_name=form_data["full_name"],
                            dob=str(form_data["dob"]),
                            email=form_data["email"],
                            glucose=form_data["glucose"],
                            haemoglobin=form_data["haemoglobin"],
                            cholesterol=form_data["cholesterol"],
                            remarks=patient.get("remarks", ""),
                        )
                        st.session_state.edit_patient_id = None
                        show_success("Patient record updated successfully.")
                        st.rerun()

                if cancel_edit:
                    st.session_state.edit_patient_id = None
                    st.rerun()

    # ── Export ──
    if patients:
        st.divider()
        df = pd.DataFrame(patients)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export to CSV",
            data=csv,
            file_name="mira_patients.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

elif "About" in section:
    st.markdown("## ℹ️ About MIRA")
    st.markdown("""
    **MIRA — Medical Intelligence Robotic Automation**

    MIRA is a Python-based Health Prediction Application designed to help healthcare
    professionals manage patient blood test records and receive AI-powered health insights.

    ### Features
    - ✅ Full CRUD management of patient records
    - 🔍 Search & filter functionality
    - 🧪 Blood test tracking (Glucose, Haemoglobin, Cholesterol)
    - 🤖 AI-generated health remarks via Claude API
    - 📊 Risk level classification (Low / Moderate / High)
    - 📥 CSV export
    - 💾 Persistent SQLite database
    - ✔️ Input validation on all fields

    ### Tech Stack
    | Layer      | Technology            |
    |------------|-----------------------|
    | Frontend   | Streamlit             |
    | Backend    | Python 3.11+          |
    | Database   | SQLite                |
    | AI Engine  | Anthropic Claude API  |
    | Packaging  | pip / requirements.txt|

    ### Reference Ranges Used
    | Marker       | Normal Range                  |
    |--------------|-------------------------------|
    | Glucose      | 70–100 mg/dL (fasting)        |
    | Haemoglobin  | 12.0–17.5 g/dL                |
    | Cholesterol  | < 200 mg/dL (desirable)       |
    """)
