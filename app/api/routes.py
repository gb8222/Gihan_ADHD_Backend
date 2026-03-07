import secrets
import string
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext

from app.api.deps import require_supabase_user
from app.api.schemas import (
    MeOut,
    DoctorRegisterIn,
    PatientCreateIn,
    PatientCreateOut,
    PatientLoginIn,
    TokenOut,
)
from app.core.security import create_patient_token

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/me", response_model=MeOut)
async def me(user=Depends(require_supabase_user)):
    meta = user["claims"].get("user_metadata", {}) or {}
    role = meta.get("role") or "guest"
    # Return mock profile data
    return MeOut(
        role=role, 
        status="approved" if role != "doctor" else "pending", 
        email=user["claims"].get("email") or "", 
        full_name=meta.get("full_name") or meta.get("name") or "User"
    )

@router.post("/doctors/register")
async def doctor_register(payload: DoctorRegisterIn, user=Depends(require_supabase_user)):
    # Mock successful submission
    return {"submitted": True}


@router.get("/admin/doctor-requests")
async def admin_list_doctor_requests():
    # Mock empty list
    return []


@router.post("/admin/doctors/{doctor_profile_id}/approve")
async def admin_approve_doctor(doctor_profile_id: str):
    # Mock successful approval
    return {"approved": True}


@router.post("/admin/doctors/{doctor_profile_id}/reject")
async def admin_reject_doctor(doctor_profile_id: str):
    # Mock successful rejection
    return {"rejected": True}


def _gen_patient_id():
    alphabet = string.ascii_uppercase + string.digits
    return "PT-" + "".join(secrets.choice(alphabet) for _ in range(6))


def _gen_password():
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))


@router.post("/doctor/patients", response_model=PatientCreateOut)
async def doctor_create_patient(payload: PatientCreateIn):
    patient_id = payload.patient_id or _gen_patient_id()
    password = payload.password or _gen_password()

    # Mock successful patient creation
    return PatientCreateOut(patient_id=patient_id, password=password)


@router.get("/doctor/patients")
async def doctor_list_patients():
    # Mock empty patient list
    return []


@router.post("/auth/patient/login", response_model=TokenOut)
async def patient_login(payload: PatientLoginIn):
    # Mock patient login
    token = create_patient_token(sub="mock-id", patient_id=payload.patient_id)
    return TokenOut(access_token=token)


@router.get("/parent/me")
async def parent_me():
    # Mock parent profile
    return {
        "patient_id": "PT-MOCK01",
        "parent_name": "Mock Parent",
        "child_name": "Mock Child",
        "contact_email": "mock@example.com",
    }
