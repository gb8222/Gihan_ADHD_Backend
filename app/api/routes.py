import secrets
import string
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext

from app.api.deps import get_db, require_admin, require_doctor, require_supabase_user, require_parent
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
async def me(user=Depends(require_supabase_user), db=Depends(get_db)):
    profile = await db.profile.find_unique(where={"id": user["user_id"]})
    if not profile:
        meta = user["claims"].get("user_metadata", {}) or {}
        role = meta.get("role") or "guest"
        # Doctors should default to pending until admin approves.
        status_val = "pending" if role == "doctor" else "approved"
        profile = await db.profile.create(
            data={
                "id": user["user_id"],
                "email": user["claims"].get("email") or "",
                "fullName": meta.get("full_name") or meta.get("name"),
                "role": role,
                "status": status_val,
            }
        )
    return MeOut(role=profile.role, status=profile.status, email=profile.email, full_name=profile.fullName)

@router.post("/doctors/register")
async def doctor_register(payload: DoctorRegisterIn, user=Depends(require_supabase_user), db=Depends(get_db)):
    # Ensure profile exists (role doctor, status pending)
    prof = await db.profile.upsert(
        where={"id": user["user_id"]},
        data={
            "create": {
                "id": user["user_id"],
                "email": user["claims"].get("email") or "",
                "fullName": user["claims"].get("user_metadata", {}).get("full_name"),
                "role": "doctor",
                "status": "pending",
            },
            "update": {
                "role": "doctor",
                "status": "pending",
            },
        },
    )
    await db.doctor.upsert(
        where={"profileId": prof.id},
        data={
            "create": {
                "profileId": prof.id,
                "licenseId": payload.license_id,
                "idImagePath": payload.id_image_path,
                "approvedAt": None,
            },
            "update": {
                "licenseId": payload.license_id,
                "idImagePath": payload.id_image_path,
            },
        },
    )
    return {"submitted": True}


@router.get("/admin/doctor-requests")
async def admin_list_doctor_requests(admin=Depends(require_admin), db=Depends(get_db)):
    pending = await db.profile.find_many(
        where={"role": "doctor", "status": "pending"},
        include={"doctor": True},
        order={"createdAt": "desc"},
    )
    return pending


@router.post("/admin/doctors/{doctor_profile_id}/approve")
async def admin_approve_doctor(doctor_profile_id: str, admin=Depends(require_admin), db=Depends(get_db)):
    prof = await db.profile.find_unique(where={"id": doctor_profile_id})
    if not prof or prof.role != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found")
    await db.profile.update(where={"id": doctor_profile_id}, data={"status": "approved"})
    await db.doctor.update(where={"profileId": doctor_profile_id}, data={"approvedAt": datetime.utcnow()})
    return {"approved": True}


@router.post("/admin/doctors/{doctor_profile_id}/reject")
async def admin_reject_doctor(doctor_profile_id: str, admin=Depends(require_admin), db=Depends(get_db)):
    prof = await db.profile.find_unique(where={"id": doctor_profile_id})
    if not prof or prof.role != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found")
    await db.profile.update(where={"id": doctor_profile_id}, data={"status": "rejected"})
    return {"rejected": True}


def _gen_patient_id():
    alphabet = string.ascii_uppercase + string.digits
    return "PT-" + "".join(secrets.choice(alphabet) for _ in range(6))


def _gen_password():
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(10))


@router.post("/doctor/patients", response_model=PatientCreateOut)
async def doctor_create_patient(payload: PatientCreateIn, doctor=Depends(require_doctor), db=Depends(get_db)):
    patient_id = payload.patient_id or _gen_patient_id()
    password = payload.password or _gen_password()

    exists = await db.patient.find_unique(where={"patientId": patient_id})
    if exists:
        raise HTTPException(status_code=409, detail="Patient ID already exists")

    patient = await db.patient.create(
        data={
            "patientId": patient_id,
            "doctorId": doctor.id,
            "parentName": payload.parent_name,
            "childName": payload.child_name,
            "contactEmail": str(payload.contact_email),
            "passwordHash": pwd.hash(password),
        }
    )
    return PatientCreateOut(patient_id=patient.patientId, password=password)


@router.get("/doctor/patients")
async def doctor_list_patients(doctor=Depends(require_doctor), db=Depends(get_db)):
    return await db.patient.find_many(where={"doctorId": doctor.id}, order={"createdAt": "desc"})


@router.post("/auth/patient/login", response_model=TokenOut)
async def patient_login(payload: PatientLoginIn, db=Depends(get_db)):
    patient = await db.patient.find_unique(where={"patientId": payload.patient_id})
    if not patient or not pwd.verify(payload.password, patient.passwordHash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_patient_token(sub=patient.id, patient_id=patient.patientId)
    return TokenOut(access_token=token)


@router.get("/parent/me")
async def parent_me(patient=Depends(require_parent)):
    return {
        "patient_id": patient.patientId,
        "parent_name": patient.parentName,
        "child_name": patient.childName,
        "contact_email": patient.contactEmail,
    }
