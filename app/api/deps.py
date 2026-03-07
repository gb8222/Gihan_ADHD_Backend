from fastapi import Depends, Header, HTTPException, status

from app.core.security import decode_supabase_jwt, decode_patient_token
from app.db.client import db


async def get_db():
    if not db.is_connected():
        await db.connect()
    return db


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


async def require_supabase_user(authorization: str | None = Header(default=None)):
    token = _bearer(authorization)
    payload = decode_supabase_jwt(token)
    # Supabase uses 'sub' as user id (uuid)
    return {"token": token, "user_id": payload.get("sub"), "claims": payload}


async def require_admin(user=Depends(require_supabase_user), db=Depends(get_db)):
    profile = await db.profile.find_unique(where={"id": user["user_id"]})
    if not profile or profile.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return profile


async def require_doctor(user=Depends(require_supabase_user), db=Depends(get_db)):
    profile = await db.profile.find_unique(where={"id": user["user_id"]})
    if not profile or profile.role != "doctor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor only")
    if profile.status != "approved":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor not approved yet")
    return profile


async def require_guest(user=Depends(require_supabase_user), db=Depends(get_db)):
    profile = await db.profile.find_unique(where={"id": user["user_id"]})
    if not profile or profile.role != "guest":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Guest only")
    return profile


async def require_parent(authorization: str | None = Header(default=None), db=Depends(get_db)):
    token = _bearer(authorization)
    payload = decode_patient_token(token)
    patient_id = payload.get("patient_id")
    if not patient_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid patient token")
    patient = await db.patient.find_unique(where={"patientId": patient_id})
    if not patient:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Patient not found")
    return patient
