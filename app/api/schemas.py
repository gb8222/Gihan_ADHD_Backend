from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any, Dict, List


class MeOut(BaseModel):
    role: str
    status: str
    email: str
    full_name: Optional[str] = None


class DoctorRegisterIn(BaseModel):
    license_id: str = Field(min_length=3, max_length=64)
    id_image_path: str


class PatientCreateIn(BaseModel):
    parent_name: str
    child_name: str
    contact_email: EmailStr
    password: Optional[str] = None  # if None, backend generates
    patient_id: Optional[str] = None  # if None, backend generates


class PatientCreateOut(BaseModel):
    patient_id: str
    password: str


class PatientLoginIn(BaseModel):
    patient_id: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
