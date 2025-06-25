from pydantic import BaseModel
from typing import Optional, List

from datetime import datetime


class BranchSchema(BaseModel):
    id: str
    code: str
    name: str
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
    parentId: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserSchema(BaseModel):
    id: str
    khmer_name: str
    english_name: str
    id_card_number: str
    phone_number: str
    address: Optional[str] = None
    occupation: Optional[str] = None
    monthly_income: Optional[float] = None
    id_card_photo_url: Optional[str] = None
    profile_photo_url: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    loans: List["LoanSchema"] = []

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    khmer_name: str
    english_name: str
    id_card_number: str
    phone_number: str
    password: str
    address: Optional[str] = None
    occupation: Optional[str] = None
    monthly_income: Optional[float] = None
    id_card_photo_url: Optional[str] = None
    profile_photo_url: Optional[str] = None
