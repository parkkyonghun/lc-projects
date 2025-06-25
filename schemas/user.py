from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List
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
    id: str  # Accept any string (not UUID)
    email: EmailStr
    name: str
    password: str
    role: str
    branchId: Optional[str] = None  # Accept any string (not UUID)
    username: Optional[str] = None
    image: Optional[str] = None
    preferences: Optional[Dict] = None
    isActive: Optional[bool] = True
    lastLogin: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    failedLoginAttempts: Optional[int] = 0
    lockedUntil: Optional[datetime] = None
    branch: Optional[BranchSchema] = None
    loans: List['LoanSchema'] = []

    class Config:
        from_attributes = True
