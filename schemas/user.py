from pydantic import BaseModel
from typing import Optional, Dict
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
    username: str
    email: str
    phone: str
    firstName: str
    lastName: str
    role: str
    isActive: bool
    isVerified: bool
    branchId: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    lastLogin: Optional[datetime] = None
    profileData: Optional[Dict] = None
    
    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    username: str
    email: str
    phone: str
    password: str
    firstName: str
    lastName: str
    role: str = "customer"
    branchId: Optional[str] = None
    profileData: Optional[Dict] = None


class UserUpdate(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    isActive: Optional[bool] = None
    isVerified: Optional[bool] = None
    branchId: Optional[str] = None
    profileData: Optional[Dict] = None
