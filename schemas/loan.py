from pydantic import BaseModel
from typing import Optional, List, Dict
from .user import UserSchema
from datetime import datetime

class LoanSchema(BaseModel):
    id: str
    userId: str
    branchId: str
    amount: float
    interestRate: float
    term: int
    status: str
    applicationDate: datetime
    approvalDate: Optional[datetime] = None
    repaymentSchedule: Optional[Dict] = None
    createdAt: datetime
    updatedAt: datetime
    user: Optional[UserSchema] = None

    class Config:
        from_attributes = True

class LoanCreate(BaseModel):
    userId: str
    branchId: str
    amount: float
    interestRate: float
    term: int