from pydantic import BaseModel
from typing import Optional, Dict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .user import UserSchema


class LoanSchema(BaseModel):
    id: str
    customerId: str
    branchId: str
    loanAmount: float
    interestRate: float
    termMonths: int
    monthlyPayment: Optional[float] = None
    purpose: Optional[str] = None
    status: str
    applicationDate: datetime
    startDate: Optional[datetime] = None
    nextPaymentDate: Optional[datetime] = None
    remainingBalance: Optional[float] = None
    collateralDescription: Optional[str] = None
    repaymentSchedule: Optional[Dict] = None
    createdAt: datetime
    updatedAt: datetime
    user: Optional["UserSchema"] = None
    
    model_config = {"from_attributes": True}


class LoanCreate(BaseModel):
    customerId: str
    branchId: str
    loanAmount: float
    interestRate: float
    termMonths: int
    purpose: str
    collateralDescription: Optional[str] = None


class LoanUpdate(BaseModel):
    status: Optional[str] = None
    interestRate: Optional[float] = None
    nextPaymentDate: Optional[datetime] = None
    remainingBalance: Optional[float] = None
