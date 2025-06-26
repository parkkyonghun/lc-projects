from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from models.payment import PaymentMethod


class PaymentSchema(BaseModel):
    id: str
    loan_id: str
    amount: float
    payment_date: datetime
    payment_method: PaymentMethod
    receipt_number: Optional[str] = None
    notes: Optional[str] = None
    late_fee: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}


class PaymentCreate(BaseModel):
    loan_id: str
    amount: float
    payment_method: PaymentMethod
    receipt_number: Optional[str] = None
    notes: Optional[str] = None
    late_fee: Optional[float] = None


class PaymentUpdate(BaseModel):
    amount: Optional[float] = None
    payment_method: Optional[PaymentMethod] = None
    receipt_number: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
