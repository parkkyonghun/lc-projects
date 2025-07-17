from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from controllers import payment_controller
from models.user import User
from schemas.payment import PaymentSchema, PaymentCreate
from services.application_repository import ApplicationRepository
from services.websocket_manager import notification_manager
from .dependencies import get_db, get_current_user, get_repository

router = APIRouter(
    prefix="/payments",
    tags=["payments"],
    responses={404: {"description": "Not found"}},
)


class PaymentUpdateRequest(BaseModel):
    amount: Optional[float] = None
    payment_date: Optional[str] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None


@router.post("/", response_model=PaymentSchema)
async def create_payment(
    payment: PaymentCreate, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Create a new payment with automatic sync queuing"""
    # Note: This would need to be implemented in the repository
    # For now, using the existing controller with async session
    db = await repository.get_session()
    payment_obj = await payment_controller.create_payment(db=db, payment=payment)
    
    # Send real-time notification
    await notification_manager.send_notification(
        user_id=current_user.id,
        notification_type="payment_created",
        message=f"Payment of ${payment.amount} created for loan {payment.loan_id}",
        data={
            "payment_id": payment_obj.id,
            "loan_id": payment.loan_id,
            "amount": payment.amount
        }
    )
    
    return payment_obj


@router.get("/loan/{loan_id}", response_model=List[PaymentSchema])
async def read_payments_by_loan(
    loan_id: str, 
    skip: int = 0, 
    limit: int = 100, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all payments for a specific loan"""
    db = await repository.get_session()
    payments = await payment_controller.get_payments_by_loan(
        db, loan_id=loan_id, skip=skip, limit=limit
    )
    return payments


@router.get("/{payment_id}", response_model=PaymentSchema)
async def read_payment(
    payment_id: str, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get a specific payment by ID"""
    db = await repository.get_session()
    db_payment = await payment_controller.get_payment(db, payment_id=payment_id)
    if db_payment is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return db_payment


@router.get("/", response_model=List[PaymentSchema])
async def read_all_payments(
    skip: int = 0, 
    limit: int = 100, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all payments with pagination"""
    db = await repository.get_session()
    payments = await payment_controller.get_all_payments(db, skip=skip, limit=limit)
    return payments


@router.put("/{payment_id}", response_model=PaymentSchema)
async def update_payment(
    payment_id: str,
    payment_data: PaymentUpdateRequest,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Update a payment"""
    db = await repository.get_session()
    
    # Check if payment exists
    existing_payment = await payment_controller.get_payment(db, payment_id=payment_id)
    if not existing_payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    # Update payment (this would need to be implemented in the controller)
    update_data = {k: v for k, v in payment_data.dict().items() if v is not None}
    updated_payment = await payment_controller.update_payment(db, payment_id, update_data)
    
    # Send real-time notification
    await notification_manager.send_notification(
        user_id=current_user.id,
        notification_type="payment_updated",
        message=f"Payment {payment_id} has been updated",
        data={
            "payment_id": payment_id,
            "loan_id": updated_payment.loan_id
        }
    )
    
    return updated_payment


@router.delete("/{payment_id}")
async def delete_payment(
    payment_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Delete a payment"""
    db = await repository.get_session()
    
    # Check if payment exists
    existing_payment = await payment_controller.get_payment(db, payment_id=payment_id)
    if not existing_payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    # Delete payment (this would need to be implemented in the controller)
    success = await payment_controller.delete_payment(db, payment_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete payment")
    
    return {"message": "Payment deleted successfully"}
