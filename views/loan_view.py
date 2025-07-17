from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from controllers.loan_controller import (
    create_loan_controller,
    get_loans_controller,
    get_loan_controller,
)
from models.user import User
from models.loan import Loan, LoanStatus
from schemas.loan import LoanSchema, LoanCreate
from services.application_repository import ApplicationRepository
from services.websocket_manager import notification_manager
from .dependencies import get_db, get_current_user, get_repository

router = APIRouter(prefix="/loans", tags=["loans"])


class LoanUpdateRequest(BaseModel):
    loanAmount: Optional[float] = None
    interestRate: Optional[float] = None
    loanTerm: Optional[int] = None
    loanStatus: Optional[LoanStatus] = None
    monthlyPayment: Optional[float] = None
    totalAmount: Optional[float] = None
    notes: Optional[str] = None


@router.post("/", response_model=LoanSchema)
async def create_loan(
    loan_data: LoanCreate,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user),
):
    """Create a new loan with automatic sync queuing"""
    loan = await repository.create_loan(loan_data.dict())
    
    # Send real-time notification
    await notification_manager.send_new_application_notification(
        user_id=current_user.id,
        application_type="loan",
        application_id=loan.id,
        customer_name=f"Loan #{loan.id}"
    )
    
    return {
        **LoanSchema.model_validate(loan).dict(),
        "sync_status": loan.sync_status.value if loan.sync_status else None,
        "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
    }


@router.get("/", response_model=List[LoanSchema])
async def get_loans(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all loans with sync status information"""
    loans = await repository.get_all_loans()
    return [
        {
            **LoanSchema.model_validate(loan).dict(),
            "sync_status": loan.sync_status.value if loan.sync_status else None,
            "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
        }
        for loan in loans
    ]


@router.get("/{loan_id}", response_model=LoanSchema)
async def get_loan(
    loan_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user),
):
    """Get a specific loan by ID"""
    loan = await repository.get_loan_by_id(loan_id)
    if not loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    
    return {
        **LoanSchema.model_validate(loan).dict(),
        "sync_status": loan.sync_status.value if loan.sync_status else None,
        "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
    }


@router.put("/{loan_id}", response_model=LoanSchema)
async def update_loan(
    loan_id: str,
    loan_data: LoanUpdateRequest,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Update a loan with automatic sync queuing"""
    # Get current loan for status change notification
    current_loan = await repository.get_loan_by_id(loan_id)
    if not current_loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    
    old_status = current_loan.loanStatus
    
    # Update the loan
    update_data = {k: v for k, v in loan_data.dict().items() if v is not None}
    loan = await repository.update_loan(loan_id, update_data)
    
    # Send real-time notification if status changed
    if loan_data.loanStatus and loan_data.loanStatus != old_status:
        await notification_manager.send_loan_status_change_notification(
            user_id=current_user.id,
            loan_id=loan_id,
            old_status=old_status.value if old_status else "unknown",
            new_status=loan_data.loanStatus.value,
            customer_name=f"Loan #{loan_id}"
        )
    
    return {
        **LoanSchema.model_validate(loan).dict(),
        "sync_status": loan.sync_status.value if loan.sync_status else None,
        "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
    }


@router.delete("/{loan_id}")
async def delete_loan(
    loan_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Soft delete a loan with automatic sync queuing"""
    success = await repository.delete_loan(loan_id)
    if not success:
        raise HTTPException(status_code=404, detail="Loan not found")
    
    return {"message": "Loan deleted successfully"}


@router.get("/customer/{customer_id}", response_model=List[LoanSchema])
async def get_loans_by_customer(
    customer_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all loans for a specific customer"""
    loans = await repository.get_loans_by_customer_id(customer_id)
    return [
        {
            **LoanSchema.model_validate(loan).dict(),
            "sync_status": loan.sync_status.value if loan.sync_status else None,
            "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
        }
        for loan in loans
    ]


@router.get("/status/{status}", response_model=List[LoanSchema])
async def get_loans_by_status(
    status: LoanStatus,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all loans with a specific status"""
    loans = await repository.get_loans_by_status(status)
    return [
        {
            **LoanSchema.model_validate(loan).dict(),
            "sync_status": loan.sync_status.value if loan.sync_status else None,
            "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
        }
        for loan in loans
    ]
