from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from controllers.loan_controller import (
    create_loan_controller,
    get_loans_controller,
    get_loan_controller,
)
from schemas.loan import LoanSchema, LoanCreate
from .dependencies import get_db, get_current_user

router = APIRouter(prefix="/loans", tags=["loans"])


@router.post("/", response_model=LoanSchema)
async def create_loan(
    loan_data: LoanCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    return await create_loan_controller(loan_data, db)


@router.get("/", response_model=List[LoanSchema])
async def get_loans(
    db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    return await get_loans_controller(db)


@router.get("/{loan_id}", response_model=LoanSchema)
async def get_loan(
    loan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    loan = await get_loan_controller(loan_id, db)
    if not loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    return loan
