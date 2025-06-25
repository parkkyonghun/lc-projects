from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from controllers import payment_controller
from schemas.payment import PaymentSchema, PaymentCreate
from views.dependencies import get_db

router = APIRouter(
    prefix="/payments",
    tags=["payments"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=PaymentSchema)
def create_payment(payment: PaymentCreate, db: Session = Depends(get_db)):
    return payment_controller.create_payment(db=db, payment=payment)


@router.get("/loan/{loan_id}", response_model=List[PaymentSchema])
def read_payments_by_loan(
    loan_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    payments = payment_controller.get_payments_by_loan(
        db, loan_id=loan_id, skip=skip, limit=limit
    )
    return payments


@router.get("/{payment_id}", response_model=PaymentSchema)
def read_payment(payment_id: str, db: Session = Depends(get_db)):
    db_payment = payment_controller.get_payment(db, payment_id=payment_id)
    if db_payment is None:
        raise HTTPException(status_code=404, detail="Payment not found")
    return db_payment


@router.get("/", response_model=List[PaymentSchema])
def read_all_payments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    payments = payment_controller.get_all_payments(db, skip=skip, limit=limit)
    return payments
