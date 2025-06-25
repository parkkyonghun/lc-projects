from sqlalchemy.orm import Session
from models.payment import Payment
from schemas.payment import PaymentCreate
import uuid


def create_payment(db: Session, payment: PaymentCreate):
    db_payment = Payment(
        id=str(uuid.uuid4()),
        loan_id=payment.loan_id,
        amount=payment.amount,
        payment_date=payment.payment_date,
        payment_method=payment.payment_method,
        receipt_number=payment.receipt_number,
        notes=payment.notes,
        late_fee=payment.late_fee,
    )
    db.add(db_payment)
    db.commit()
    db.refresh(db_payment)
    return db_payment


def get_payment(db: Session, payment_id: str):
    return db.query(Payment).filter(Payment.id == payment_id).first()


def get_payments_by_loan(db: Session, loan_id: str, skip: int = 0, limit: int = 100):
    return (
        db.query(Payment)
        .filter(Payment.loan_id == loan_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_all_payments(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Payment).offset(skip).limit(limit).all()
