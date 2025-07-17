from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.payment import Payment
from schemas.payment import PaymentCreate
import uuid


async def create_payment(db: AsyncSession, payment: PaymentCreate):
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
    await db.commit()
    await db.refresh(db_payment)
    return db_payment


async def get_payment(db: AsyncSession, payment_id: str):
    result = await db.execute(select(Payment).filter(Payment.id == payment_id))
    return result.scalar_one_or_none()


async def get_payments_by_loan(db: AsyncSession, loan_id: str, skip: int = 0, limit: int = 100):
    result = await db.execute(
        select(Payment)
        .filter(Payment.loan_id == loan_id)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def get_all_payments(db: AsyncSession, skip: int = 0, limit: int = 100):
    result = await db.execute(select(Payment).offset(skip).limit(limit))
    return result.scalars().all()


async def update_payment(db: AsyncSession, payment_id: str, payment_data: dict):
    result = await db.execute(select(Payment).filter(Payment.id == payment_id))
    payment = result.scalar_one_or_none()
    if payment:
        for key, value in payment_data.items():
            setattr(payment, key, value)
        await db.commit()
        await db.refresh(payment)
    return payment


async def delete_payment(db: AsyncSession, payment_id: str):
    result = await db.execute(select(Payment).filter(Payment.id == payment_id))
    payment = result.scalar_one_or_none()
    if payment:
        await db.delete(payment)
        await db.commit()
        return True
    return False
