from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from models.loan import Loan
from models.user import User
from schemas.loan import LoanSchema, LoanCreate
from typing import List, Optional
import uuid

async def create_loan_controller(loan_data: LoanCreate, db: AsyncSession) -> LoanSchema:
    new_loan = Loan(
        id=str(uuid.uuid4()),
        **loan_data.dict()
    )
    db.add(new_loan)
    await db.commit()
    await db.refresh(new_loan)
    return LoanSchema.from_orm(new_loan)

async def get_loans_controller(db: AsyncSession) -> List[LoanSchema]:
    result = await db.execute(select(Loan).options(selectinload(Loan.user)))
    loans = result.scalars().all()
    return [LoanSchema.from_orm(l) for l in loans]

async def get_loan_controller(loan_id: str, db: AsyncSession) -> Optional[LoanSchema]:
    result = await db.execute(select(Loan).where(Loan.id == loan_id).options(selectinload(Loan.user)))
    loan = result.scalar_one_or_none()
    if loan:
        return LoanSchema.from_orm(loan)
    return None