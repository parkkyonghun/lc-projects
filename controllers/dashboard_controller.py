from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from models.loan import Loan
from models.user import User
from schemas.dashboard import DashboardSchema
from schemas.loan import LoanSchema


async def get_dashboard_stats_controller(db: AsyncSession) -> DashboardSchema:
    total_loans_result = await db.execute(select(func.count(Loan.id)))
    total_loans = total_loans_result.scalar_one()

    active_loans_result = await db.execute(
        select(func.count(Loan.id)).where(Loan.status == "active")
    )
    active_loans = active_loans_result.scalar_one()

    total_customers_result = await db.execute(select(func.count(User.id)))
    total_customers = total_customers_result.scalar_one()

    total_amount_result = await db.execute(select(func.sum(Loan.loanAmount)))
    total_amount = total_amount_result.scalar_one() or 0.0

    recent_loans_result = await db.execute(
        select(Loan)
        .options(selectinload(Loan.user))
        .order_by(Loan.applicationDate.desc())
        .limit(5)
    )
    recent_loans = recent_loans_result.scalars().all()

    payment_alerts_result = await db.execute(
        select(Loan).options(selectinload(Loan.user)).where(Loan.status == "pending")
    )
    payment_alerts = payment_alerts_result.scalars().all()

    recent_loans_schemas = [LoanSchema.model_validate(loan) for loan in recent_loans]
    payment_alerts_schemas = [LoanSchema.model_validate(alert) for alert in payment_alerts]

    return DashboardSchema(
        totalLoans=total_loans,
        activeLoans=active_loans,
        totalCustomers=total_customers,
        totalAmount=total_amount,
        recentLoans=recent_loans_schemas,
        paymentAlerts=payment_alerts_schemas,
    )
