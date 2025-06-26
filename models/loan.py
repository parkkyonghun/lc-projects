from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa

Base = declarative_base()


class Loan(Base):
    __tablename__ = "Loan"
    id = Column(String, primary_key=True)
    customerId = Column(String, ForeignKey("User.id"), nullable=False)
    branchId = Column(String, ForeignKey("Branch.id"), nullable=False)
    loanAmount = Column(Float, nullable=False)
    interestRate = Column(Float, nullable=False)
    termMonths = Column(Integer, nullable=False)
    monthlyPayment = Column(Float, nullable=True)
    purpose = Column(String, nullable=True)
    status = Column(
        String, nullable=False, default="pending"
    )  # pending, approved, active, completed, rejected
    applicationDate = Column(DateTime, server_default=sa.func.now())
    startDate = Column(DateTime, nullable=True)
    nextPaymentDate = Column(DateTime, nullable=True)
    remainingBalance = Column(Float, nullable=True)
    collateralDescription = Column(String, nullable=True)
    repaymentSchedule = Column(JSON, nullable=True)
    createdAt = Column(DateTime, server_default=sa.func.now())
    updatedAt = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())

    user = relationship("User", back_populates="loans")
