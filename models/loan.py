from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa

Base = declarative_base()

class Loan(Base):
    __tablename__ = "Loan"
    id = Column(String, primary_key=True)
    userId = Column(String, ForeignKey('User.id'), nullable=False)
    branchId = Column(String, ForeignKey('Branch.id'), nullable=False)
    amount = Column(Float, nullable=False)
    interestRate = Column(Float, nullable=False)
    term = Column(Integer, nullable=False) # in months
    status = Column(String, nullable=False, default='pending') # pending, approved, rejected, paid
    applicationDate = Column(DateTime, server_default=sa.func.now())
    approvalDate = Column(DateTime, nullable=True)
    repaymentSchedule = Column(JSON, nullable=True)
    createdAt = Column(DateTime, server_default=sa.func.now())
    updatedAt = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())

    user = relationship("User", back_populates="loans")