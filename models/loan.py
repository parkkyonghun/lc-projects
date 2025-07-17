from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa
from enum import Enum

Base = declarative_base()


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class LoanStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"


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
    
    # Sync-related fields
    server_id = Column(String, nullable=True, unique=True)  # ID from server
    sync_status = Column(String, nullable=False, default=SyncStatus.PENDING)
    last_synced_at = Column(DateTime, nullable=True)
    version = Column(Integer, nullable=False, default=1)  # For conflict resolution
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete
    sync_retry_count = Column(Integer, nullable=False, default=0)
    sync_error_message = Column(String, nullable=True)

    user = relationship("User", back_populates="loans")
