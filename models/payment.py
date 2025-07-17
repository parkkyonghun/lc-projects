from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum as SQLEnum, Numeric, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ENUM
from .base import Base


class PaymentMethod(PyEnum):
    CASH = "cash"
    BANK_TRANSFER = "bank_transfer"
    MOBILE_PAYMENT = "mobile_payment"
    CHECK = "check"
    CREDIT_CARD = "credit_card"
    WALLET = "wallet"


class PaymentStatus(PyEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    CANCELLED = "cancelled"


class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    loan_id = Column(String, ForeignKey("loans.id"), nullable=False, index=True)
    collected_by = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    # Payment details
    amount = Column(Numeric(12, 2), nullable=False)
    payment_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    payment_method = Column(ENUM(PaymentMethod, name="payment_method_enum"), nullable=False)
    status = Column(ENUM(PaymentStatus, name="payment_status_enum"), default=PaymentStatus.COMPLETED, nullable=False)
    
    # Reference and receipt
    reference_number = Column(String(50), unique=True, nullable=True)
    receipt_number = Column(String(50), unique=True, nullable=True)
    transaction_id = Column(String(100), nullable=True)
    
    # Financial details
    principal_amount = Column(Numeric(12, 2), nullable=False)
    interest_amount = Column(Numeric(12, 2), nullable=False)
    late_fee = Column(Numeric(12, 2), default=0.0)
    other_fees = Column(Numeric(12, 2), default=0.0)
    
    # Additional information
    notes = Column(Text, nullable=True)
    attachment_url = Column(String(255), nullable=True)
    
    # Relationships
    loan = relationship("Loan", back_populates="payments")
    collector = relationship("User", back_populates="collected_payments")
    
    def __repr__(self) -> str:
        return f"<Payment {self.reference_number or self.id}: {self.amount} {self.payment_method}>"
