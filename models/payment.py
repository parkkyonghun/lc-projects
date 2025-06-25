from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa
import enum

Base = declarative_base()


class PaymentMethod(enum.Enum):
    cash = "cash"
    bank_transfer = "bank_transfer"
    mobile_payment = "mobile_payment"
    check = "check"


class Payment(Base):
    __tablename__ = "Payment"
    id = Column(String, primary_key=True)
    loan_id = Column(String, ForeignKey("Loan.id"), nullable=False)
    amount = Column(Float, nullable=False)
    payment_date = Column(DateTime, nullable=False)
    payment_method = Column(Enum(PaymentMethod), nullable=False)
    receipt_number = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    late_fee = Column(Float, nullable=True, default=0)
    createdAt = Column(DateTime, server_default=sa.func.now())
    updatedAt = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())

    loan = relationship("Loan", back_populates="payments")
