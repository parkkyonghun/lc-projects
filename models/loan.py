<<<<<<< HEAD
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa
from enum import Enum
=======
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum, event, func
from sqlalchemy.orm import relationship, validates
from .base import Base
from .user import UserType
>>>>>>> 206bdf9ddbe6e66e57ff16327692889b7c787595


class LoanStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEFAULTED = "defaulted"


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
    __tablename__ = "loans"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False, index=True)
    loan_officer_id = Column(String, ForeignKey("loan_officers.id"), nullable=True, index=True)
    branch_id = Column(String, ForeignKey("branches.id"), nullable=False, index=True)
    
    # Loan details
    loan_amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    term_months = Column(Integer, nullable=False)
    monthly_payment = Column(Float, nullable=True)
    purpose = Column(String, nullable=True)
<<<<<<< HEAD
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
=======
    
    # Status and dates
    status = Column(SQLEnum(LoanStatus), default=LoanStatus.PENDING, nullable=False)
    application_date = Column(DateTime, server_default=func.now())
    approval_date = Column(DateTime, nullable=True)
    start_date = Column(DateTime, nullable=True)
    next_payment_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Financial tracking
    remaining_balance = Column(Float, nullable=True)
    total_paid = Column(Float, default=0.0)
    
    # Additional details
    collateral_description = Column(String, nullable=True)
    collateral_value = Column(Float, nullable=True)
    repayment_schedule = Column(JSON, nullable=True)
    notes = Column(String, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="loans")
    loan_officer = relationship("LoanOfficer", back_populates="loans")
    branch = relationship("Branch", back_populates="loans")
    payments = relationship("Payment", back_populates="loan", cascade="all, delete-orphan")
    
    def calculate_monthly_payment(self) -> float:
        """Calculate the monthly payment using the loan amount, interest rate, and term."""
        if self.interest_rate == 0:
            return self.loan_amount / self.term_months
            
        monthly_rate = self.interest_rate / 100 / 12
        return (self.loan_amount * monthly_rate * (1 + monthly_rate) ** self.term_months) / \
               ((1 + monthly_rate) ** self.term_months - 1)
    
    # Validations
    @validates('loan_amount', 'interest_rate', 'term_months')
    def validate_loan_terms(self, key: str, value: Any) -> Any:
        if key == 'loan_amount' and value <= 0:
            raise ValueError("Loan amount must be greater than 0")
        if key == 'interest_rate' and (value < 0 or value > 100):
            raise ValueError("Interest rate must be between 0 and 100")
        if key == 'term_months' and value <= 0:
            raise ValueError("Loan term must be at least 1 month")
        return value
    
    def update_remaining_balance(self, payment_amount: Decimal) -> Dict[str, Any]:
        """
        Update the remaining balance after a payment.
        
        Returns:
            Dict containing payment allocation details
        """
        if not isinstance(payment_amount, Decimal):
            payment_amount = Decimal(str(payment_amount))
            
        if self.remaining_balance is None:
            self.remaining_balance = Decimal(str(self.loan_amount))
        else:
            self.remaining_balance = Decimal(str(self.remaining_balance))
            
        payment_details = {
            'principal_paid': Decimal('0'),
            'interest_paid': Decimal('0'),
            'fees_paid': Decimal('0'),
            'remaining_balance': self.remaining_balance
        }
        
        # First, allocate to fees if any
        # (implementation depends on your fee structure)
        
        # Then to interest
        monthly_interest = (self.remaining_balance * Decimal(str(self.interest_rate))) / Decimal('1200')
        interest_paid = min(monthly_interest, payment_amount)
        payment_amount -= interest_paid
        payment_details['interest_paid'] = interest_paid
        
        # Then to principal
        principal_paid = min(payment_amount, self.remaining_balance)
        self.remaining_balance -= principal_paid
        payment_amount -= principal_paid
        payment_details['principal_paid'] = principal_paid
        
        # Update total paid
        self.total_paid = float(Decimal(str(self.total_paid or 0)) + principal_paid + interest_paid)
        
        # Update status if paid off
        if self.remaining_balance <= 0:
            self.remaining_balance = Decimal('0')
            self.status = LoanStatus.COMPLETED
            self.end_date = datetime.utcnow()
        
        payment_details['remaining_balance'] = self.remaining_balance
        return payment_details
    
    def generate_repayment_schedule(self) -> List[Dict[str, Any]]:
        """Generate an amortization schedule for the loan."""
        if not all([self.loan_amount, self.interest_rate, self.term_months]):
            raise ValueError("Loan amount, interest rate, and term are required")
            
        schedule = []
        monthly_payment = self.calculate_monthly_payment()
        balance = Decimal(str(self.loan_amount))
        rate = Decimal(str(self.interest_rate)) / Decimal('1200')
        
        for month in range(1, self.term_months + 1):
            interest_payment = balance * rate
            principal_payment = Decimal(str(monthly_payment)) - interest_payment
            
            # Adjust final payment if needed
            if month == self.term_months:
                principal_payment = balance
                monthly_payment = principal_payment + interest_payment
            
            balance -= principal_payment
            
            schedule.append({
                'month': month,
                'payment_date': (self.start_date or datetime.utcnow()) + timedelta(days=30*month),
                'payment_amount': float(monthly_payment),
                'principal': float(principal_payment),
                'interest': float(interest_payment),
                'remaining_balance': float(max(balance, 0))
            })
            
            if balance <= 0:
                break
                
        return schedule
    
    def get_next_payment_due(self) -> Optional[datetime]:
        """Get the next payment due date."""
        if not self.start_date or not self.next_payment_date:
            return None
            
        now = datetime.utcnow()
        if now > self.next_payment_date:
            # Find the next payment date after today
            months_since_last = ((now.year - self.next_payment_date.year) * 12 + 
                               now.month - self.next_payment_date.month)
            return self.next_payment_date + timedelta(days=30 * (months_since_last + 1))
        return self.next_payment_date
    
    def get_total_interest(self) -> float:
        """Calculate total interest to be paid over the life of the loan."""
        if not all([self.loan_amount, self.interest_rate, self.term_months]):
            return 0.0
            
        monthly_payment = self.calculate_monthly_payment()
        return (monthly_payment * self.term_months) - self.loan_amount
>>>>>>> 206bdf9ddbe6e66e57ff16327692889b7c787595


@event.listens_for(Loan, 'before_insert')
def set_initial_values(mapper, connection, target):
    """Set initial values before a new loan is inserted."""
    if target.remaining_balance is None:
        target.remaining_balance = target.loan_amount
    if target.status is None:
        target.status = LoanStatus.PENDING
    if target.application_date is None:
        target.application_date = datetime.utcnow()
    if target.monthly_payment is None:
        target.monthly_payment = target.calculate_monthly_payment()
