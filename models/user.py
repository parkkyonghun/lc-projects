import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import String, Column, Float, Boolean, ForeignKey, Enum as SQLEnum, DateTime, func, Integer
from sqlalchemy.orm import relationship, deferred
from .base import Base


class UserType(str, Enum):
    CUSTOMER = "customer"
    LOAN_OFFICER = "loan_officer"
    ADMIN = "admin"


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class User(Base):
    __tablename__ = "users"
    
    # Common fields for all user types
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_type = Column(SQLEnum(UserType), nullable=False)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    phone_number = Column(String(20), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Branch information
    branch_id = Column(String, ForeignKey('branches.id'), nullable=True, index=True)
    
    # Status and timestamps
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    branch = relationship("Branch", back_populates="users")
    collected_payments = relationship("Payment", back_populates="collector")
    
    # Audit fields
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    customer = relationship("Customer", back_populates="user", uselist=False)
    loan_officer = relationship("LoanOfficer", back_populates="user", uselist=False)
    
    __mapper_args__ = {
        'polymorphic_on': user_type,
        'polymorphic_identity': 'user'
    }


class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(String, ForeignKey('users.id'), primary_key=True)
    khmer_name = Column(String, nullable=False)
    english_name = Column(String, nullable=False)
    id_card_number = Column(String, unique=True, nullable=False)
    address = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    monthly_income = Column(Float, nullable=True)
    id_card_photo_url = Column(String, nullable=True)
    profile_photo_url = Column(String, nullable=True)
    
    # Sync-related fields
    server_id = Column(String, nullable=True, unique=True)  # ID from server
    sync_status = Column(String, nullable=False, default=SyncStatus.PENDING)
    last_synced_at = Column(DateTime, nullable=True)
    version = Column(Integer, nullable=False, default=1)  # For conflict resolution
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete
    sync_retry_count = Column(Integer, nullable=False, default=0)
    sync_error_message = Column(String, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="customer")
    loans = relationship("Loan", back_populates="customer")
    
    __mapper_args__ = {
        'polymorphic_identity': UserType.CUSTOMER
    }

class LoanOfficer(Base):
    __tablename__ = "loan_officers"
    
    id = Column(String, ForeignKey('users.id'), primary_key=True)
    employee_id = Column(String, unique=True, nullable=False)
    hire_date = Column(DateTime, nullable=False)
    department = Column(String, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="loan_officer")
    loans = relationship("Loan", back_populates="loan_officer")
    
    __mapper_args__ = {
        'polymorphic_identity': UserType.LOAN_OFFICER
    }
