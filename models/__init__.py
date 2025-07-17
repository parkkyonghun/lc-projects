from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, relationship, configure_mappers
import os
from typing import AsyncGenerator

# Import all models to ensure they are registered with SQLAlchemy
from .base import Base
from .user import User, Customer, LoanOfficer, UserType
from .loan import Loan, LoanStatus
from .branch import Branch
from .payment import Payment, PaymentMethod, PaymentStatus

# Configure relationships after all models are imported
def configure_relationships():
    """Configure all model relationships after imports."""
    # User relationships
    User.customer = relationship("Customer", back_populates="user", uselist=False, foreign_keys="[Customer.id]")
    User.loan_officer = relationship("LoanOfficer", back_populates="user", uselist=False, foreign_keys="[LoanOfficer.id]")
    
    # Customer relationships
    Customer.loans = relationship("Loan", back_populates="customer", foreign_keys="[Loan.customer_id]")
    
    # LoanOfficer relationships
    LoanOfficer.loans = relationship("Loan", back_populates="loan_officer", foreign_keys="[Loan.loan_officer_id]")
    
    # Call configure_mappers to ensure all relationships are properly set up
    configure_mappers()

# Configure relationships
configure_relationships()

# Create async session factory

# Create async session factory
SessionLocal = sessionmaker(
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
)

# Function to create database engine
async def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        await db.close()
