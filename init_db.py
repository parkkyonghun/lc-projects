#!/usr/bin/env python3
"""
Database initialization script for Khmer Loan Management System

This script creates all database tables and sets up initial data.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config import settings
from core.database import Base
from core.security import hash_password
from models.user import User, UserRole
from models.branch import Branch
from models.loan import Loan, LoanStatus, LoanType
from models.payment import Payment, PaymentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_tables():
    """Create all database tables"""
    logger.info("Creating database tables...")
    
    engine = create_async_engine(settings.database_url)
    
    async with engine.begin() as conn:
        # Drop all tables (for development)
        await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    logger.info("Database tables created successfully")

async def create_initial_data():
    """Create initial data for the system"""
    logger.info("Creating initial data...")
    
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            # Create branches
            branches = [
                Branch(
                    name="សាខាកណ្តាល",  # Central Branch
                    name_en="Central Branch",
                    address="ផ្លូវ២៧១, សង្កាត់ទន្លេបាសាក់, ខណ្ឌចំការមន, រាជធានីភ្នំពេញ",
                    phone="023-123-456",
                    email="central@loanapp.kh",
                    manager_name="លោក សុខ វិចិត្រ",
                    is_active=True
                ),
                Branch(
                    name="សាខាទួលគោក",  # Tuol Kork Branch
                    name_en="Tuol Kork Branch",
                    address="ផ្លូវ៣១៥, សង្កាត់បឹងកក់២, ខណ្ឌទួលគោក, រាជធានីភ្នំពេញ",
                    phone="023-234-567",
                    email="tuolkork@loanapp.kh",
                    manager_name="លោកស្រី ចាន់ សុភាព",
                    is_active=True
                ),
                Branch(
                    name="សាខាសៀមរាប",  # Siem Reap Branch
                    name_en="Siem Reap Branch",
                    address="ផ្លូវជាតិលេខ៦, ក្រុងសៀមរាប, ខេត្តសៀមរាប",
                    phone="063-345-678",
                    email="siemreap@loanapp.kh",
                    manager_name="លោក ពេជ្រ សុវណ្ណ",
                    is_active=True
                )
            ]
            
            for branch in branches:
                session.add(branch)
            
            await session.commit()
            
            # Get branch IDs for user creation
            await session.refresh(branches[0])
            central_branch_id = branches[0].id
            
            # Create admin user
            admin_user = User(
                email="admin@loanapp.kh",
                hashed_password=hash_password("admin123"),
                full_name="អ្នកគ្រប់គ្រងប្រព័ន្ធ",  # System Administrator
                phone="012-345-678",
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow(),
                branch_id=central_branch_id
            )
            session.add(admin_user)
            
            # Create manager user
            manager_user = User(
                email="manager@loanapp.kh",
                hashed_password=hash_password("manager123"),
                full_name="លោក សុខ វិចិត្រ",
                phone="012-456-789",
                role=UserRole.MANAGER,
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow(),
                branch_id=central_branch_id
            )
            session.add(manager_user)
            
            # Create staff user
            staff_user = User(
                email="staff@loanapp.kh",
                hashed_password=hash_password("staff123"),
                full_name="លោកស្រី ចាន់ សុភាព",
                phone="012-567-890",
                role=UserRole.STAFF,
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow(),
                branch_id=central_branch_id
            )
            session.add(staff_user)
            
            # Create sample customer users
            customers = [
                User(
                    email="customer1@example.com",
                    hashed_password=hash_password("customer123"),
                    full_name="លោក ហេង សុវណ្ណ",
                    phone="012-678-901",
                    date_of_birth=datetime(1985, 5, 15),
                    address="ផ្ទះលេខ១២៣, ផ្លូវ២៧១, សង្កាត់ទន្លេបាសាក់, ខណ្ឌចំការមន, រាជធានីភ្នំពេញ",
                    national_id="123456789",
                    role=UserRole.CUSTOMER,
                    is_active=True,
                    is_verified=True,
                    created_at=datetime.utcnow(),
                    branch_id=central_branch_id
                ),
                User(
                    email="customer2@example.com",
                    hashed_password=hash_password("customer123"),
                    full_name="លោកស្រី ស៊ីម ចន្ទ្រា",
                    phone="012-789-012",
                    date_of_birth=datetime(1990, 8, 22),
                    address="ផ្ទះលេខ៤៥៦, ផ្លូវ៣១៥, សង្កាត់បឹងកក់២, ខណ្ឌទួលគោក, រាជធានីភ្នំពេញ",
                    national_id="987654321",
                    role=UserRole.CUSTOMER,
                    is_active=True,
                    is_verified=True,
                    created_at=datetime.utcnow(),
                    branch_id=central_branch_id
                )
            ]
            
            for customer in customers:
                session.add(customer)
            
            await session.commit()
            
            # Get customer IDs for loan creation
            await session.refresh(customers[0])
            await session.refresh(customers[1])
            
            # Create sample loans
            loans = [
                Loan(
                    user_id=customers[0].id,
                    amount=Decimal('5000000'),  # 5M KHR
                    interest_rate=Decimal('0.15'),  # 15%
                    term_months=24,
                    monthly_payment=Decimal('254000'),
                    loan_type=LoanType.PERSONAL,
                    purpose="ការិយាល័យផ្ទាល់ខ្លួន",  # Personal business
                    status=LoanStatus.ACTIVE,
                    application_date=datetime.utcnow(),
                    approved_date=datetime.utcnow(),
                    disbursement_date=datetime.utcnow(),
                    first_payment_date=datetime.utcnow(),
                    employment_info="ម្ចាស់ហាងលក់គ្រឿងសម្អាង",  # Cosmetics shop owner
                    monthly_income=Decimal('800000'),  # 800K KHR
                    branch_id=central_branch_id
                ),
                Loan(
                    user_id=customers[1].id,
                    amount=Decimal('10000000'),  # 10M KHR
                    interest_rate=Decimal('0.12'),  # 12%
                    term_months=36,
                    monthly_payment=Decimal('332000'),
                    loan_type=LoanType.BUSINESS,
                    purpose="ពង្រីកអាជីវកម្ម",  # Business expansion
                    status=LoanStatus.PENDING,
                    application_date=datetime.utcnow(),
                    employment_info="ម្ចាស់ហាងលក់អាហារ",  # Restaurant owner
                    monthly_income=Decimal('1200000'),  # 1.2M KHR
                    branch_id=central_branch_id
                )
            ]
            
            for loan in loans:
                session.add(loan)
            
            await session.commit()
            
            # Create sample payments for the active loan
            await session.refresh(loans[0])
            
            payments = [
                Payment(
                    loan_id=loans[0].id,
                    user_id=customers[0].id,
                    amount=Decimal('254000'),
                    due_date=datetime.utcnow(),
                    payment_date=datetime.utcnow(),
                    status=PaymentStatus.COMPLETED,
                    payment_method="ធនាគារ",  # Bank
                    payment_number=1,
                    branch_id=central_branch_id
                )
            ]
            
            for payment in payments:
                session.add(payment)
            
            await session.commit()
            
            logger.info("Initial data created successfully")
            
        except Exception as e:
            logger.error(f"Error creating initial data: {str(e)}")
            await session.rollback()
            raise
        
        finally:
            await session.close()
    
    await engine.dispose()

async def main():
    """Main initialization function"""
    try:
        logger.info("Starting database initialization...")
        
        # Create tables
        await create_tables()
        
        # Create initial data
        await create_initial_data()
        
        logger.info("Database initialization completed successfully!")
        
        # Print initial login credentials
        print("\n" + "="*50)
        print("DATABASE INITIALIZATION COMPLETED")
        print("="*50)
        print("\nInitial Login Credentials:")
        print("-" * 30)
        print("Admin User:")
        print("  Email: admin@loanapp.kh")
        print("  Password: admin123")
        print("\nManager User:")
        print("  Email: manager@loanapp.kh")
        print("  Password: manager123")
        print("\nStaff User:")
        print("  Email: staff@loanapp.kh")
        print("  Password: staff123")
        print("\nSample Customers:")
        print("  Email: customer1@example.com")
        print("  Password: customer123")
        print("  Email: customer2@example.com")
        print("  Password: customer123")
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())