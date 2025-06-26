from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
from datetime import datetime, timedelta
import logging
from decimal import Decimal

from models.loan import Loan, LoanStatus, LoanType
from models.user import User
from models.payment import Payment, PaymentStatus
from schemas.loan import LoanCreate, LoanUpdate, LoanResponse
from services.notification_service import notification_service
from core.config import settings

logger = logging.getLogger(__name__)

class LoanService:
    """Service for handling loan operations"""
    
    def __init__(self):
        self.max_loan_amount = Decimal('50000000')  # 50M KHR
        self.min_loan_amount = Decimal('1000000')   # 1M KHR
        self.default_interest_rate = Decimal('0.15')  # 15% annual
        self.max_loan_term_months = 60  # 5 years
    
    async def create_loan_application(
        self, 
        db: AsyncSession, 
        loan_data: LoanCreate, 
        user_id: str
    ) -> Loan:
        """Create a new loan application"""
        
        # Validate loan amount
        if loan_data.amount < self.min_loan_amount or loan_data.amount > self.max_loan_amount:
            raise ValueError(
                f"Loan amount must be between {self.min_loan_amount:,} and {self.max_loan_amount:,} KHR"
            )
        
        # Validate loan term
        if loan_data.term_months > self.max_loan_term_months:
            raise ValueError(f"Maximum loan term is {self.max_loan_term_months} months")
        
        # Check if user has pending applications
        existing_pending = await self._get_pending_loans(db, user_id)
        if existing_pending:
            raise ValueError("You already have a pending loan application")
        
        # Calculate interest rate based on loan type and amount
        interest_rate = await self._calculate_interest_rate(loan_data.loan_type, loan_data.amount)
        
        # Calculate monthly payment
        monthly_payment = self._calculate_monthly_payment(
            loan_data.amount, 
            interest_rate, 
            loan_data.term_months
        )
        
        # Create loan record
        loan = Loan(
            user_id=user_id,
            amount=loan_data.amount,
            interest_rate=interest_rate,
            term_months=loan_data.term_months,
            monthly_payment=monthly_payment,
            loan_type=loan_data.loan_type,
            purpose=loan_data.purpose,
            status=LoanStatus.PENDING,
            application_date=datetime.utcnow(),
            collateral_description=loan_data.collateral_description,
            employment_info=loan_data.employment_info,
            monthly_income=loan_data.monthly_income,
            existing_debts=loan_data.existing_debts or Decimal('0')
        )
        
        db.add(loan)
        await db.commit()
        await db.refresh(loan)
        
        # Send notification
        try:
            user = await db.get(User, user_id)
            if user and user.email:
                await notification_service.send_loan_application_confirmation(
                    user.email, 
                    user.full_name, 
                    loan.id,
                    loan.amount
                )
        except Exception as e:
            logger.error(f"Failed to send loan application notification: {str(e)}")
        
        logger.info(f"Loan application created: {loan.id} for user {user_id}")
        return loan
    
    async def review_loan_application(
        self, 
        db: AsyncSession, 
        loan_id: str, 
        reviewer_id: str,
        decision: str,
        notes: Optional[str] = None
    ) -> Loan:
        """Review and approve/reject loan application"""
        
        loan = await db.get(Loan, loan_id)
        if not loan:
            raise ValueError("Loan not found")
        
        if loan.status != LoanStatus.PENDING:
            raise ValueError("Loan is not in pending status")
        
        # Update loan status
        if decision.upper() == "APPROVE":
            loan.status = LoanStatus.APPROVED
            loan.approved_date = datetime.utcnow()
            loan.disbursement_date = datetime.utcnow() + timedelta(days=1)  # Next business day
        elif decision.upper() == "REJECT":
            loan.status = LoanStatus.REJECTED
            loan.rejected_date = datetime.utcnow()
        else:
            raise ValueError("Decision must be 'APPROVE' or 'REJECT'")
        
        loan.reviewer_id = reviewer_id
        loan.review_notes = notes
        loan.reviewed_date = datetime.utcnow()
        
        await db.commit()
        await db.refresh(loan)
        
        # Send notification to user
        try:
            user = await db.get(User, loan.user_id)
            if user and user.email:
                if loan.status == LoanStatus.APPROVED:
                    await notification_service.send_loan_approval_notification(
                        user.email,
                        user.full_name,
                        loan.id,
                        loan.amount,
                        loan.disbursement_date
                    )
                else:
                    await notification_service.send_loan_rejection_notification(
                        user.email,
                        user.full_name,
                        loan.id,
                        notes or "Application did not meet our criteria"
                    )
        except Exception as e:
            logger.error(f"Failed to send loan decision notification: {str(e)}")
        
        logger.info(f"Loan {loan_id} {decision.lower()}ed by {reviewer_id}")
        return loan
    
    async def disburse_loan(self, db: AsyncSession, loan_id: str) -> Loan:
        """Mark loan as disbursed and create payment schedule"""
        
        loan = await db.get(Loan, loan_id)
        if not loan:
            raise ValueError("Loan not found")
        
        if loan.status != LoanStatus.APPROVED:
            raise ValueError("Loan must be approved before disbursement")
        
        # Update loan status
        loan.status = LoanStatus.ACTIVE
        loan.disbursement_date = datetime.utcnow()
        loan.first_payment_date = datetime.utcnow() + timedelta(days=30)  # First payment in 30 days
        
        await db.commit()
        
        # Create payment schedule
        await self._create_payment_schedule(db, loan)
        
        logger.info(f"Loan {loan_id} disbursed successfully")
        return loan
    
    async def get_user_loans(
        self, 
        db: AsyncSession, 
        user_id: str,
        status: Optional[LoanStatus] = None
    ) -> List[Loan]:
        """Get all loans for a user"""
        
        query = select(Loan).where(Loan.user_id == user_id)
        
        if status:
            query = query.where(Loan.status == status)
        
        query = query.order_by(Loan.application_date.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_loan_details(self, db: AsyncSession, loan_id: str) -> Optional[Loan]:
        """Get detailed loan information"""
        return await db.get(Loan, loan_id)
    
    async def calculate_loan_summary(self, db: AsyncSession, loan_id: str) -> Dict[str, any]:
        """Calculate loan summary including payments made and outstanding balance"""
        
        loan = await db.get(Loan, loan_id)
        if not loan:
            raise ValueError("Loan not found")
        
        # Get all payments for this loan
        payments_query = select(Payment).where(
            and_(
                Payment.loan_id == loan_id,
                Payment.status == PaymentStatus.COMPLETED
            )
        )
        result = await db.execute(payments_query)
        payments = result.scalars().all()
        
        # Calculate totals
        total_paid = sum(payment.amount for payment in payments)
        total_loan_amount = loan.amount + (loan.amount * loan.interest_rate * loan.term_months / 12)
        outstanding_balance = total_loan_amount - total_paid
        
        # Calculate next payment date
        next_payment_date = None
        if loan.status == LoanStatus.ACTIVE:
            if loan.first_payment_date:
                payments_made = len(payments)
                next_payment_date = loan.first_payment_date + timedelta(days=30 * payments_made)
        
        return {
            "loan_id": loan_id,
            "original_amount": loan.amount,
            "total_amount_with_interest": total_loan_amount,
            "total_paid": total_paid,
            "outstanding_balance": max(outstanding_balance, Decimal('0')),
            "monthly_payment": loan.monthly_payment,
            "payments_made": len(payments),
            "total_payments": loan.term_months,
            "next_payment_date": next_payment_date,
            "status": loan.status,
            "is_overdue": self._is_loan_overdue(loan, payments)
        }
    
    async def get_overdue_loans(self, db: AsyncSession) -> List[Tuple[Loan, int]]:
        """Get all overdue loans with days overdue"""
        
        # Get all active loans
        query = select(Loan).where(Loan.status == LoanStatus.ACTIVE)
        result = await db.execute(query)
        active_loans = result.scalars().all()
        
        overdue_loans = []
        
        for loan in active_loans:
            if loan.first_payment_date:
                # Get payments for this loan
                payments_query = select(Payment).where(
                    and_(
                        Payment.loan_id == loan.id,
                        Payment.status == PaymentStatus.COMPLETED
                    )
                )
                payments_result = await db.execute(payments_query)
                payments = payments_result.scalars().all()
                
                if self._is_loan_overdue(loan, payments):
                    days_overdue = self._calculate_days_overdue(loan, payments)
                    overdue_loans.append((loan, days_overdue))
        
        return overdue_loans
    
    async def _get_pending_loans(self, db: AsyncSession, user_id: str) -> List[Loan]:
        """Get pending loans for a user"""
        query = select(Loan).where(
            and_(
                Loan.user_id == user_id,
                Loan.status == LoanStatus.PENDING
            )
        )
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _calculate_interest_rate(self, loan_type: LoanType, amount: Decimal) -> Decimal:
        """Calculate interest rate based on loan type and amount"""
        base_rate = self.default_interest_rate
        
        # Adjust rate based on loan type
        if loan_type == LoanType.PERSONAL:
            base_rate += Decimal('0.02')  # +2% for personal loans
        elif loan_type == LoanType.BUSINESS:
            base_rate -= Decimal('0.01')  # -1% for business loans
        elif loan_type == LoanType.MORTGAGE:
            base_rate -= Decimal('0.03')  # -3% for mortgages
        
        # Adjust rate based on amount (higher amounts get better rates)
        if amount >= Decimal('20000000'):  # 20M KHR
            base_rate -= Decimal('0.01')
        elif amount >= Decimal('10000000'):  # 10M KHR
            base_rate -= Decimal('0.005')
        
        return max(base_rate, Decimal('0.08'))  # Minimum 8% rate
    
    def _calculate_monthly_payment(self, amount: Decimal, annual_rate: Decimal, months: int) -> Decimal:
        """Calculate monthly payment using loan formula"""
        monthly_rate = annual_rate / 12
        
        if monthly_rate == 0:
            return amount / months
        
        # PMT formula: P * [r(1+r)^n] / [(1+r)^n - 1]
        factor = (1 + monthly_rate) ** months
        monthly_payment = amount * (monthly_rate * factor) / (factor - 1)
        
        return monthly_payment.quantize(Decimal('0.01'))
    
    async def _create_payment_schedule(self, db: AsyncSession, loan: Loan):
        """Create payment schedule for a loan"""
        current_date = loan.first_payment_date
        
        for i in range(loan.term_months):
            payment = Payment(
                loan_id=loan.id,
                user_id=loan.user_id,
                amount=loan.monthly_payment,
                due_date=current_date,
                status=PaymentStatus.PENDING,
                payment_number=i + 1
            )
            
            db.add(payment)
            current_date += timedelta(days=30)  # Next month
        
        await db.commit()
        logger.info(f"Created payment schedule for loan {loan.id}")
    
    def _is_loan_overdue(self, loan: Loan, payments: List[Payment]) -> bool:
        """Check if loan has overdue payments"""
        if not loan.first_payment_date or loan.status != LoanStatus.ACTIVE:
            return False
        
        payments_made = len(payments)
        expected_payments = self._calculate_expected_payments(loan.first_payment_date)
        
        return payments_made < expected_payments
    
    def _calculate_days_overdue(self, loan: Loan, payments: List[Payment]) -> int:
        """Calculate how many days the loan is overdue"""
        if not loan.first_payment_date:
            return 0
        
        payments_made = len(payments)
        expected_payments = self._calculate_expected_payments(loan.first_payment_date)
        
        if payments_made >= expected_payments:
            return 0
        
        # Calculate the date of the missed payment
        missed_payment_date = loan.first_payment_date + timedelta(days=30 * payments_made)
        days_overdue = (datetime.utcnow() - missed_payment_date).days
        
        return max(days_overdue, 0)
    
    def _calculate_expected_payments(self, first_payment_date: datetime) -> int:
        """Calculate how many payments should have been made by now"""
        days_since_first = (datetime.utcnow() - first_payment_date).days
        return max(0, (days_since_first // 30) + 1)

# Global loan service instance
loan_service = LoanService()