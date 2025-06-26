from typing import Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, extract
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from models.loan import Loan, LoanStatus, LoanType
from models.user import User
from models.payment import Payment, PaymentStatus
from models.branch import Branch

logger = logging.getLogger(__name__)

class DashboardService:
    """Service for dashboard analytics and reporting"""
    
    async def get_overview_stats(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get overview statistics for dashboard"""
        
        # Base query filters
        loan_filter = []
        user_filter = []
        payment_filter = []
        
        if branch_id:
            loan_filter.append(Loan.branch_id == branch_id)
            user_filter.append(User.branch_id == branch_id)
            payment_filter.append(Payment.branch_id == branch_id)
        
        # Total loans
        total_loans_query = select(func.count(Loan.id))
        if loan_filter:
            total_loans_query = total_loans_query.where(and_(*loan_filter))
        total_loans_result = await db.execute(total_loans_query)
        total_loans = total_loans_result.scalar() or 0
        
        # Active loans
        active_loans_query = select(func.count(Loan.id)).where(Loan.status == LoanStatus.ACTIVE)
        if loan_filter:
            active_loans_query = active_loans_query.where(and_(*loan_filter))
        active_loans_result = await db.execute(active_loans_query)
        active_loans = active_loans_result.scalar() or 0
        
        # Pending applications
        pending_loans_query = select(func.count(Loan.id)).where(Loan.status == LoanStatus.PENDING)
        if loan_filter:
            pending_loans_query = pending_loans_query.where(and_(*loan_filter))
        pending_loans_result = await db.execute(pending_loans_query)
        pending_loans = pending_loans_result.scalar() or 0
        
        # Total loan amount disbursed
        disbursed_amount_query = select(func.sum(Loan.amount)).where(
            Loan.status.in_([LoanStatus.ACTIVE, LoanStatus.COMPLETED])
        )
        if loan_filter:
            disbursed_amount_query = disbursed_amount_query.where(and_(*loan_filter))
        disbursed_amount_result = await db.execute(disbursed_amount_query)
        total_disbursed = disbursed_amount_result.scalar() or Decimal('0')
        
        # Total payments received
        payments_received_query = select(func.sum(Payment.amount)).where(
            Payment.status == PaymentStatus.COMPLETED
        )
        if payment_filter:
            payments_received_query = payments_received_query.where(and_(*payment_filter))
        payments_received_result = await db.execute(payments_received_query)
        total_payments = payments_received_result.scalar() or Decimal('0')
        
        # Outstanding amount
        outstanding_amount = total_disbursed - total_payments
        
        # Total users
        total_users_query = select(func.count(User.id))
        if user_filter:
            total_users_query = total_users_query.where(and_(*user_filter))
        total_users_result = await db.execute(total_users_query)
        total_users = total_users_result.scalar() or 0
        
        # Overdue loans count
        overdue_count = await self._get_overdue_loans_count(db, branch_id)
        
        return {
            'total_loans': total_loans,
            'active_loans': active_loans,
            'pending_applications': pending_loans,
            'total_disbursed_amount': float(total_disbursed),
            'total_payments_received': float(total_payments),
            'outstanding_amount': float(outstanding_amount),
            'total_users': total_users,
            'overdue_loans': overdue_count,
            'collection_rate': float(total_payments / total_disbursed * 100) if total_disbursed > 0 else 0
        }
    
    async def get_loan_statistics(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get detailed loan statistics"""
        
        base_filter = []
        if branch_id:
            base_filter.append(Loan.branch_id == branch_id)
        
        # Loans by status
        status_query = select(
            Loan.status,
            func.count(Loan.id).label('count'),
            func.sum(Loan.amount).label('total_amount')
        ).group_by(Loan.status)
        
        if base_filter:
            status_query = status_query.where(and_(*base_filter))
        
        status_result = await db.execute(status_query)
        loans_by_status = {
            row.status.value: {
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in status_result
        }
        
        # Loans by type
        type_query = select(
            Loan.loan_type,
            func.count(Loan.id).label('count'),
            func.sum(Loan.amount).label('total_amount')
        ).group_by(Loan.loan_type)
        
        if base_filter:
            type_query = type_query.where(and_(*base_filter))
        
        type_result = await db.execute(type_query)
        loans_by_type = {
            row.loan_type.value: {
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in type_result
        }
        
        # Monthly loan disbursements (last 12 months)
        monthly_disbursements = await self._get_monthly_disbursements(db, branch_id)
        
        # Average loan amount
        avg_amount_query = select(func.avg(Loan.amount))
        if base_filter:
            avg_amount_query = avg_amount_query.where(and_(*base_filter))
        avg_amount_result = await db.execute(avg_amount_query)
        avg_loan_amount = avg_amount_result.scalar() or 0
        
        return {
            'loans_by_status': loans_by_status,
            'loans_by_type': loans_by_type,
            'monthly_disbursements': monthly_disbursements,
            'average_loan_amount': float(avg_loan_amount)
        }
    
    async def get_payment_analytics(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get payment analytics"""
        
        base_filter = []
        if branch_id:
            base_filter.append(Payment.branch_id == branch_id)
        
        # Payments by status
        status_query = select(
            Payment.status,
            func.count(Payment.id).label('count'),
            func.sum(Payment.amount).label('total_amount')
        ).group_by(Payment.status)
        
        if base_filter:
            status_query = status_query.where(and_(*base_filter))
        
        status_result = await db.execute(status_query)
        payments_by_status = {
            row.status.value: {
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in status_result
        }
        
        # Monthly payment collections (last 12 months)
        monthly_collections = await self._get_monthly_collections(db, branch_id)
        
        # Payment methods distribution
        method_query = select(
            Payment.payment_method,
            func.count(Payment.id).label('count'),
            func.sum(Payment.amount).label('total_amount')
        ).where(Payment.status == PaymentStatus.COMPLETED).group_by(Payment.payment_method)
        
        if base_filter:
            method_query = method_query.where(and_(*base_filter))
        
        method_result = await db.execute(method_query)
        payments_by_method = {
            (row.payment_method or 'Unknown'): {
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in method_result
        }
        
        # Late payment statistics
        late_payments = await self._get_late_payment_stats(db, branch_id)
        
        return {
            'payments_by_status': payments_by_status,
            'monthly_collections': monthly_collections,
            'payments_by_method': payments_by_method,
            'late_payment_stats': late_payments
        }
    
    async def get_user_analytics(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get user analytics"""
        
        base_filter = []
        if branch_id:
            base_filter.append(User.branch_id == branch_id)
        
        # New users by month (last 12 months)
        monthly_users = await self._get_monthly_user_registrations(db, branch_id)
        
        # Users by verification status
        verification_query = select(
            User.is_verified,
            func.count(User.id).label('count')
        ).group_by(User.is_verified)
        
        if base_filter:
            verification_query = verification_query.where(and_(*base_filter))
        
        verification_result = await db.execute(verification_query)
        users_by_verification = {
            ('Verified' if row.is_verified else 'Unverified'): row.count
            for row in verification_result
        }
        
        # Active users (users with loans)
        active_users_query = select(func.count(func.distinct(Loan.user_id)))
        if branch_id:
            active_users_query = active_users_query.where(Loan.branch_id == branch_id)
        active_users_result = await db.execute(active_users_query)
        active_users = active_users_result.scalar() or 0
        
        return {
            'monthly_registrations': monthly_users,
            'users_by_verification': users_by_verification,
            'active_users': active_users
        }
    
    async def get_branch_performance(self, db: AsyncSession) -> List[Dict[str, any]]:
        """Get performance metrics by branch"""
        
        # Branch performance query
        branch_query = select(
            Branch.id,
            Branch.name,
            func.count(Loan.id).label('total_loans'),
            func.sum(Loan.amount).label('total_disbursed'),
            func.count(User.id).label('total_users')
        ).select_from(
            Branch
        ).outerjoin(
            Loan, Branch.id == Loan.branch_id
        ).outerjoin(
            User, Branch.id == User.branch_id
        ).group_by(Branch.id, Branch.name)
        
        result = await db.execute(branch_query)
        
        branch_performance = []
        for row in result:
            # Get payments for this branch
            payments_query = select(func.sum(Payment.amount)).where(
                and_(
                    Payment.branch_id == row.id,
                    Payment.status == PaymentStatus.COMPLETED
                )
            )
            payments_result = await db.execute(payments_query)
            total_payments = payments_result.scalar() or Decimal('0')
            
            # Calculate collection rate
            collection_rate = 0
            if row.total_disbursed and row.total_disbursed > 0:
                collection_rate = float(total_payments / row.total_disbursed * 100)
            
            branch_performance.append({
                'branch_id': row.id,
                'branch_name': row.name,
                'total_loans': row.total_loans or 0,
                'total_disbursed': float(row.total_disbursed or 0),
                'total_payments': float(total_payments),
                'total_users': row.total_users or 0,
                'collection_rate': collection_rate
            })
        
        return branch_performance
    
    async def get_risk_metrics(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get risk assessment metrics"""
        
        base_filter = []
        if branch_id:
            base_filter.append(Loan.branch_id == branch_id)
        
        # Default rate (loans that are significantly overdue)
        total_active_query = select(func.count(Loan.id)).where(Loan.status == LoanStatus.ACTIVE)
        if base_filter:
            total_active_query = total_active_query.where(and_(*base_filter))
        total_active_result = await db.execute(total_active_query)
        total_active_loans = total_active_result.scalar() or 0
        
        # Get overdue loans
        overdue_count = await self._get_overdue_loans_count(db, branch_id)
        
        # Calculate default rate
        default_rate = (overdue_count / total_active_loans * 100) if total_active_loans > 0 else 0
        
        # Portfolio at risk (PAR)
        par_30 = await self._calculate_portfolio_at_risk(db, 30, branch_id)
        par_60 = await self._calculate_portfolio_at_risk(db, 60, branch_id)
        par_90 = await self._calculate_portfolio_at_risk(db, 90, branch_id)
        
        return {
            'default_rate': default_rate,
            'overdue_loans_count': overdue_count,
            'total_active_loans': total_active_loans,
            'portfolio_at_risk': {
                'par_30': par_30,
                'par_60': par_60,
                'par_90': par_90
            }
        }
    
    async def _get_monthly_disbursements(self, db: AsyncSession, branch_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Get monthly loan disbursements for the last 12 months"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)
        
        query = select(
            extract('year', Loan.disbursement_date).label('year'),
            extract('month', Loan.disbursement_date).label('month'),
            func.count(Loan.id).label('count'),
            func.sum(Loan.amount).label('total_amount')
        ).where(
            and_(
                Loan.disbursement_date >= start_date,
                Loan.disbursement_date <= end_date,
                Loan.status.in_([LoanStatus.ACTIVE, LoanStatus.COMPLETED])
            )
        ).group_by(
            extract('year', Loan.disbursement_date),
            extract('month', Loan.disbursement_date)
        ).order_by(
            extract('year', Loan.disbursement_date),
            extract('month', Loan.disbursement_date)
        )
        
        if branch_id:
            query = query.where(Loan.branch_id == branch_id)
        
        result = await db.execute(query)
        
        return [
            {
                'year': int(row.year),
                'month': int(row.month),
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in result
        ]
    
    async def _get_monthly_collections(self, db: AsyncSession, branch_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Get monthly payment collections for the last 12 months"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)
        
        query = select(
            extract('year', Payment.payment_date).label('year'),
            extract('month', Payment.payment_date).label('month'),
            func.count(Payment.id).label('count'),
            func.sum(Payment.amount).label('total_amount')
        ).where(
            and_(
                Payment.payment_date >= start_date,
                Payment.payment_date <= end_date,
                Payment.status == PaymentStatus.COMPLETED
            )
        ).group_by(
            extract('year', Payment.payment_date),
            extract('month', Payment.payment_date)
        ).order_by(
            extract('year', Payment.payment_date),
            extract('month', Payment.payment_date)
        )
        
        if branch_id:
            query = query.where(Payment.branch_id == branch_id)
        
        result = await db.execute(query)
        
        return [
            {
                'year': int(row.year),
                'month': int(row.month),
                'count': row.count,
                'total_amount': float(row.total_amount or 0)
            }
            for row in result
        ]
    
    async def _get_monthly_user_registrations(self, db: AsyncSession, branch_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Get monthly user registrations for the last 12 months"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)
        
        query = select(
            extract('year', User.created_at).label('year'),
            extract('month', User.created_at).label('month'),
            func.count(User.id).label('count')
        ).where(
            and_(
                User.created_at >= start_date,
                User.created_at <= end_date
            )
        ).group_by(
            extract('year', User.created_at),
            extract('month', User.created_at)
        ).order_by(
            extract('year', User.created_at),
            extract('month', User.created_at)
        )
        
        if branch_id:
            query = query.where(User.branch_id == branch_id)
        
        result = await db.execute(query)
        
        return [
            {
                'year': int(row.year),
                'month': int(row.month),
                'count': row.count
            }
            for row in result
        ]
    
    async def _get_overdue_loans_count(self, db: AsyncSession, branch_id: Optional[str] = None) -> int:
        """Get count of overdue loans"""
        # This is a simplified version - in practice, you'd need to check
        # payment schedules and calculate actual overdue status
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        query = select(func.count(Loan.id)).where(
            and_(
                Loan.status == LoanStatus.ACTIVE,
                Loan.first_payment_date < thirty_days_ago
            )
        )
        
        if branch_id:
            query = query.where(Loan.branch_id == branch_id)
        
        result = await db.execute(query)
        return result.scalar() or 0
    
    async def _get_late_payment_stats(self, db: AsyncSession, branch_id: Optional[str] = None) -> Dict[str, any]:
        """Get late payment statistics"""
        
        base_filter = [Payment.status == PaymentStatus.COMPLETED]
        if branch_id:
            base_filter.append(Payment.branch_id == branch_id)
        
        # Payments made after due date
        late_payments_query = select(func.count(Payment.id)).where(
            and_(
                *base_filter,
                Payment.payment_date > Payment.due_date
            )
        )
        
        late_payments_result = await db.execute(late_payments_query)
        late_payments_count = late_payments_result.scalar() or 0
        
        # Total completed payments
        total_payments_query = select(func.count(Payment.id)).where(and_(*base_filter))
        total_payments_result = await db.execute(total_payments_query)
        total_payments_count = total_payments_result.scalar() or 0
        
        # Calculate late payment rate
        late_payment_rate = (late_payments_count / total_payments_count * 100) if total_payments_count > 0 else 0
        
        return {
            'late_payments_count': late_payments_count,
            'total_payments_count': total_payments_count,
            'late_payment_rate': late_payment_rate
        }
    
    async def _calculate_portfolio_at_risk(self, db: AsyncSession, days: int, branch_id: Optional[str] = None) -> float:
        """Calculate Portfolio at Risk (PAR) for given number of days"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get total outstanding portfolio
        total_portfolio_query = select(func.sum(Loan.amount)).where(
            Loan.status == LoanStatus.ACTIVE
        )
        
        if branch_id:
            total_portfolio_query = total_portfolio_query.where(Loan.branch_id == branch_id)
        
        total_portfolio_result = await db.execute(total_portfolio_query)
        total_portfolio = total_portfolio_result.scalar() or Decimal('0')
        
        # Get loans with payments overdue by specified days
        # This is simplified - in practice, you'd need more complex logic
        # to determine actual overdue amounts
        
        overdue_portfolio_query = select(func.sum(Loan.amount)).where(
            and_(
                Loan.status == LoanStatus.ACTIVE,
                Loan.first_payment_date < cutoff_date
            )
        )
        
        if branch_id:
            overdue_portfolio_query = overdue_portfolio_query.where(Loan.branch_id == branch_id)
        
        overdue_portfolio_result = await db.execute(overdue_portfolio_query)
        overdue_portfolio = overdue_portfolio_result.scalar() or Decimal('0')
        
        # Calculate PAR percentage
        if total_portfolio > 0:
            return float(overdue_portfolio / total_portfolio * 100)
        else:
            return 0.0

# Global dashboard service instance
dashboard_service = DashboardService()