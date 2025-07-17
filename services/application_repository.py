from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, or_, desc

from models.user import User, SyncStatus as UserSyncStatus
from models.loan import Loan, SyncStatus as LoanSyncStatus
from schemas.dto import UserDTO, LoanDTO
from services.sync_manager import sync_manager, SyncOperation, SyncPriority


class ApplicationRepositoryInterface(ABC):
    """Interface for application repository with sync support"""
    
    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        pass
    
    @abstractmethod
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[User]:
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        pass
    
    @abstractmethod
    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        pass
    
    @abstractmethod
    async def create_loan(self, loan_data: Dict[str, Any]) -> Loan:
        pass
    
    @abstractmethod
    async def update_loan(self, loan_id: str, loan_data: Dict[str, Any]) -> Optional[Loan]:
        pass
    
    @abstractmethod
    async def get_loan(self, loan_id: str) -> Optional[Loan]:
        pass
    
    @abstractmethod
    async def get_loans(self, skip: int = 0, limit: int = 100, 
                       customer_id: Optional[str] = None) -> List[Loan]:
        pass
    
    @abstractmethod
    async def delete_loan(self, loan_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_sync_status(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def force_sync(self, entity_type: str, entity_id: str) -> bool:
        pass


class ApplicationRepository(ApplicationRepositoryInterface):
    """Repository implementation with offline/online sync support"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user and queue for sync"""
        import uuid
        
        user = User(
            id=str(uuid.uuid4()),
            khmer_name=user_data["khmer_name"],
            english_name=user_data["english_name"],
            id_card_number=user_data["id_card_number"],
            phone_number=user_data["phone_number"],
            address=user_data.get("address"),
            occupation=user_data.get("occupation"),
            monthly_income=user_data.get("monthly_income"),
            id_card_photo_url=user_data.get("id_card_photo_url"),
            profile_photo_url=user_data.get("profile_photo_url"),
            hashed_password=user_data["hashed_password"],
            sync_status=UserSyncStatus.PENDING,
            version=1
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        # Queue for sync
        await sync_manager.queue_sync("user", user.id, SyncOperation.CREATE, SyncPriority.NORMAL)
        
        return user
    
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[User]:
        """Update a user and queue for sync"""
        result = await self.db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        # Update fields
        for field, value in user_data.items():
            if hasattr(user, field) and field not in ['id', 'created_at']:
                setattr(user, field, value)
        
        # Update sync-related fields
        user.sync_status = UserSyncStatus.PENDING
        user.version += 1
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(user)
        
        # Queue for sync
        await sync_manager.queue_sync("user", user.id, SyncOperation.UPDATE, SyncPriority.NORMAL)
        
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID"""
        result = await self.db.execute(
            select(User).where(and_(User.id == user_id, User.is_deleted == False))
        )
        return result.scalar_one_or_none()
    
    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination"""
        result = await self.db.execute(
            select(User)
            .where(User.is_deleted == False)
            .order_by(desc(User.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def delete_user(self, user_id: str) -> bool:
        """Soft delete a user and queue for sync"""
        result = await self.db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if not user:
            return False
        
        # Soft delete
        user.is_deleted = True
        user.sync_status = UserSyncStatus.PENDING
        user.version += 1
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        
        # Queue for sync
        await sync_manager.queue_sync("user", user.id, SyncOperation.DELETE, SyncPriority.NORMAL)
        
        return True
    
    async def create_loan(self, loan_data: Dict[str, Any]) -> Loan:
        """Create a new loan and queue for sync"""
        import uuid
        
        loan = Loan(
            id=str(uuid.uuid4()),
            customerId=loan_data["customer_id"],
            branchId=loan_data["branch_id"],
            loanAmount=loan_data["loan_amount"],
            interestRate=loan_data["interest_rate"],
            termMonths=loan_data["term_months"],
            monthlyPayment=loan_data.get("monthly_payment"),
            purpose=loan_data.get("purpose"),
            status=loan_data.get("status", "pending"),
            applicationDate=loan_data.get("application_date", datetime.utcnow()),
            startDate=loan_data.get("start_date"),
            nextPaymentDate=loan_data.get("next_payment_date"),
            remainingBalance=loan_data.get("remaining_balance"),
            collateralDescription=loan_data.get("collateral_description"),
            repaymentSchedule=loan_data.get("repayment_schedule"),
            sync_status=LoanSyncStatus.PENDING,
            version=1
        )
        
        self.db.add(loan)
        await self.db.commit()
        await self.db.refresh(loan)
        
        # Queue for sync
        await sync_manager.queue_sync("loan", loan.id, SyncOperation.CREATE, SyncPriority.HIGH)
        
        return loan
    
    async def update_loan(self, loan_id: str, loan_data: Dict[str, Any]) -> Optional[Loan]:
        """Update a loan and queue for sync"""
        result = await self.db.execute(select(Loan).where(Loan.id == loan_id))
        loan = result.scalar_one_or_none()
        
        if not loan:
            return None
        
        # Update fields
        for field, value in loan_data.items():
            if hasattr(loan, field) and field not in ['id', 'created_at']:
                setattr(loan, field, value)
        
        # Update sync-related fields
        loan.sync_status = LoanSyncStatus.PENDING
        loan.version += 1
        loan.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(loan)
        
        # Determine priority based on status changes
        priority = SyncPriority.HIGH if loan_data.get("status") else SyncPriority.NORMAL
        
        # Queue for sync
        await sync_manager.queue_sync("loan", loan.id, SyncOperation.UPDATE, priority)
        
        return loan
    
    async def get_loan(self, loan_id: str) -> Optional[Loan]:
        """Get a loan by ID"""
        result = await self.db.execute(
            select(Loan).where(and_(Loan.id == loan_id, Loan.is_deleted == False))
        )
        return result.scalar_one_or_none()
    
    async def get_loans(self, skip: int = 0, limit: int = 100, 
                       customer_id: Optional[str] = None) -> List[Loan]:
        """Get loans with pagination and optional customer filter"""
        query = select(Loan).where(Loan.is_deleted == False)
        
        if customer_id:
            query = query.where(Loan.customerId == customer_id)
        
        query = query.order_by(desc(Loan.created_at)).offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def delete_loan(self, loan_id: str) -> bool:
        """Soft delete a loan and queue for sync"""
        result = await self.db.execute(select(Loan).where(Loan.id == loan_id))
        loan = result.scalar_one_or_none()
        
        if not loan:
            return False
        
        # Soft delete
        loan.is_deleted = True
        loan.sync_status = LoanSyncStatus.PENDING
        loan.version += 1
        loan.updated_at = datetime.utcnow()
        
        await self.db.commit()
        
        # Queue for sync
        await sync_manager.queue_sync("loan", loan.id, SyncOperation.DELETE, SyncPriority.NORMAL)
        
        return True
    
    async def get_sync_status(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get sync status for an entity"""
        if entity_type == "user":
            result = await self.db.execute(select(User).where(User.id == entity_id))
            entity = result.scalar_one_or_none()
        elif entity_type == "loan":
            result = await self.db.execute(select(Loan).where(Loan.id == entity_id))
            entity = result.scalar_one_or_none()
        else:
            return {"error": "Unknown entity type"}
        
        if not entity:
            return {"error": "Entity not found"}
        
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "sync_status": entity.sync_status,
            "last_synced_at": entity.last_synced_at.isoformat() if entity.last_synced_at else None,
            "version": entity.version,
            "server_id": entity.server_id,
            "sync_retry_count": entity.sync_retry_count,
            "sync_error_message": entity.sync_error_message
        }
    
    async def force_sync(self, entity_type: str, entity_id: str) -> bool:
        """Force immediate sync of an entity"""
        return await sync_manager.force_sync_entity(self.db, entity_type, entity_id)
    
    async def get_pending_sync_items(self) -> Dict[str, Any]:
        """Get all items pending sync"""
        return await sync_manager.get_pending_sync_count(self.db)
    
    async def get_users_by_sync_status(self, sync_status: UserSyncStatus) -> List[User]:
        """Get users by sync status"""
        result = await self.db.execute(
            select(User).where(and_(
                User.sync_status == sync_status,
                User.is_deleted == False
            ))
        )
        return result.scalars().all()
    
    async def get_loans_by_sync_status(self, sync_status: LoanSyncStatus) -> List[Loan]:
        """Get loans by sync status"""
        result = await self.db.execute(
            select(Loan).where(and_(
                Loan.sync_status == sync_status,
                Loan.is_deleted == False
            ))
        )
        return result.scalars().all()
    
    async def search_loans(self, query: str, skip: int = 0, limit: int = 100) -> List[Loan]:
        """Search loans by customer name or loan purpose"""
        # Join with User table to search by customer name
        result = await self.db.execute(
            select(Loan)
            .join(User, Loan.customerId == User.id)
            .where(and_(
                Loan.is_deleted == False,
                or_(
                    User.khmer_name.ilike(f"%{query}%"),
                    User.english_name.ilike(f"%{query}%"),
                    Loan.purpose.ilike(f"%{query}%")
                )
            ))
            .order_by(desc(Loan.created_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()


def get_application_repository(db: AsyncSession) -> ApplicationRepository:
    """Factory function to create repository instance"""
    return ApplicationRepository(db)