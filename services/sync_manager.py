import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, or_
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from models.user import User, SyncStatus as UserSyncStatus
from models.loan import Loan, SyncStatus as LoanSyncStatus
from schemas.dto import (
    SyncRequestDTO, SyncResponseDTO, BatchSyncRequestDTO, 
    BatchSyncResponseDTO, ConflictResolutionDTO
)
from services.connectivity_monitor import connectivity_monitor, ConnectionQuality


class SyncOperation(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class SyncPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SyncQueueItem:
    def __init__(self, entity_type: str, entity_id: str, operation: SyncOperation, 
                 priority: SyncPriority = SyncPriority.NORMAL, data: Optional[Dict] = None):
        self.id = str(uuid.uuid4())
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.operation = operation
        self.priority = priority
        self.data = data or {}
        self.created_at = datetime.utcnow()
        self.retry_count = 0
        self.last_error: Optional[str] = None


class SyncManager:
    """Manages data synchronization between local and remote storage"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.sync_queue: List[SyncQueueItem] = []
        self.is_syncing = False
        self.sync_callbacks: List[Callable] = []
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
    
    def add_sync_callback(self, callback: Callable):
        """Add callback to be notified of sync events"""
        self.sync_callbacks.append(callback)
    
    async def _notify_callbacks(self, event: str, data: Dict):
        """Notify all callbacks of sync events"""
        for callback in self.sync_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                print(f"Error notifying sync callback: {e}")
    
    async def queue_sync(self, entity_type: str, entity_id: str, operation: SyncOperation,
                        priority: SyncPriority = SyncPriority.NORMAL, data: Optional[Dict] = None):
        """Queue an item for synchronization"""
        item = SyncQueueItem(entity_type, entity_id, operation, priority, data)
        
        # Check if item already exists in queue
        existing_item = next(
            (i for i in self.sync_queue 
             if i.entity_type == entity_type and i.entity_id == entity_id),
            None
        )
        
        if existing_item:
            # Update existing item with higher priority operation
            if priority.value > existing_item.priority.value:
                existing_item.operation = operation
                existing_item.priority = priority
                existing_item.data = data
        else:
            self.sync_queue.append(item)
        
        # Sort queue by priority
        self.sync_queue.sort(key=lambda x: (
            {"critical": 0, "high": 1, "normal": 2, "low": 3}[x.priority.value],
            x.created_at
        ))
        
        await self._notify_callbacks("item_queued", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation.value
        })
    
    async def process_sync_queue(self, db: AsyncSession):
        """Process all items in the sync queue"""
        if self.is_syncing or not connectivity_monitor.should_sync():
            return
        
        self.is_syncing = True
        
        try:
            batch_size = connectivity_monitor.get_sync_batch_size()
            if batch_size == 0:
                return
            
            # Get items to sync
            items_to_sync = self.sync_queue[:batch_size]
            if not items_to_sync:
                return
            
            await self._notify_callbacks("sync_started", {
                "batch_size": len(items_to_sync)
            })
            
            # Process items in batches
            successful_items = []
            failed_items = []
            
            for item in items_to_sync:
                try:
                    success = await self._sync_item(db, item)
                    if success:
                        successful_items.append(item)
                    else:
                        failed_items.append(item)
                except Exception as e:
                    print(f"Error syncing item {item.id}: {e}")
                    item.last_error = str(e)
                    item.retry_count += 1
                    failed_items.append(item)
            
            # Remove successful items from queue
            for item in successful_items:
                self.sync_queue.remove(item)
            
            # Handle failed items
            for item in failed_items:
                if item.retry_count >= settings.sync_retry_attempts:
                    # Mark as permanently failed
                    await self._mark_sync_failed(db, item)
                    self.sync_queue.remove(item)
            
            await self._notify_callbacks("sync_completed", {
                "successful": len(successful_items),
                "failed": len(failed_items)
            })
            
        finally:
            self.is_syncing = False
    
    async def _sync_item(self, db: AsyncSession, item: SyncQueueItem) -> bool:
        """Sync a single item"""
        try:
            if item.entity_type == "user":
                return await self._sync_user(db, item)
            elif item.entity_type == "loan":
                return await self._sync_loan(db, item)
            else:
                print(f"Unknown entity type: {item.entity_type}")
                return False
        except Exception as e:
            print(f"Error syncing {item.entity_type} {item.entity_id}: {e}")
            return False
    
    async def _sync_user(self, db: AsyncSession, item: SyncQueueItem) -> bool:
        """Sync a user entity"""
        result = await db.execute(select(User).where(User.id == item.entity_id))
        user = result.scalar_one_or_none()
        
        if not user:
            return False
        
        # Create sync request
        sync_request = SyncRequestDTO(
            entity_type="user",
            entity_id=user.id,
            action=item.operation.value,
            data={
                "khmer_name": user.khmer_name,
                "english_name": user.english_name,
                "id_card_number": user.id_card_number,
                "phone_number": user.phone_number,
                "address": user.address,
                "occupation": user.occupation,
                "monthly_income": user.monthly_income,
                "id_card_photo_url": user.id_card_photo_url,
                "profile_photo_url": user.profile_photo_url,
            },
            client_timestamp=datetime.utcnow(),
            version=user.version
        )
        
        # Send to server (mock implementation)
        response = await self._send_to_server(sync_request)
        
        if response.success:
            # Update local sync status
            user.sync_status = UserSyncStatus.SYNCED
            user.last_synced_at = datetime.utcnow()
            user.server_id = response.server_id
            user.sync_retry_count = 0
            user.sync_error_message = None
            
            await db.commit()
            return True
        else:
            # Handle sync failure
            user.sync_status = UserSyncStatus.FAILED
            user.sync_retry_count += 1
            user.sync_error_message = response.error_message
            
            await db.commit()
            return False
    
    async def _sync_loan(self, db: AsyncSession, item: SyncQueueItem) -> bool:
        """Sync a loan entity"""
        result = await db.execute(select(Loan).where(Loan.id == item.entity_id))
        loan = result.scalar_one_or_none()
        
        if not loan:
            return False
        
        # Create sync request
        sync_request = SyncRequestDTO(
            entity_type="loan",
            entity_id=loan.id,
            action=item.operation.value,
            data={
                "customer_id": loan.customerId,
                "branch_id": loan.branchId,
                "loan_amount": loan.loanAmount,
                "interest_rate": loan.interestRate,
                "term_months": loan.termMonths,
                "monthly_payment": loan.monthlyPayment,
                "purpose": loan.purpose,
                "status": loan.status,
                "application_date": loan.applicationDate.isoformat() if loan.applicationDate else None,
                "start_date": loan.startDate.isoformat() if loan.startDate else None,
                "next_payment_date": loan.nextPaymentDate.isoformat() if loan.nextPaymentDate else None,
                "remaining_balance": loan.remainingBalance,
                "collateral_description": loan.collateralDescription,
                "repayment_schedule": loan.repaymentSchedule,
            },
            client_timestamp=datetime.utcnow(),
            version=loan.version
        )
        
        # Send to server (mock implementation)
        response = await self._send_to_server(sync_request)
        
        if response.success:
            # Update local sync status
            loan.sync_status = LoanSyncStatus.SYNCED
            loan.last_synced_at = datetime.utcnow()
            loan.server_id = response.server_id
            loan.sync_retry_count = 0
            loan.sync_error_message = None
            
            await db.commit()
            return True
        else:
            # Handle sync failure
            loan.sync_status = LoanSyncStatus.FAILED
            loan.sync_retry_count += 1
            loan.sync_error_message = response.error_message
            
            await db.commit()
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _send_to_server(self, sync_request: SyncRequestDTO) -> SyncResponseDTO:
        """Send sync request to server with retry logic"""
        # Mock implementation - replace with actual HTTP client
        import random
        
        # Simulate network delay based on connection quality
        if connectivity_monitor.status.quality == ConnectionQuality.POOR:
            await asyncio.sleep(2)
        elif connectivity_monitor.status.quality == ConnectionQuality.FAIR:
            await asyncio.sleep(1)
        else:
            await asyncio.sleep(0.1)
        
        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            return SyncResponseDTO(
                success=True,
                entity_type=sync_request.entity_type,
                entity_id=sync_request.entity_id,
                server_id=str(uuid.uuid4()),
                server_timestamp=datetime.utcnow()
            )
        else:
            return SyncResponseDTO(
                success=False,
                entity_type=sync_request.entity_type,
                entity_id=sync_request.entity_id,
                error_message="Server error",
                server_timestamp=datetime.utcnow()
            )
    
    async def _mark_sync_failed(self, db: AsyncSession, item: SyncQueueItem):
        """Mark an item as permanently failed"""
        if item.entity_type == "user":
            result = await db.execute(select(User).where(User.id == item.entity_id))
            user = result.scalar_one_or_none()
            if user:
                user.sync_status = UserSyncStatus.FAILED
                user.sync_error_message = f"Max retries exceeded: {item.last_error}"
        
        elif item.entity_type == "loan":
            result = await db.execute(select(Loan).where(Loan.id == item.entity_id))
            loan = result.scalar_one_or_none()
            if loan:
                loan.sync_status = LoanSyncStatus.FAILED
                loan.sync_error_message = f"Max retries exceeded: {item.last_error}"
        
        await db.commit()
    
    async def get_pending_sync_count(self, db: AsyncSession) -> Dict[str, int]:
        """Get count of pending sync items"""
        user_result = await db.execute(
            select(User).where(User.sync_status == UserSyncStatus.PENDING)
        )
        pending_users = len(user_result.scalars().all())
        
        loan_result = await db.execute(
            select(Loan).where(Loan.sync_status == LoanSyncStatus.PENDING)
        )
        pending_loans = len(loan_result.scalars().all())
        
        return {
            "users": pending_users,
            "loans": pending_loans,
            "total": pending_users + pending_loans,
            "queue": len(self.sync_queue)
        }
    
    async def force_sync_entity(self, db: AsyncSession, entity_type: str, entity_id: str):
        """Force immediate sync of a specific entity"""
        await self.queue_sync(entity_type, entity_id, SyncOperation.UPDATE, SyncPriority.HIGH)
        
        # Process just this item
        item = next(
            (i for i in self.sync_queue 
             if i.entity_type == entity_type and i.entity_id == entity_id),
            None
        )
        
        if item:
            success = await self._sync_item(db, item)
            if success:
                self.sync_queue.remove(item)
            return success
        
        return False


# Global sync manager instance
sync_manager = SyncManager()