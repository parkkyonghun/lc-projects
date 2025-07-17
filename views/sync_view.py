from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from views.dependencies import get_db, get_current_user, get_repository
from services.application_repository import ApplicationRepository
from services.sync_manager import sync_manager, SyncOperation, SyncPriority
from services.connectivity_monitor import connectivity_monitor
from models.user import User, SyncStatus as UserSyncStatus
from models.loan import Loan, SyncStatus as LoanSyncStatus


class SyncStatusResponse(BaseModel):
    entity_type: str
    entity_id: str
    sync_status: str
    last_synced_at: Optional[datetime]
    version: int
    server_id: Optional[str]
    sync_retry_count: int
    sync_error_message: Optional[str]


class PendingSyncResponse(BaseModel):
    users: int
    loans: int
    total: int
    queue: int


class NetworkStatusResponse(BaseModel):
    is_connected: bool
    quality: str
    latency: float
    last_check: datetime
    consecutive_failures: int


class ForceSyncRequest(BaseModel):
    entity_type: str
    entity_id: str


class BatchSyncRequest(BaseModel):
    entity_types: List[str] = ["user", "loan"]
    priority: str = "normal"


router = APIRouter(prefix="/sync", tags=["synchronization"])


@router.get("/status/{entity_type}/{entity_id}", response_model=SyncStatusResponse)
async def get_sync_status(
    entity_type: str,
    entity_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get sync status for a specific entity"""
    status_info = await repository.get_sync_status(entity_type, entity_id)
    
    if "error" in status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=status_info["error"]
        )
    
    return SyncStatusResponse(**status_info)


@router.get("/pending", response_model=PendingSyncResponse)
async def get_pending_sync_count(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get count of items pending synchronization"""
    pending_info = await repository.get_pending_sync_items()
    return PendingSyncResponse(**pending_info)


@router.post("/force")
async def force_sync_entity(
    request: ForceSyncRequest,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Force immediate sync of a specific entity"""
    success = await repository.force_sync(request.entity_type, request.entity_id)
    
    if success:
        return {"message": f"Successfully synced {request.entity_type} {request.entity_id}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to sync entity"
        )


@router.post("/batch")
async def trigger_batch_sync(
    request: BatchSyncRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger batch synchronization"""
    if not connectivity_monitor.should_sync():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Network connectivity insufficient for sync"
        )
    
    # Queue all pending items for sync
    priority_map = {
        "low": SyncPriority.LOW,
        "normal": SyncPriority.NORMAL,
        "high": SyncPriority.HIGH,
        "critical": SyncPriority.CRITICAL
    }
    
    priority = priority_map.get(request.priority, SyncPriority.NORMAL)
    
    # Get pending items and queue them
    if "user" in request.entity_types:
        repository = get_repository(db)
        pending_users = await repository.get_users_by_sync_status(UserSyncStatus.PENDING)
        for user in pending_users:
            await sync_manager.queue_sync("user", user.id, SyncOperation.UPDATE, priority)
    
    if "loan" in request.entity_types:
        repository = get_repository(db)
        pending_loans = await repository.get_loans_by_sync_status(LoanSyncStatus.PENDING)
        for loan in pending_loans:
            await sync_manager.queue_sync("loan", loan.id, SyncOperation.UPDATE, priority)
    
    # Process the sync queue
    await sync_manager.process_sync_queue(db)
    
    return {"message": "Batch sync initiated", "priority": request.priority}


@router.get("/network-status", response_model=NetworkStatusResponse)
async def get_network_status(current_user: User = Depends(get_current_user)):
    """Get current network connectivity status"""
    # Check current connectivity
    status_info = await connectivity_monitor.check_connectivity()
    
    return NetworkStatusResponse(
        is_connected=status_info.is_connected,
        quality=status_info.quality.value,
        latency=status_info.latency,
        last_check=status_info.last_check,
        consecutive_failures=status_info.consecutive_failures
    )


@router.get("/failed-items")
async def get_failed_sync_items(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get items that failed to sync"""
    failed_users = await repository.get_users_by_sync_status(UserSyncStatus.FAILED)
    failed_loans = await repository.get_loans_by_sync_status(LoanSyncStatus.FAILED)
    
    return {
        "failed_users": [
            {
                "id": user.id,
                "english_name": user.english_name,
                "sync_error_message": user.sync_error_message,
                "sync_retry_count": user.sync_retry_count,
                "last_synced_at": user.last_synced_at.isoformat() if user.last_synced_at else None
            }
            for user in failed_users
        ],
        "failed_loans": [
            {
                "id": loan.id,
                "customer_id": loan.customerId,
                "loan_amount": loan.loanAmount,
                "sync_error_message": loan.sync_error_message,
                "sync_retry_count": loan.sync_retry_count,
                "last_synced_at": loan.last_synced_at.isoformat() if loan.last_synced_at else None
            }
            for loan in failed_loans
        ]
    }


@router.post("/retry-failed")
async def retry_failed_sync_items(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retry synchronization for all failed items"""
    repository = get_repository(db)
    
    # Get failed items
    failed_users = await repository.get_users_by_sync_status(UserSyncStatus.FAILED)
    failed_loans = await repository.get_loans_by_sync_status(LoanSyncStatus.FAILED)
    
    # Reset their status and queue for sync
    for user in failed_users:
        user.sync_status = UserSyncStatus.PENDING
        user.sync_retry_count = 0
        user.sync_error_message = None
        await sync_manager.queue_sync("user", user.id, SyncOperation.UPDATE, SyncPriority.HIGH)
    
    for loan in failed_loans:
        loan.sync_status = LoanSyncStatus.PENDING
        loan.sync_retry_count = 0
        loan.sync_error_message = None
        await sync_manager.queue_sync("loan", loan.id, SyncOperation.UPDATE, SyncPriority.HIGH)
    
    await db.commit()
    
    # Process the sync queue
    await sync_manager.process_sync_queue(db)
    
    return {
        "message": "Retry initiated for failed items",
        "failed_users_count": len(failed_users),
        "failed_loans_count": len(failed_loans)
    }