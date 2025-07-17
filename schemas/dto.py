from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SyncStatusEnum(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class LoanStatusEnum(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"


class BaseSyncDTO(BaseModel):
    """Base DTO with sync-related fields"""
    server_id: Optional[str] = None
    sync_status: SyncStatusEnum = SyncStatusEnum.PENDING
    last_synced_at: Optional[datetime] = None
    version: int = 1
    is_deleted: bool = False


class UserDTO(BaseSyncDTO):
    """Data Transfer Object for User"""
    id: str
    khmer_name: str
    english_name: str
    id_card_number: str
    phone_number: str
    address: Optional[str] = None
    occupation: Optional[str] = None
    monthly_income: Optional[float] = None
    id_card_photo_url: Optional[str] = None
    profile_photo_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LoanDTO(BaseSyncDTO):
    """Data Transfer Object for Loan"""
    id: str
    customer_id: str
    branch_id: str
    loan_amount: float
    interest_rate: float
    term_months: int
    monthly_payment: Optional[float] = None
    purpose: Optional[str] = None
    status: LoanStatusEnum
    application_date: datetime
    start_date: Optional[datetime] = None
    next_payment_date: Optional[datetime] = None
    remaining_balance: Optional[float] = None
    collateral_description: Optional[str] = None
    repayment_schedule: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SyncRequestDTO(BaseModel):
    """DTO for sync requests"""
    entity_type: str  # 'user', 'loan', etc.
    entity_id: str
    action: str  # 'create', 'update', 'delete'
    data: dict
    client_timestamp: datetime
    version: int


class SyncResponseDTO(BaseModel):
    """DTO for sync responses"""
    success: bool
    entity_type: str
    entity_id: str
    server_id: Optional[str] = None
    conflict: bool = False
    conflict_data: Optional[dict] = None
    error_message: Optional[str] = None
    server_timestamp: datetime


class BatchSyncRequestDTO(BaseModel):
    """DTO for batch sync requests"""
    requests: List[SyncRequestDTO]
    client_id: str
    batch_timestamp: datetime


class BatchSyncResponseDTO(BaseModel):
    """DTO for batch sync responses"""
    responses: List[SyncResponseDTO]
    batch_id: str
    server_timestamp: datetime
    total_processed: int
    total_successful: int
    total_failed: int


class ConflictResolutionDTO(BaseModel):
    """DTO for conflict resolution"""
    entity_type: str
    entity_id: str
    resolution_strategy: str  # 'client_wins', 'server_wins', 'merge'
    merged_data: Optional[dict] = None
    resolved_by: str  # user ID who resolved the conflict