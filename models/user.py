from sqlalchemy import String, Column, DateTime, Float, Boolean, Integer
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa
from enum import Enum

Base = declarative_base()


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class User(Base):
    __tablename__ = "User"
    id = Column(String, primary_key=True)
    khmer_name = Column(String, nullable=False)
    english_name = Column(String, nullable=False, unique=True, index=True)
    id_card_number = Column(String, nullable=False, unique=True)
    phone_number = Column(String, nullable=False, unique=True)
    address = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    monthly_income = Column(Float, nullable=True)
    id_card_photo_url = Column(String, nullable=True)
    profile_photo_url = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
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

    loans = relationship("Loan", back_populates="user")
