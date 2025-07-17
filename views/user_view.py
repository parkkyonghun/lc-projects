from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from controllers.user_controller import (
    create_user_controller,
    get_users_controller,
    get_user_controller,
)
from models.branch import Branch
from models.user import User
from schemas.user import BranchSchema, UserCreate, UserSchema
from services.application_repository import ApplicationRepository
from services.websocket_manager import notification_manager
from .dependencies import get_current_user, get_db, get_repository

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserSchema)
async def get_me(
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """Get current user information with sync status"""
    branch = None
    if current_user.branchId:
        result = await db.execute(
            select(Branch).where(Branch.id == current_user.branchId)
        )
        branch_obj = result.scalar_one_or_none()
        if branch_obj:
            branch = BranchSchema.model_validate(branch_obj, from_attributes=True)
    
    user_data = UserSchema.model_validate(current_user).dict()
    user_data["branch"] = branch
    user_data["sync_status"] = current_user.sync_status.value if current_user.sync_status else None
    user_data["last_synced_at"] = current_user.last_synced_at.isoformat() if current_user.last_synced_at else None
    return user_data


@router.get("/", response_model=List[UserSchema])
async def get_users(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get all users with sync status information"""
    users = await repository.get_all_users()
    return [
        {
            **UserSchema.model_validate(user).dict(),
            "sync_status": user.sync_status.value if user.sync_status else None,
            "last_synced_at": user.last_synced_at.isoformat() if user.last_synced_at else None
        }
        for user in users
    ]


@router.post("/", response_model=UserSchema)
async def create_user(
    user_data: UserCreate, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Create a new user with automatic sync queuing"""
    user = await repository.create_user(user_data.dict())
    
    # Send real-time notification
    await notification_manager.send_new_application_notification(
        user_id=current_user.id,
        application_type="user",
        application_id=user.id,
        customer_name=user.english_name
    )
    
    return {
        **UserSchema.model_validate(user).dict(),
        "sync_status": user.sync_status.value if user.sync_status else None,
        "last_synced_at": user.last_synced_at.isoformat() if user.last_synced_at else None
    }


@router.get("/{user_id}", response_model=UserSchema)
async def get_user(
    user_id: str, 
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get a specific user by ID"""
    user = await repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        **UserSchema.model_validate(user).dict(),
        "sync_status": user.sync_status.value if user.sync_status else None,
        "last_synced_at": user.last_synced_at.isoformat() if user.last_synced_at else None
    }


@router.put("/{user_id}", response_model=UserSchema)
async def update_user(
    user_id: str,
    user_data: UserCreate,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Update a user with automatic sync queuing"""
    user = await repository.update_user(user_id, user_data.dict())
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Send real-time notification for update
    await notification_manager.send_loan_status_change_notification(
        user_id=current_user.id,
        loan_id=user_id,
        old_status="updated",
        new_status="updated",
        customer_name=user.english_name
    )
    
    return {
        **UserSchema.model_validate(user).dict(),
        "sync_status": user.sync_status.value if user.sync_status else None,
        "last_synced_at": user.last_synced_at.isoformat() if user.last_synced_at else None
    }


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Soft delete a user with automatic sync queuing"""
    success = await repository.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}
