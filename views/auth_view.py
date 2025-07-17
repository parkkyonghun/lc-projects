from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Dict, Any

from views.dependencies import get_db, get_current_user
from services.auth_manager import auth_manager
from models.user import User


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user_id: str
    username: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return tokens"""
    user = await auth_manager.authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = await auth_manager.create_user_tokens(user)
    
    return LoginResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        user_id=user.id,
        username=user.username
    )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        tokens = await auth_manager.refresh_access_token(request.refresh_token)
        return RefreshTokenResponse(
            access_token=tokens["access_token"],
            token_type=tokens["token_type"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user and revoke tokens"""
    await auth_manager.revoke_token(current_user.id)
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    base_info = {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "phone_number": current_user.phone_number,
        "user_type": current_user.user_type,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }
    
    # Add type-specific information
    if current_user.user_type == "customer" and current_user.customer:
        base_info.update({
            "khmer_name": current_user.customer.khmer_name,
            "english_name": current_user.customer.english_name,
            "id_card_number": current_user.customer.id_card_number,
            "address": current_user.customer.address,
            "occupation": current_user.customer.occupation,
            "monthly_income": current_user.customer.monthly_income,
            "sync_status": current_user.customer.sync_status,
            "last_synced_at": current_user.customer.last_synced_at.isoformat() if current_user.customer.last_synced_at else None
        })
    elif current_user.user_type == "loan_officer" and current_user.loan_officer:
        base_info.update({
            "employee_id": current_user.loan_officer.employee_id,
            "hire_date": current_user.loan_officer.hire_date.isoformat() if current_user.loan_officer.hire_date else None,
            "department": current_user.loan_officer.department
        })
    
    return base_info