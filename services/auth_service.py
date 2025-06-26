from typing import Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException, status

from models.user import User, UserRole
from schemas.user import UserCreate, UserLogin
from core.security import hash_password, verify_password, create_access_token, verify_token
from core.config import settings
from services.notification_service import notification_service

logger = logging.getLogger(__name__)

class AuthService:
    """Service for authentication and user management"""
    
    def __init__(self):
        self.token_expire_minutes = settings.access_token_expire_minutes
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
    
    async def register_user(
        self, 
        db: AsyncSession, 
        user_data: UserCreate
    ) -> Tuple[User, str]:
        """Register a new user"""
        
        # Check if user already exists
        existing_user = await self._get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Check if phone number already exists
        if user_data.phone:
            existing_phone = await self._get_user_by_phone(db, user_data.phone)
            if existing_phone:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this phone number already exists"
                )
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            phone=user_data.phone,
            date_of_birth=user_data.date_of_birth,
            address=user_data.address,
            national_id=user_data.national_id,
            role=UserRole.CUSTOMER,  # Default role
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
            branch_id=user_data.branch_id
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id, "role": user.role.value}
        )
        
        # Send welcome email
        try:
            await notification_service.send_welcome_email(
                user.email,
                user.full_name
            )
        except Exception as e:
            logger.error(f"Failed to send welcome email: {str(e)}")
        
        logger.info(f"User registered successfully: {user.email}")
        return user, access_token
    
    async def authenticate_user(
        self, 
        db: AsyncSession, 
        login_data: UserLogin
    ) -> Tuple[User, str]:
        """Authenticate user and return user and token"""
        
        # Get user by email
        user = await self._get_user_by_email(db, login_data.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if account is locked
        if await self._is_account_locked(user):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account is locked due to too many failed login attempts. Try again after {self.lockout_duration_minutes} minutes."
            )
        
        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Verify password
        if not verify_password(login_data.password, user.hashed_password):
            # Increment failed login attempts
            await self._increment_failed_attempts(db, user)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Reset failed login attempts on successful login
        await self._reset_failed_attempts(db, user)
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id, "role": user.role.value}
        )
        
        logger.info(f"User authenticated successfully: {user.email}")
        return user, access_token
    
    async def get_current_user(self, db: AsyncSession, token: str) -> User:
        """Get current user from token"""
        
        try:
            # Verify token
            payload = verify_token(token)
            email = payload.get("sub")
            
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Get user
            user = await self._get_user_by_email(db, email)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            return user
            
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def refresh_token(self, db: AsyncSession, token: str) -> str:
        """Refresh access token"""
        
        user = await self.get_current_user(db, token)
        
        # Create new access token
        new_token = create_access_token(
            data={"sub": user.email, "user_id": user.id, "role": user.role.value}
        )
        
        logger.info(f"Token refreshed for user: {user.email}")
        return new_token
    
    async def change_password(
        self, 
        db: AsyncSession, 
        user_id: str, 
        current_password: str, 
        new_password: str
    ) -> bool:
        """Change user password"""
        
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        user.hashed_password = hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Password changed for user: {user.email}")
        return True
    
    async def reset_password_request(self, db: AsyncSession, email: str) -> bool:
        """Request password reset"""
        
        user = await self._get_user_by_email(db, email)
        if not user:
            # Don't reveal if email exists or not
            logger.warning(f"Password reset requested for non-existent email: {email}")
            return True
        
        # Generate reset token (valid for 1 hour)
        reset_token = create_access_token(
            data={"sub": user.email, "type": "password_reset"},
            expires_delta=timedelta(hours=1)
        )
        
        # Store reset token (in practice, you might want to store this in database)
        user.password_reset_token = reset_token
        user.password_reset_requested_at = datetime.utcnow()
        
        await db.commit()
        
        # Send password reset email
        try:
            await notification_service.send_password_reset_email(
                user.email,
                user.full_name,
                reset_token
            )
        except Exception as e:
            logger.error(f"Failed to send password reset email: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send password reset email"
            )
        
        logger.info(f"Password reset requested for user: {user.email}")
        return True
    
    async def reset_password(
        self, 
        db: AsyncSession, 
        token: str, 
        new_password: str
    ) -> bool:
        """Reset password using reset token"""
        
        try:
            # Verify reset token
            payload = verify_token(token)
            email = payload.get("sub")
            token_type = payload.get("type")
            
            if email is None or token_type != "password_reset":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid reset token"
                )
            
            # Get user
            user = await self._get_user_by_email(db, email)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Check if token matches stored token
            if user.password_reset_token != token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired reset token"
                )
            
            # Check if token is not too old (additional security)
            if user.password_reset_requested_at:
                token_age = datetime.utcnow() - user.password_reset_requested_at
                if token_age > timedelta(hours=1):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Reset token has expired"
                    )
            
            # Update password
            user.hashed_password = hash_password(new_password)
            user.password_changed_at = datetime.utcnow()
            user.password_reset_token = None
            user.password_reset_requested_at = None
            
            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.last_failed_login = None
            
            await db.commit()
            
            logger.info(f"Password reset completed for user: {user.email}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Password reset failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
    
    async def verify_user_account(self, db: AsyncSession, user_id: str) -> bool:
        """Verify user account (admin function)"""
        
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user.is_verified = True
        user.verified_at = datetime.utcnow()
        
        await db.commit()
        
        # Send verification confirmation email
        try:
            await notification_service.send_account_verified_email(
                user.email,
                user.full_name
            )
        except Exception as e:
            logger.error(f"Failed to send verification email: {str(e)}")
        
        logger.info(f"User account verified: {user.email}")
        return True
    
    async def deactivate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Deactivate user account"""
        
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user.is_active = False
        user.deactivated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"User account deactivated: {user.email}")
        return True
    
    async def _get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        query = select(User).where(User.email == email)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_user_by_phone(self, db: AsyncSession, phone: str) -> Optional[User]:
        """Get user by phone number"""
        query = select(User).where(User.phone == phone)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def _is_account_locked(self, user: User) -> bool:
        """Check if account is locked due to failed login attempts"""
        if user.failed_login_attempts >= self.max_login_attempts:
            if user.last_failed_login:
                lockout_end = user.last_failed_login + timedelta(minutes=self.lockout_duration_minutes)
                return datetime.utcnow() < lockout_end
        return False
    
    async def _increment_failed_attempts(self, db: AsyncSession, user: User):
        """Increment failed login attempts"""
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        user.last_failed_login = datetime.utcnow()
        await db.commit()
    
    async def _reset_failed_attempts(self, db: AsyncSession, user: User):
        """Reset failed login attempts"""
        user.failed_login_attempts = 0
        user.last_failed_login = None
        await db.commit()
    
    def check_user_permissions(self, user: User, required_role: UserRole) -> bool:
        """Check if user has required permissions"""
        role_hierarchy = {
            UserRole.CUSTOMER: 1,
            UserRole.STAFF: 2,
            UserRole.MANAGER: 3,
            UserRole.ADMIN: 4
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    async def get_user_profile(self, db: AsyncSession, user_id: str) -> Dict[str, any]:
        """Get user profile information"""
        
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name,
            'phone': user.phone,
            'date_of_birth': user.date_of_birth,
            'address': user.address,
            'national_id': user.national_id,
            'role': user.role.value,
            'is_active': user.is_active,
            'is_verified': user.is_verified,
            'created_at': user.created_at,
            'last_login': user.last_login,
            'branch_id': user.branch_id
        }

# Global auth service instance
auth_service = AuthService()