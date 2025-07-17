from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from fastapi import HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import redis.asyncio as redis
from config.settings import settings
from models.user import User
import uuid


class AuthManager:
    """Handles authentication, token management, and user sessions"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = None
        
    async def init_redis(self):
        """Initialize Redis connection for token storage"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a new access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a new refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    
    async def authenticate_user(self, db: AsyncSession, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        result = await db.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    async def create_user_tokens(self, user: User) -> Dict[str, str]:
        """Create access and refresh tokens for a user"""
        await self.init_redis()
        
        # Create tokens
        access_token = self.create_access_token({"sub": user.id, "username": user.username})
        refresh_token = self.create_refresh_token({"sub": user.id, "username": user.username})
        
        # Store refresh token in Redis with expiration
        refresh_token_key = f"refresh_token:{user.id}"
        await self.redis_client.setex(
            refresh_token_key,
            timedelta(days=settings.jwt_refresh_token_expire_days),
            refresh_token
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    async def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token, 
                settings.jwt_secret_key, 
                algorithms=[settings.jwt_algorithm]
            )
            
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return payload
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh an access token using a refresh token"""
        await self.init_redis()
        
        # Verify refresh token
        payload = await self.verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        # Check if refresh token exists in Redis
        refresh_token_key = f"refresh_token:{user_id}"
        stored_token = await self.redis_client.get(refresh_token_key)
        
        if not stored_token or stored_token != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        access_token = self.create_access_token({
            "sub": user_id,
            "username": payload.get("username")
        })
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    async def revoke_token(self, user_id: str):
        """Revoke all tokens for a user"""
        await self.init_redis()
        
        # Remove refresh token from Redis
        refresh_token_key = f"refresh_token:{user_id}"
        await self.redis_client.delete(refresh_token_key)
        
        # Add user to blacklist (optional - for immediate access token invalidation)
        blacklist_key = f"blacklist:{user_id}"
        await self.redis_client.setex(
            blacklist_key,
            timedelta(minutes=settings.jwt_access_token_expire_minutes),
            "revoked"
        )
    
    async def is_token_blacklisted(self, user_id: str) -> bool:
        """Check if a user's tokens are blacklisted"""
        await self.init_redis()
        
        blacklist_key = f"blacklist:{user_id}"
        return await self.redis_client.exists(blacklist_key)
    
    async def get_current_user(self, db: AsyncSession, token: str) -> User:
        """Get current user from token"""
        payload = await self.verify_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Check if token is blacklisted
        if await self.is_token_blacklisted(user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        # Get user from database with related data
        from sqlalchemy.orm import selectinload
        result = await db.execute(
            select(User)
            .options(selectinload(User.customer), selectinload(User.loan_officer))
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user


# Global auth manager instance
auth_manager = AuthManager()