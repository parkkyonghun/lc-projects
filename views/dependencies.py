from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.user import User
from sqlalchemy.future import select
from core.database import get_db
from core.security import verify_token
from core.config import settings

security = HTTPBearer()

async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    """Get current authenticated user from JWT token"""
    credentials: HTTPAuthorizationCredentials = await security(request)
    token = credentials.credentials
    
    # Verify token using centralized security utility
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or expired token"
        )
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid token payload"
        )
    
    # Get user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="User not found"
        )
    
    return user
