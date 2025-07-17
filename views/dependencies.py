from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import AsyncGenerator
from models.user import User
from services.auth_manager import auth_manager
from services.application_repository import get_application_repository, ApplicationRepository
from config.settings import settings

<<<<<<< HEAD
# Database setup
engine = create_async_engine(settings.database_url, echo=True)
=======
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:FkiecpnHecHqVUHRDxEhqssqtReDzPau@shinkansen.proxy.rlwy.net:49713/dblc_opd_daily",
)

engine = create_async_engine(DATABASE_URL, echo=True)
>>>>>>> 206bdf9ddbe6e66e57ff16327692889b7c787595
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Security
security = HTTPBearer()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with async_session() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    return await auth_manager.get_current_user(db, token)


async def get_repository(db: AsyncSession = Depends(get_db)) -> ApplicationRepository:
    """Get application repository instance"""
    return get_application_repository(db)


# Optional: Admin user dependency
async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify admin privileges"""
    # Add admin check logic here if needed
    # For now, return the current user
    return current_user
