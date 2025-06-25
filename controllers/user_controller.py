from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from models.user import User
from schemas.user import UserSchema
from typing import List, Optional
import bcrypt
from jose import jwt
from datetime import datetime, timedelta
import os

# JWT settings
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'supersecretkey')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

async def get_users_controller(db: AsyncSession) -> List[UserSchema]:
    result = await db.execute(select(User).options(selectinload(User.loans)))
    users = result.scalars().all()
    return [UserSchema.model_validate(u) for u in users]

async def login_user_controller(username: str, password: str, db: AsyncSession):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None, 'Incorrect username or password.'
    if not user.isActive:
        return None, 'User account is inactive.'
    # Issue JWT
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {'sub': user.id, 'exp': expire}
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token, None

async def get_user_controller(user_id: str, db: AsyncSession) -> Optional[UserSchema]:
    result = await db.execute(select(User).where(User.id == user_id).options(selectinload(User.loans)))
    user = result.scalar_one_or_none()
    if user:
        return UserSchema.model_validate(user)
    return None
