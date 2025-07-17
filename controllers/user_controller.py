from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from models.user import User, Customer, UserType
from schemas.user import UserSchema, UserCreate
from utils.security import get_password_hash
import uuid
from typing import List, Optional
import bcrypt
from jose import jwt
from datetime import datetime, timedelta
import os
from sqlalchemy.exc import IntegrityError

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


async def get_users_controller(db: AsyncSession) -> List[UserSchema]:
    result = await db.execute(select(User).options(selectinload(User.loans)))
    users = result.scalars().all()
    return [UserSchema.model_validate(u) for u in users]


async def login_user_controller(username: str, password: str, db: AsyncSession):
    # First try to find by username (which is in the User model)
    result = await db.execute(select(User).where(User.username == username).options(selectinload(User.customer)))
    user = result.scalar_one_or_none()
    
    # If not found by username, try to find by email
    if not user:
        result = await db.execute(select(User).where(User.email == username).options(selectinload(User.customer)))
        user = result.scalar_one_or_none()
    
    if not user or not bcrypt.checkpw(
        password.encode("utf-8"), user.hashed_password.encode("utf-8")
    ):
        return None, "Incorrect username/email or password."
        
    if not user.is_active:
        return None, "User account is inactive."
        
    # Issue JWT
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": str(user.id), "exp": expire}
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token, None


async def get_user_controller(user_id: str, db: AsyncSession) -> Optional[UserSchema]:
    result = await db.execute(
        select(User).where(User.id == user_id).options(selectinload(User.loans))
    )
    user = result.scalar_one_or_none()
    if user:
        return UserSchema.model_validate(user)
    return None


async def create_user_controller(user_data: UserCreate, db: AsyncSession) -> UserSchema:
    # Prepare user data
    user_data_dict = user_data.model_dump()
    password = user_data_dict.pop("password")
    
    # Extract customer-specific fields
    customer_fields = {}
    for field in ["khmer_name", "english_name", "id_card_number", "address", 
                 "occupation", "monthly_income", "id_card_photo_url", "profile_photo_url"]:
        if field in user_data_dict:
            customer_fields[field] = user_data_dict.pop(field)
    
    # Create user
    try:
        new_user = User(
            id=str(uuid.uuid4()),
            hashed_password=get_password_hash(password),
            user_type=UserType.CUSTOMER,  # Default to customer
            **user_data_dict
        )
        db.add(new_user)
        
        # If this is a customer, create customer record
        if user_data.user_type == UserType.CUSTOMER:
            customer = Customer(
                id=new_user.id,
                **customer_fields
            )
            db.add(customer)
        
        await db.commit()
        await db.refresh(new_user)
        
        # Reload the user with relationships
        result = await db.execute(
            select(User)
            .where(User.id == new_user.id)
            .options(selectinload(User.customer))
        )
        user = result.scalar_one()
        
        return UserSchema.model_validate(user, from_attributes=True)
        
    except IntegrityError as e:
        await db.rollback()
        if "duplicate key value violates unique constraint" in str(e):
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            elif "phone_number" in str(e):
                raise ValueError("Phone number already exists")
        raise
