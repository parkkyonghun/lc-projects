from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from controllers.user_controller import get_users_controller, get_user_controller, login_user_controller
from schemas.user import UserSchema
from .dependencies import get_db
from fastapi import Body

router = APIRouter(prefix="/users", tags=["users"])

from .dependencies import get_current_user

from schemas.user import LoginRequest

@router.post("/login")
async def login_user(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    username = data.username
    password = data.password
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required.")
    token, err = await login_user_controller(username, password, db)
    if err:
        raise HTTPException(status_code=401, detail=err)
    return {"access_token": token, "token_type": "bearer"}

from models.branch import Branch
from schemas.user import BranchSchema
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

@router.get("/me", response_model=UserSchema)
async def get_me(current_user = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    branch = None
    if current_user.branchId:
        result = await db.execute(select(Branch).where(Branch.id == current_user.branchId))
        branch_obj = result.scalar_one_or_none()
        if branch_obj:
            branch = BranchSchema.model_validate(branch_obj, from_attributes=True)
    user_data = UserSchema.model_validate(current_user).dict()
    user_data["branch"] = branch
    return user_data

@router.get("/", response_model=List[UserSchema])
async def get_users(db: AsyncSession = Depends(get_db)):
    return await get_users_controller(db)

@router.get("/{user_id}", response_model=UserSchema)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    user = await get_user_controller(user_id, db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
