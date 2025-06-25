from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from controllers.dashboard_controller import get_dashboard_stats_controller
from schemas.dashboard import DashboardSchema
from .dependencies import get_db, get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardSchema)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)
):
    return await get_dashboard_stats_controller(db)
