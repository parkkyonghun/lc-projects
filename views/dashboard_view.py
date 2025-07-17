from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from controllers.dashboard_controller import get_dashboard_stats_controller
from models.user import User
from schemas.dashboard import DashboardSchema
from services.application_repository import ApplicationRepository
from services.connectivity_monitor import connectivity_monitor
from .dependencies import get_db, get_current_user, get_repository

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardSchema)
async def get_dashboard_stats(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
):
    """Get dashboard statistics with sync status information"""
    db = await repository.get_session()
    stats = await get_dashboard_stats_controller(db)
    
    # Add sync status information
    sync_stats = await repository.get_sync_status_counts()
    
    # Enhance stats with sync information
    stats_dict = stats.dict() if hasattr(stats, 'dict') else stats
    enhanced_stats = {
        **stats_dict,
        "sync_status": {
            "pending_users": sync_stats.get("pending_users", 0),
            "pending_loans": sync_stats.get("pending_loans", 0),
            "failed_users": sync_stats.get("failed_users", 0),
            "failed_loans": sync_stats.get("failed_loans", 0),
            "synced_users": sync_stats.get("synced_users", 0),
            "synced_loans": sync_stats.get("synced_loans", 0)
        }
    }
    
    return enhanced_stats


@router.get("/network-status")
async def get_network_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current network connectivity status"""
    status = await connectivity_monitor.get_network_status()
    quality = await connectivity_monitor.get_connection_quality()
    
    return {
        "is_online": status.get("is_online", False),
        "last_check": status.get("last_check"),
        "quality": {
            "latency_ms": quality.get("latency_ms"),
            "bandwidth_mbps": quality.get("bandwidth_mbps"),
            "quality_score": quality.get("quality_score"),
            "recommended_batch_size": quality.get("recommended_batch_size"),
            "recommended_timeout": quality.get("recommended_timeout")
        }
    }


@router.get("/sync-summary")
async def get_sync_summary(
    repository: ApplicationRepository = Depends(get_repository),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get a comprehensive sync summary for the dashboard"""
    sync_stats = await repository.get_sync_status_counts()
    network_status = await connectivity_monitor.get_network_status()
    
    # Calculate sync health score
    total_items = (
        sync_stats.get("pending_users", 0) + 
        sync_stats.get("pending_loans", 0) + 
        sync_stats.get("synced_users", 0) + 
        sync_stats.get("synced_loans", 0)
    )
    
    synced_items = sync_stats.get("synced_users", 0) + sync_stats.get("synced_loans", 0)
    failed_items = sync_stats.get("failed_users", 0) + sync_stats.get("failed_loans", 0)
    
    sync_health_score = 0
    if total_items > 0:
        sync_health_score = (synced_items / total_items) * 100
        if failed_items > 0:
            sync_health_score -= (failed_items / total_items) * 20  # Penalty for failures
    
    return {
        "sync_health_score": round(sync_health_score, 2),
        "total_items": total_items,
        "synced_items": synced_items,
        "pending_items": sync_stats.get("pending_users", 0) + sync_stats.get("pending_loans", 0),
        "failed_items": failed_items,
        "network_online": network_status.get("is_online", False),
        "last_sync_check": network_status.get("last_check"),
        "sync_recommendations": {
            "should_sync": await connectivity_monitor.should_sync(),
            "recommended_batch_size": (await connectivity_monitor.get_connection_quality()).get("recommended_batch_size", 10)
        }
    }
