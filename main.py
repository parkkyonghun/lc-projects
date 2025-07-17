from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as redis

from config.settings import settings
from views.user_view import router as user_router
from views.loan_view import router as loan_router
from views.dashboard_view import router as dashboard_router
from views.payment_view import router as payment_router
from views.auth_view import router as auth_router
from views.sync_view import router as sync_router
from views.websocket_view import router as websocket_router
from services.connectivity_monitor import connectivity_monitor
from services.websocket_manager import connection_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Startup
    print("Starting up application...")
    
    # Initialize Redis connection
    try:
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        print("✓ Redis connection established")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
    
    # Start connectivity monitoring
    try:
        await connectivity_monitor.start_monitoring()
        print("✓ Connectivity monitoring started")
    except Exception as e:
        print(f"✗ Connectivity monitoring failed: {e}")
    
    print("Application startup complete!")
    
    yield
    
    # Shutdown
    print("Shutting down application...")
    
    # Stop connectivity monitoring
    try:
        await connectivity_monitor.stop_monitoring()
        print("✓ Connectivity monitoring stopped")
    except Exception as e:
        print(f"✗ Error stopping connectivity monitoring: {e}")
    
    # Close Redis connections
    try:
        if 'redis_client' in locals():
            await redis_client.close()
        print("✓ Redis connections closed")
    except Exception as e:
        print(f"✗ Error closing Redis connections: {e}")
    
    print("Application shutdown complete!")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(loan_router)
app.include_router(dashboard_router)
app.include_router(payment_router)
app.include_router(sync_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Loan Management API!",
        "version": settings.api_version,
        "features": [
            "User Management",
            "Loan Processing",
            "Payment Tracking",
            "Real-time Sync",
            "WebSocket Notifications",
            "Dashboard Analytics"
        ],
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        redis_status = "healthy"
        await redis_client.close()
    except Exception:
        redis_status = "unhealthy"
    
    # Check connectivity monitor
    network_status = await connectivity_monitor.get_network_status()
    
    return {
        "status": "healthy",
        "services": {
            "redis": redis_status,
            "network": "online" if network_status.get("is_online") else "offline",
            "websocket_connections": len(connection_manager.active_connections)
        },
        "timestamp": network_status.get("last_check")
    }
