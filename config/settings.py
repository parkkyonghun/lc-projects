from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql+asyncpg://dbmasteruser:123456@localhost:5432/dblc_opd_daily"
    
    # JWT settings
    jwt_secret_key: str = "supersecretkey"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Redis settings for caching and real-time features
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # API settings
    api_title: str = "LC Work Flow API"
    api_version: str = "1.0.0"
    api_description: str = "Loan application management system with offline/online sync"
    
    # Sync settings
    sync_batch_size: int = 50
    sync_retry_attempts: int = 3
    sync_retry_delay: int = 5  # seconds
    sync_timeout: int = 30  # seconds
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = ["jpg", "jpeg", "png", "pdf", "doc", "docx"]
    upload_path: str = "uploads"
    
    # WebSocket settings
    websocket_heartbeat_interval: int = 30  # seconds
    
    # Security settings
    cors_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Application settings
    debug: bool = False
    default_page_size: int = 20
    max_page_size: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()