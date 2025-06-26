import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database settings
    database_url: str = os.getenv(
        'DATABASE_URL', 
        'postgresql+asyncpg://postgres:FkiecpnHecHqVUHRDxEhqssqtReDzPau@shinkansen.proxy.rlwy.net:49713/dblc_opd_daily'
    )
    
    # JWT settings
    jwt_secret_key: str = os.getenv('JWT_SECRET_KEY', 'supersecretkey')
    jwt_algorithm: str = 'HS256'
    jwt_access_token_expire_minutes: int = 30
    
    # Security settings
    bcrypt_rounds: int = 12
    
    # OCR settings
    tesseract_cmd: Optional[str] = os.getenv('TESSERACT_CMD')
    
    # Redis settings (for caching and background tasks)
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Email settings
    smtp_server: Optional[str] = os.getenv('SMTP_SERVER')
    smtp_port: int = int(os.getenv('SMTP_PORT', '587'))
    smtp_username: Optional[str] = os.getenv('SMTP_USERNAME')
    smtp_password: Optional[str] = os.getenv('SMTP_PASSWORD')
    
    # Payment gateway settings
    momo_api_key: Optional[str] = os.getenv('MOMO_API_KEY')
    zalopay_api_key: Optional[str] = os.getenv('ZALOPAY_API_KEY')
    
    # Application settings
    app_name: str = "Khmer Loan Management System"
    app_version: str = "1.0.0"
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = ["image/jpeg", "image/png", "image/tiff", "application/pdf"]
    upload_directory: str = "uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()