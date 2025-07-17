#!/usr/bin/env python3
"""
Setup script for the Loan Management API with sync integration.
This script helps configure the environment and initialize the application.
"""

import asyncio
import os
import sys
from pathlib import Path
import subprocess
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_step(step_name: str):
    """Print a formatted step name"""
    print(f"\n{'='*50}")
    print(f"STEP: {step_name}")
    print(f"{'='*50}")


def print_success(message: str):
    """Print a success message"""
    print(f"✓ {message}")


def print_error(message: str):
    """Print an error message"""
    print(f"✗ {message}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"⚠ {message}")


def check_python_version():
    """Check if Python version is compatible"""
    print_step("Checking Python Version")
    
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        return False
    
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing Dependencies")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def create_env_file():
    """Create .env file with default configuration"""
    print_step("Creating Environment Configuration")
    
    env_file = project_root / ".env"
    
    if env_file.exists():
        print_warning(".env file already exists, skipping creation")
        return True
    
    env_content = """# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/loan_management

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_TITLE=Loan Management API
API_DESCRIPTION=API for managing loans with real-time sync capabilities
API_VERSION=1.0.0

# Sync Configuration
SYNC_BATCH_SIZE=50
SYNC_RETRY_ATTEMPTS=3
SYNC_RETRY_DELAY=5
SYNC_TIMEOUT=30

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=pdf,doc,docx,jpg,jpeg,png

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=1000

# Security Configuration
BCRYPT_ROUNDS=12
SESSION_TIMEOUT=3600
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print_success(".env file created")
        print_warning("Please update the .env file with your actual configuration values")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


async def test_redis_connection():
    """Test Redis connection"""
    print_step("Testing Redis Connection")
    
    try:
        redis_client = redis.from_url("redis://localhost:6379/0")
        await redis_client.ping()
        await redis_client.close()
        print_success("Redis connection successful")
        return True
    except Exception as e:
        print_error(f"Redis connection failed: {e}")
        print_warning("Make sure Redis is installed and running on localhost:6379")
        return False


async def test_database_connection():
    """Test database connection"""
    print_step("Testing Database Connection")
    
    # This is a basic test - in production you'd use the actual DATABASE_URL from settings
    try:
        # For demo purposes, we'll just check if the URL format is valid
        print_warning("Database connection test skipped - please configure DATABASE_URL in .env")
        print_warning("Run the migration script after configuring your database")
        return True
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print_step("Creating Directories")
    
    directories = [
        "logs",
        "uploads",
        "temp",
        "backups"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        try:
            dir_path.mkdir(exist_ok=True)
            print_success(f"Directory created: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory {directory}: {e}")
            return False
    
    return True


def print_next_steps():
    """Print next steps for the user"""
    print_step("Setup Complete - Next Steps")
    
    print("""
1. Configure your environment:
   - Update the .env file with your actual database and Redis URLs
   - Set a secure JWT_SECRET_KEY
   - Configure other settings as needed

2. Set up your database:
   - Create a PostgreSQL database
   - Run the migration script: python migrations/add_sync_fields.py

3. Start Redis server:
   - Make sure Redis is running on localhost:6379
   - Or update REDIS_URL in .env to point to your Redis instance

4. Run the application:
   - Development: uvicorn main:app --reload
   - Production: uvicorn main:app --host 0.0.0.0 --port 8000

5. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Root Endpoint: http://localhost:8000/

6. Test the features:
   - Create users and loans
   - Test WebSocket connections
   - Monitor sync status
   - Use the dashboard endpoints

For more information, check the documentation in the docs/ directory.
    """)


async def main():
    """Main setup function"""
    print("🚀 Loan Management API Setup")
    print("This script will help you set up the application environment.")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create environment file
    if not create_env_file():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Test connections (optional)
    await test_redis_connection()
    await test_database_connection()
    
    # Print next steps
    print_next_steps()
    
    print_success("Setup completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)