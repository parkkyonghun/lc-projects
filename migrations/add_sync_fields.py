"""
Database migration script to add sync-related fields to User and Loan tables.
This script should be run after updating the models to include sync functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import settings


async def run_migration():
    """Run the database migration to add sync fields"""
    
    # Create async engine
    engine = create_async_engine(settings.database_url, echo=True)
    
    # Create async session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        try:
            print("Starting database migration...")
            
            # Add sync fields to users table
            print("Adding sync fields to users table...")
            await session.execute(text("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS server_id VARCHAR(255),
                ADD COLUMN IF NOT EXISTS sync_status VARCHAR(20) DEFAULT 'pending',
                ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
                ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS sync_retry_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS sync_error_message TEXT;
            """))
            
            # Add sync fields to loans table
            print("Adding sync fields to loans table...")
            await session.execute(text("""
                ALTER TABLE loans 
                ADD COLUMN IF NOT EXISTS server_id VARCHAR(255),
                ADD COLUMN IF NOT EXISTS sync_status VARCHAR(20) DEFAULT 'pending',
                ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
                ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS sync_retry_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS sync_error_message TEXT;
            """))
            
            # Create indexes for better performance
            print("Creating indexes for sync fields...")
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_users_sync_status ON users(sync_status)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_users_server_id ON users(server_id)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_users_last_synced ON users(last_synced_at)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_users_is_deleted ON users(is_deleted)"))

            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_loans_sync_status ON loans(sync_status)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_loans_server_id ON loans(server_id)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_loans_last_synced ON loans(last_synced_at)"))
            await session.execute(text("CREATE INDEX IF NOT EXISTS idx_loans_is_deleted ON loans(is_deleted)"))
            
            # Commit the changes
            await session.commit()
            print("✓ Migration completed successfully!")
            
        except Exception as e:
            print(f"✗ Migration failed: {e}")
            await session.rollback()
            raise
        
        finally:
            await engine.dispose()


async def rollback_migration():
    """Rollback the migration (remove sync fields)"""
    
    # Create async engine
    engine = create_async_engine(settings.database_url, echo=True)
    
    # Create async session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        try:
            print("Starting migration rollback...")
            
            # Remove indexes
            print("Removing indexes...")
            await session.execute(text("DROP INDEX IF EXISTS idx_users_sync_status"))
            await session.execute(text("DROP INDEX IF EXISTS idx_users_server_id"))
            await session.execute(text("DROP INDEX IF EXISTS idx_users_last_synced"))
            await session.execute(text("DROP INDEX IF EXISTS idx_users_is_deleted"))
            await session.execute(text("DROP INDEX IF EXISTS idx_loans_sync_status"))
            await session.execute(text("DROP INDEX IF EXISTS idx_loans_server_id"))
            await session.execute(text("DROP INDEX IF EXISTS idx_loans_last_synced"))
            await session.execute(text("DROP INDEX IF EXISTS idx_loans_is_deleted"))
            
            # Remove sync fields from users table
            print("Removing sync fields from users table...")
            await session.execute(text("""
                ALTER TABLE users 
                DROP COLUMN IF EXISTS server_id,
                DROP COLUMN IF EXISTS sync_status,
                DROP COLUMN IF EXISTS last_synced_at,
                DROP COLUMN IF EXISTS version,
                DROP COLUMN IF EXISTS is_deleted,
                DROP COLUMN IF EXISTS sync_retry_count,
                DROP COLUMN IF EXISTS sync_error_message;
            """))
            
            # Remove sync fields from loans table
            print("Removing sync fields from loans table...")
            await session.execute(text("""
                ALTER TABLE loans 
                DROP COLUMN IF EXISTS server_id,
                DROP COLUMN IF EXISTS sync_status,
                DROP COLUMN IF EXISTS last_synced_at,
                DROP COLUMN IF EXISTS version,
                DROP COLUMN IF EXISTS is_deleted,
                DROP COLUMN IF EXISTS sync_retry_count,
                DROP COLUMN IF EXISTS sync_error_message;
            """))
            
            # Commit the changes
            await session.commit()
            print("✓ Migration rollback completed successfully!")
            
        except Exception as e:
            print(f"✗ Migration rollback failed: {e}")
            await session.rollback()
            raise
        
        finally:
            await engine.dispose()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration for sync fields")
    parser.add_argument(
        "--rollback", 
        action="store_true", 
        help="Rollback the migration (remove sync fields)"
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        asyncio.run(rollback_migration())
    else:
        asyncio.run(run_migration())