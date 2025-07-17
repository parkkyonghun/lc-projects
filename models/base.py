from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, Boolean, func
from typing import Any

class BaseModel:
    """Base model class that provides common fields and methods."""
    
    # Common fields
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

# Create a base class for SQLAlchemy models
Base = declarative_base(cls=BaseModel)
