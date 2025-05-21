from sqlalchemy import String, Boolean, JSON, Column, DateTime, Integer
from sqlalchemy.orm import declarative_base
import sqlalchemy as sa

Base = declarative_base()

class User(Base):
    __tablename__ = "User"
    id = Column(String, primary_key=True)  # Changed from UUID to String
    email = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    branchId = Column(String, nullable=True)  # Changed from UUID to String
    username = Column(String, nullable=True)
    image = Column(String, nullable=True)
    preferences = Column(JSON, nullable=True)
    isActive = Column(Boolean, default=True)
    lastLogin = Column(DateTime, nullable=True)
    createdAt = Column(DateTime, server_default=sa.func.now())
    updatedAt = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    failedLoginAttempts = Column(Integer, default=0)
    lockedUntil = Column(DateTime, nullable=True)
