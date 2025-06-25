from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Branch(Base):
    __tablename__ = "Branch"
    id = Column(String, primary_key=True)
    code = Column(String, nullable=False)
    name = Column(String, nullable=False)
    isActive = Column(String, nullable=False)  # Consider Boolean if DB supports
    createdAt = Column(String, nullable=False)  # Consider DateTime if DB supports
    updatedAt = Column(String, nullable=False)
    parentId = Column(String, nullable=True)
