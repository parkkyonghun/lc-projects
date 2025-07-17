from sqlalchemy import Column, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from .base import Base


class Branch(Base):
    __tablename__ = "branches"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Location information
    address = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True, default="Cambodia")
    postal_code = Column(String(20), nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    
    # Hierarchy
    parent_id = Column(String, ForeignKey('branches.id'), nullable=True)
    
    # Relationships
    parent = relationship("Branch", remote_side=[id], back_populates="children", uselist=False)
    children = relationship("Branch", back_populates="parent")
    users = relationship("User", back_populates="branch")
    loans = relationship("Loan", back_populates="branch")
    
    def __repr__(self) -> str:
        return f"<Branch {self.code}: {self.name}>"
