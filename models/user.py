from sqlalchemy import String, Column, DateTime, Float
from sqlalchemy.orm import declarative_base, relationship
import sqlalchemy as sa

Base = declarative_base()


class User(Base):
    __tablename__ = "User"
    id = Column(String, primary_key=True)
    khmer_name = Column(String, nullable=False)
    english_name = Column(String, nullable=False, unique=True, index=True)
    id_card_number = Column(String, nullable=False, unique=True)
    phone_number = Column(String, nullable=False, unique=True)
    address = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    monthly_income = Column(Float, nullable=True)
    id_card_photo_url = Column(String, nullable=True)
    profile_photo_url = Column(String, nullable=True)

    loans = relationship("Loan", back_populates="user")
    hashed_password = Column(String, nullable=False)
    createdAt = Column(DateTime, server_default=sa.func.now())
    updatedAt = Column(DateTime, server_default=sa.func.now(), onupdate=sa.func.now())

    loans = relationship("Loan", back_populates="user")
