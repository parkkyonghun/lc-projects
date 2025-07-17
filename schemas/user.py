from pydantic import BaseModel, model_validator, Field
from typing import Optional, List, Any, Dict

from datetime import datetime


class BranchSchema(BaseModel):
    id: str
    code: str
    name: str
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
    parentId: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserSchema(BaseModel):
    id: str
    username: str
    email: str
    phone_number: str
    user_type: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    # Customer-specific fields (will be None for non-customers)
    khmer_name: Optional[str] = None
    english_name: Optional[str] = None
    id_card_number: Optional[str] = None
    address: Optional[str] = None
    occupation: Optional[str] = None
    monthly_income: Optional[float] = None
    id_card_photo_url: Optional[str] = None
    profile_photo_url: Optional[str] = None
    
    # Relationships
    loans: List["LoanSchema"] = []
    branch: Optional[BranchSchema] = None

    model_config = {"from_attributes": True}
    
    @model_validator(mode='after')
    def extract_customer_fields(self) -> 'UserSchema':
        # If this is a customer, extract the customer fields
        if hasattr(self, 'customer') and self.customer:
            customer = self.customer
            # Create a dictionary of customer fields to update
            customer_fields = {
                'khmer_name': getattr(customer, 'khmer_name', None),
                'english_name': getattr(customer, 'english_name', None),
                'id_card_number': getattr(customer, 'id_card_number', None),
                'address': getattr(customer, 'address', None),
                'occupation': getattr(customer, 'occupation', None),
                'monthly_income': getattr(customer, 'monthly_income', None),
                'id_card_photo_url': getattr(customer, 'id_card_photo_url', None),
                'profile_photo_url': getattr(customer, 'profile_photo_url', None)
            }
            # Update the model's fields
            for field, value in customer_fields.items():
                if value is not None:
                    setattr(self, field, value)
        return self


class UserCreate(BaseModel):
    # Base user fields
    username: str
    email: str
    phone_number: str
    password: str
    user_type: str = "customer"  # Default to customer
    
    # Customer-specific fields (only used if user_type is 'customer')
    khmer_name: Optional[str] = None
    english_name: Optional[str] = None
    id_card_number: Optional[str] = None
    address: Optional[str] = None
    occupation: Optional[str] = None
    monthly_income: Optional[float] = None
    id_card_photo_url: Optional[str] = None
    profile_photo_url: Optional[str] = None
