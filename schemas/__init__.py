from .user import UserSchema, UserCreate, UserUpdate, BranchSchema, LoginRequest
from .loan import LoanSchema, LoanCreate, LoanUpdate
from .payment import PaymentSchema, PaymentCreate, PaymentUpdate
from .dashboard import DashboardSchema

# Rebuild models to resolve forward references
UserSchema.model_rebuild()
LoanSchema.model_rebuild()
PaymentSchema.model_rebuild()
DashboardSchema.model_rebuild()

__all__ = [
    "UserSchema",
    "UserCreate", 
    "UserUpdate",
    "BranchSchema",
    "LoginRequest",
    "LoanSchema",
    "LoanCreate",
    "LoanUpdate", 
    "PaymentSchema",
    "PaymentCreate",
    "PaymentUpdate",
    "DashboardSchema"
]