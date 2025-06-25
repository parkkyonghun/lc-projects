from .user import UserSchema
from .loan import LoanSchema

UserSchema.model_rebuild()
LoanSchema.model_rebuild()
