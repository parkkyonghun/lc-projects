from pydantic import BaseModel
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loan import LoanSchema


class DashboardSchema(BaseModel):
    totalLoans: int
    activeLoans: int
    totalCustomers: int
    totalAmount: float
    recentLoans: List["LoanSchema"]
    paymentAlerts: List["LoanSchema"]
    
    model_config = {"from_attributes": True}
