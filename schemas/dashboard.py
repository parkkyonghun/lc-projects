from pydantic import BaseModel
from typing import List
from .loan import LoanSchema


class DashboardSchema(BaseModel):
    totalLoans: int
    activeLoans: int
    totalCustomers: int
    totalAmount: float
    recentLoans: List[LoanSchema]
    paymentAlerts: List[LoanSchema]
