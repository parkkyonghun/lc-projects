from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views.user_view import router as user_router
from views.ocr_view import router as ocr_router
from views.dashboard_view import router as dashboard_router
from views.loan_view import router as loan_router
from views.payment_view import router as payment_router

app = FastAPI(
    title="Khmer Loan Management System",
    description="A comprehensive loan management system with Khmer language support and ID scanning",
    version="1.0.0"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(user_router)
app.include_router(ocr_router)
app.include_router(dashboard_router)
app.include_router(loan_router)
app.include_router(payment_router)

# Optionally, add root endpoint
def root():
    return {"message": "Welcome to the FastAPI MVC project!"}

app.get("/")(root)
