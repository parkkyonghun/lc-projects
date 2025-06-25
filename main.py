from fastapi import FastAPI
from views.user_view import router as user_router
from views.loan_view import router as loan_router
from views.dashboard_view import router as dashboard_router
from views.payment_view import router as payment_router

app = FastAPI()

app.include_router(user_router)
app.include_router(loan_router)
app.include_router(dashboard_router)
app.include_router(payment_router)


# Optionally, add root endpoint
def root():
    return {"message": "Welcome to the FastAPI MVC project!"}


app.get("/")(root)
