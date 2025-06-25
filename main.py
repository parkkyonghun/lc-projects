from fastapi import FastAPI
from views.user_view import router as user_router
from views.loan_view import router as loan_router

app = FastAPI()

app.include_router(user_router)
app.include_router(loan_router)


# Optionally, add root endpoint
def root():
    return {"message": "Welcome to the FastAPI MVC project!"}

app.get("/")(root)
