# FastAPI Project Setup

This project uses FastAPI, SQLAlchemy (async), and other modern Python libraries.

## Setup Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd lc-projects
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

The main entry point is `main.py`. To start the FastAPI server, run:

```bash
uvicorn main:app --reload
```

- The API will be available at: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

## Project Structure

- `main.py` - Application entry point
- `controllers/` - Route controllers
- `models/` - Database models
- `schemas/` - Pydantic schemas
- `views/` - (If used) View logic
- `tests/` - Test cases

## Environment Variables
If your project uses environment variables, create a `.env` file in the root directory and add your variables there.

---
If you have any questions or need additional setup, let me know!
