from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from views.user_view import router as user_router
from views.ocr_view import router as ocr_router
from api.training_endpoints import router as training_router
from api.smart_training_ui import router as ui_router

# Create FastAPI app with metadata
app = FastAPI(
    title="AI OCR Training System",
    description="Advanced AI OCR system with training capabilities for Cambodian ID cards",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Flutter integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(user_router)
app.include_router(ocr_router)
app.include_router(training_router)
app.include_router(ui_router)

@app.get("/")
def root():
    return {
        "message": "AI OCR Training System API",
        "version": "2.0.0",
        "features": [
            "Cambodian ID Card OCR",
            "AI Model Training",
            "Flutter Integration",
            "Real-time Training Progress",
            "Smart Training Orchestration"
        ],
        "endpoints": {
            "ocr": "/ocr/idcard",
            "training": "/training/session/start",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": "2024-01-27T10:00:00Z",
        "services": {
            "ocr_engine": "active",
            "training_system": "active",
            "database": "active"
        }
    }
