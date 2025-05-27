"""
Training API Endpoints for Flutter Integration

This module provides comprehensive API endpoints for AI model training,
specifically designed for Flutter app integration with camera-based training.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
import json
import asyncio
from datetime import datetime
from pathlib import Path

try:
    from schemas.training import (
        TrainingDataRequest,
        TrainingProgressResponse,
        ModelMetricsResponse,
        TrainingSessionResponse,
        BatchTrainingRequest
    )
    from core.training_manager import TrainingManager
    from core.model_evaluator import ModelEvaluator
except ImportError:
    # Fallback for development - create simple mock classes
    from pydantic import BaseModel
    from datetime import datetime

    class TrainingSessionResponse(BaseModel):
        session_id: str
        status: str
        created_at: datetime
        target_accuracy: float
        max_samples: int
        current_samples: int
        current_accuracy: float

    class TrainingProgressResponse(BaseModel):
        session_id: str
        status: str
        current_samples: int
        target_samples: int
        current_accuracy: float
        target_accuracy: float
        training_epochs_completed: int = 0
        estimated_completion_time: Optional[str] = None
        last_updated: datetime

    class ModelMetricsResponse(BaseModel):
        session_id: str
        accuracy: float
        precision: float
        recall: float
        f1_score: float
        confidence_score: float
        processing_time: float
        field_accuracies: Dict[str, float]
        quality_improvements: Dict[str, float]
        training_history: List[Dict[str, Any]]

    class BatchTrainingRequest(BaseModel):
        ground_truths: List[Dict[str, str]]
        session_id: str

    # Mock training manager and evaluator
    class TrainingManager:
        async def start_session(self, **kwargs):
            return {"session_id": "mock-session-123", "created_at": datetime.now()}

        async def add_training_sample(self, **kwargs):
            return {"sample_id": "mock-sample-123", "should_retrain": False, "session_progress": {}}

        async def get_session_progress(self, session_id):
            return {"session_id": session_id, "status": "active", "current_samples": 5, "target_samples": 100}

        async def list_sessions(self, **kwargs):
            return []

        async def delete_session(self, session_id):
            pass

        async def validate_model_for_deployment(self, session_id):
            return {"ready": True, "reason": ""}

        async def deploy_model(self, session_id, model_name):
            pass

        async def list_available_models(self):
            return []

        async def trigger_retraining(self, session_id):
            return {"success": True}

    class ModelEvaluator:
        async def get_session_metrics(self, session_id):
            return {
                "accuracy": 0.85, "precision": 0.87, "recall": 0.83, "f1_score": 0.85,
                "confidence_score": 0.82, "processing_time": 1.2,
                "field_accuracies": {}, "quality_improvements": {}, "training_history": []
            }

router = APIRouter(prefix="/training", tags=["AI Training"])

# Initialize training components
training_manager = TrainingManager()
model_evaluator = ModelEvaluator()

@router.post("/session/start", response_model=TrainingSessionResponse)
async def start_training_session(
    session_name: str = Query(..., description="Name for this training session"),
    target_accuracy: float = Query(0.95, description="Target accuracy (0.0-1.0)"),
    max_samples: int = Query(100, description="Maximum training samples")
):
    """
    Start a new AI training session for Flutter app.

    This endpoint initializes a training session that can collect data
    from Flutter camera captures and improve the AI model in real-time.
    """
    try:
        session = await training_manager.start_session(
            name=session_name,
            target_accuracy=target_accuracy,
            max_samples=max_samples
        )

        return TrainingSessionResponse(
            session_id=session["session_id"],
            status="active",
            created_at=session["created_at"],
            target_accuracy=target_accuracy,
            max_samples=max_samples,
            current_samples=0,
            current_accuracy=session.get("baseline_accuracy", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training session: {str(e)}")

@router.post("/data/camera", response_model=Dict[str, Any])
async def add_camera_training_data(
    session_id: str = Query(..., description="Training session ID"),
    file: UploadFile = File(..., description="Camera captured image"),
    ground_truth: str = Query(..., description="JSON string of correct field values"),
    image_quality: float = Query(None, description="Image quality score (0.0-1.0)"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Add training data from Flutter camera capture.

    This endpoint receives images captured by the Flutter app camera
    along with ground truth data for training the AI model.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Parse ground truth data
        ground_truth_data = json.loads(ground_truth)

        # Process the training data
        result = await training_manager.add_training_sample(
            session_id=session_id,
            image_file=file,
            ground_truth=ground_truth_data,
            image_quality=image_quality,
            source="flutter_camera"
        )

        # Trigger background training if enough samples collected
        if result["should_retrain"]:
            background_tasks.add_task(
                training_manager.trigger_retraining,
                session_id
            )

        return {
            "success": True,
            "sample_id": result["sample_id"],
            "session_progress": result["session_progress"],
            "training_triggered": result["should_retrain"],
            "estimated_accuracy_improvement": result.get("estimated_improvement", 0.0)
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid ground truth JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process training data: {str(e)}")

@router.post("/data/batch", response_model=Dict[str, Any])
async def add_batch_training_data(
    session_id: str = Query(..., description="Training session ID"),
    files: List[UploadFile] = File(..., description="Multiple images for batch training"),
    batch_request: BatchTrainingRequest = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Add multiple training samples in batch for efficient processing.

    Useful for uploading multiple images at once from Flutter gallery
    or for bulk training data import.
    """
    try:
        results = []

        for i, file in enumerate(files):
            if not file.content_type.startswith("image/"):
                continue

            # Get ground truth for this image if provided
            ground_truth = {}
            if batch_request and i < len(batch_request.ground_truths):
                ground_truth = batch_request.ground_truths[i]

            result = await training_manager.add_training_sample(
                session_id=session_id,
                image_file=file,
                ground_truth=ground_truth,
                source="flutter_batch"
            )
            results.append(result)

        # Trigger retraining after batch processing
        background_tasks.add_task(
            training_manager.trigger_retraining,
            session_id
        )

        return {
            "success": True,
            "processed_count": len(results),
            "sample_ids": [r["sample_id"] for r in results],
            "training_triggered": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch training data: {str(e)}")

@router.get("/progress/{session_id}", response_model=TrainingProgressResponse)
async def get_training_progress(session_id: str):
    """
    Get real-time training progress for Flutter UI updates.

    Returns current training status, accuracy metrics, and progress indicators
    that can be displayed in the Flutter app.
    """
    try:
        progress = await training_manager.get_session_progress(session_id)

        return TrainingProgressResponse(
            session_id=session_id,
            status=progress["status"],
            current_samples=progress["current_samples"],
            target_samples=progress["target_samples"],
            current_accuracy=progress["current_accuracy"],
            target_accuracy=progress["target_accuracy"],
            training_epochs_completed=progress.get("epochs_completed", 0),
            estimated_completion_time=progress.get("estimated_completion", None),
            last_updated=progress["last_updated"]
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Training session not found: {str(e)}")

@router.get("/metrics/{session_id}", response_model=ModelMetricsResponse)
async def get_model_metrics(session_id: str):
    """
    Get detailed model performance metrics for analysis.

    Provides comprehensive metrics that can be displayed in Flutter
    for training analysis and model performance monitoring.
    """
    try:
        metrics = await model_evaluator.get_session_metrics(session_id)

        return ModelMetricsResponse(
            session_id=session_id,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            confidence_score=metrics["confidence_score"],
            processing_time=metrics["processing_time"],
            field_accuracies=metrics["field_accuracies"],
            quality_improvements=metrics["quality_improvements"],
            training_history=metrics["training_history"]
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Metrics not found: {str(e)}")

@router.post("/model/deploy/{session_id}")
async def deploy_trained_model(
    session_id: str,
    model_name: str = Query(..., description="Name for the deployed model"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Deploy a trained model for production use.

    This endpoint allows Flutter app to deploy a successfully trained model
    for use in production OCR processing.
    """
    try:
        # Validate model is ready for deployment
        is_ready = await training_manager.validate_model_for_deployment(session_id)

        if not is_ready["ready"]:
            raise HTTPException(
                status_code=400,
                detail=f"Model not ready for deployment: {is_ready['reason']}"
            )

        # Deploy model in background
        background_tasks.add_task(
            training_manager.deploy_model,
            session_id,
            model_name
        )

        return {
            "success": True,
            "message": "Model deployment initiated",
            "model_name": model_name,
            "estimated_deployment_time": "2-5 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {str(e)}")

@router.get("/sessions", response_model=List[TrainingSessionResponse])
async def list_training_sessions(
    status: Optional[str] = Query(None, description="Filter by status: active, completed, failed"),
    limit: int = Query(10, description="Maximum number of sessions to return")
):
    """
    List all training sessions for Flutter app management.

    Allows Flutter app to display and manage multiple training sessions.
    """
    try:
        sessions = await training_manager.list_sessions(status=status, limit=limit)

        return [
            TrainingSessionResponse(
                session_id=session["session_id"],
                status=session["status"],
                created_at=session["created_at"],
                target_accuracy=session["target_accuracy"],
                max_samples=session["max_samples"],
                current_samples=session["current_samples"],
                current_accuracy=session["current_accuracy"]
            )
            for session in sessions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.delete("/session/{session_id}")
async def delete_training_session(session_id: str):
    """
    Delete a training session and its associated data.

    Allows Flutter app to clean up completed or failed training sessions.
    """
    try:
        await training_manager.delete_session(session_id)

        return {
            "success": True,
            "message": f"Training session {session_id} deleted successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.get("/models/available")
async def list_available_models():
    """
    List all available trained models for Flutter app selection.

    Returns models that can be used for OCR processing in the Flutter app.
    """
    try:
        models = await training_manager.list_available_models()

        return {
            "models": models,
            "default_model": models[0]["model_id"] if models else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
