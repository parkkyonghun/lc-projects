"""
Training API Schemas for Flutter Integration

Pydantic models for training-related API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class TrainingDataRequest(BaseModel):
    """Request model for adding training data."""
    session_id: str = Field(..., description="Training session ID")
    ground_truth: Dict[str, str] = Field(..., description="Correct field values")
    image_quality: Optional[float] = Field(None, description="Image quality score (0.0-1.0)")
    source: str = Field("flutter_camera", description="Source of the training data")

class BatchTrainingRequest(BaseModel):
    """Request model for batch training data."""
    ground_truths: List[Dict[str, str]] = Field(..., description="Ground truth data for each image")
    session_id: str = Field(..., description="Training session ID")

class TrainingSessionResponse(BaseModel):
    """Response model for training session information."""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status: active, completed, failed")
    created_at: datetime = Field(..., description="Session creation timestamp")
    target_accuracy: float = Field(..., description="Target accuracy goal")
    max_samples: int = Field(..., description="Maximum training samples")
    current_samples: int = Field(..., description="Current number of samples")
    current_accuracy: float = Field(..., description="Current model accuracy")

class TrainingProgressResponse(BaseModel):
    """Response model for training progress updates."""
    session_id: str = Field(..., description="Training session ID")
    status: str = Field(..., description="Current training status")
    current_samples: int = Field(..., description="Number of samples collected")
    target_samples: int = Field(..., description="Target number of samples")
    current_accuracy: float = Field(..., description="Current model accuracy")
    target_accuracy: float = Field(..., description="Target accuracy goal")
    training_epochs_completed: int = Field(0, description="Number of training epochs completed")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    last_updated: datetime = Field(..., description="Last update timestamp")

class ModelMetricsResponse(BaseModel):
    """Response model for detailed model performance metrics."""
    session_id: str = Field(..., description="Training session ID")
    accuracy: float = Field(..., description="Overall accuracy")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    confidence_score: float = Field(..., description="Average confidence score")
    processing_time: float = Field(..., description="Average processing time (seconds)")
    field_accuracies: Dict[str, float] = Field(..., description="Accuracy per field")
    quality_improvements: Dict[str, float] = Field(..., description="Quality improvement metrics")
    training_history: List[Dict[str, Any]] = Field(..., description="Training history data")

class SmartTrainingConfig(BaseModel):
    """Configuration for smart training features."""
    auto_quality_assessment: bool = Field(True, description="Automatically assess image quality")
    active_learning: bool = Field(True, description="Use active learning for sample selection")
    real_time_feedback: bool = Field(True, description="Provide real-time training feedback")
    quality_threshold: float = Field(0.3, description="Minimum quality threshold for training")
    confidence_threshold: float = Field(0.8, description="Confidence threshold for auto-labeling")

class CameraTrainingRequest(BaseModel):
    """Request model for camera-based training."""
    session_id: str = Field(..., description="Training session ID")
    capture_settings: Dict[str, Any] = Field({}, description="Camera capture settings")
    auto_enhance: bool = Field(True, description="Automatically enhance captured images")
    quality_check: bool = Field(True, description="Perform quality check before training")

class TrainingFeedback(BaseModel):
    """Model for training feedback and suggestions."""
    sample_id: str = Field(..., description="Training sample ID")
    feedback_type: str = Field(..., description="Type of feedback: suggestion, correction, validation")
    message: str = Field(..., description="Feedback message")
    suggested_improvements: List[str] = Field([], description="Suggested improvements")
    confidence: float = Field(..., description="Confidence in the feedback")

class ModelDeploymentRequest(BaseModel):
    """Request model for model deployment."""
    session_id: str = Field(..., description="Training session ID")
    model_name: str = Field(..., description="Name for the deployed model")
    deployment_target: str = Field("production", description="Deployment target environment")
    auto_backup: bool = Field(True, description="Automatically backup current model")

class TrainingAnalytics(BaseModel):
    """Analytics data for training performance."""
    total_sessions: int = Field(..., description="Total number of training sessions")
    successful_sessions: int = Field(..., description="Number of successful sessions")
    average_accuracy_improvement: float = Field(..., description="Average accuracy improvement")
    total_training_samples: int = Field(..., description="Total training samples collected")
    average_training_time: float = Field(..., description="Average training time (minutes)")
    quality_distribution: Dict[str, int] = Field(..., description="Distribution of image qualities")

class RealTimeTrainingUpdate(BaseModel):
    """Real-time training update for WebSocket connections."""
    session_id: str = Field(..., description="Training session ID")
    update_type: str = Field(..., description="Type of update: progress, metrics, completion")
    data: Dict[str, Any] = Field(..., description="Update data")
    timestamp: datetime = Field(..., description="Update timestamp")

class TrainingRecommendation(BaseModel):
    """AI-generated training recommendations."""
    session_id: str = Field(..., description="Training session ID")
    recommendation_type: str = Field(..., description="Type of recommendation")
    priority: str = Field(..., description="Priority: high, medium, low")
    description: str = Field(..., description="Recommendation description")
    expected_improvement: float = Field(..., description="Expected accuracy improvement")
    implementation_effort: str = Field(..., description="Implementation effort: easy, medium, hard")
