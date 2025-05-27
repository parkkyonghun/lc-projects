"""
Training Manager for AI Model Improvement

This module manages the complete training workflow, from data collection
to model deployment, with special focus on Flutter app integration.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from fastapi import UploadFile

# Import existing training systems
from ai_training_system import CambodianIDTrainingSystem
from advanced_ai_training import AdvancedAITrainingSystem
from smart_training_orchestrator import SmartTrainingOrchestrator

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Comprehensive training manager for AI model improvement.
    
    Features:
    - Session-based training management
    - Real-time progress tracking
    - Flutter app integration
    - Automated model deployment
    - Performance monitoring
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """Initialize the training manager."""
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.models_dir = self.data_dir / "deployed_models"
        
        # Create directories
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training systems
        self.training_system = CambodianIDTrainingSystem(str(data_dir))
        self.advanced_system = AdvancedAITrainingSystem(str(data_dir))
        self.orchestrator = SmartTrainingOrchestrator(str(data_dir))
        
        # Initialize session database
        self._init_session_db()
        
        logger.info("Training Manager initialized")
    
    def _init_session_db(self):
        """Initialize the session database."""
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP,
                target_accuracy REAL,
                max_samples INTEGER,
                current_samples INTEGER DEFAULT 0,
                current_accuracy REAL DEFAULT 0.0,
                config TEXT,
                last_updated TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_samples (
                sample_id TEXT PRIMARY KEY,
                session_id TEXT,
                image_path TEXT,
                ground_truth TEXT,
                quality_score REAL,
                source TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                recorded_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def start_session(
        self, 
        name: str, 
        target_accuracy: float = 0.95, 
        max_samples: int = 100,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Start a new training session.
        
        Args:
            name: Session name
            target_accuracy: Target accuracy goal
            max_samples: Maximum training samples
            config: Additional configuration
            
        Returns:
            Session information
        """
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Get baseline accuracy
        baseline_accuracy = await self._get_baseline_accuracy()
        
        # Store session in database
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_sessions 
            (session_id, name, status, created_at, target_accuracy, max_samples, current_accuracy, config, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, name, "active", created_at, target_accuracy, 
            max_samples, baseline_accuracy, json.dumps(config or {}), created_at
        ))
        
        conn.commit()
        conn.close()
        
        # Create session directory
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Started training session: {session_id}")
        
        return {
            "session_id": session_id,
            "created_at": created_at,
            "baseline_accuracy": baseline_accuracy
        }
    
    async def add_training_sample(
        self,
        session_id: str,
        image_file: UploadFile,
        ground_truth: Dict[str, str],
        image_quality: Optional[float] = None,
        source: str = "flutter_camera"
    ) -> Dict[str, Any]:
        """
        Add a training sample to a session.
        
        Args:
            session_id: Training session ID
            image_file: Uploaded image file
            ground_truth: Correct field values
            image_quality: Image quality score
            source: Source of the sample
            
        Returns:
            Sample processing result
        """
        # Validate session exists
        session = await self._get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Save image file
        sample_id = str(uuid.uuid4())
        session_dir = self.sessions_dir / session_id
        image_path = session_dir / f"{sample_id}.jpg"
        
        # Save uploaded file
        with open(image_path, "wb") as f:
            content = await image_file.read()
            f.write(content)
        
        # Assess image quality if not provided
        if image_quality is None:
            image_quality = await self._assess_image_quality(str(image_path))
        
        # Store sample in database
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO session_samples 
            (sample_id, session_id, image_path, ground_truth, quality_score, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id, session_id, str(image_path), json.dumps(ground_truth),
            image_quality, source, datetime.now()
        ))
        
        # Update session sample count
        cursor.execute("""
            UPDATE training_sessions 
            SET current_samples = current_samples + 1, last_updated = ?
            WHERE session_id = ?
        """, (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
        
        # Add to training system
        training_id = self.training_system.collect_training_data(str(image_path), ground_truth)
        
        # Check if we should trigger retraining
        should_retrain = await self._should_trigger_retraining(session_id)
        
        # Get session progress
        progress = await self.get_session_progress(session_id)
        
        logger.info(f"Added training sample {sample_id} to session {session_id}")
        
        return {
            "sample_id": sample_id,
            "training_id": training_id,
            "quality_score": image_quality,
            "should_retrain": should_retrain,
            "session_progress": progress,
            "estimated_improvement": 0.02 if image_quality > 0.7 else 0.01
        }
    
    async def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get training session progress."""
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, status, target_accuracy, max_samples, current_samples, current_accuracy, last_updated
            FROM training_sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "name": row[0],
            "status": row[1],
            "target_accuracy": row[2],
            "target_samples": row[3],
            "current_samples": row[4],
            "current_accuracy": row[5],
            "last_updated": row[6],
            "progress_percentage": (row[4] / row[3]) * 100 if row[3] > 0 else 0,
            "accuracy_progress": (row[5] / row[2]) * 100 if row[2] > 0 else 0
        }
    
    async def trigger_retraining(self, session_id: str) -> Dict[str, Any]:
        """Trigger model retraining for a session."""
        logger.info(f"Triggering retraining for session {session_id}")
        
        try:
            # Use the smart orchestrator for intelligent retraining
            result = await self.orchestrator.orchestrate_smart_training()
            
            # Update session accuracy based on training results
            if result.get("success"):
                new_accuracy = result.get("final_accuracy", 0.0)
                await self._update_session_accuracy(session_id, new_accuracy)
            
            return {
                "success": result.get("success", False),
                "message": "Retraining completed",
                "new_accuracy": result.get("final_accuracy", 0.0),
                "training_time": result.get("training_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Retraining failed for session {session_id}: {e}")
            return {
                "success": False,
                "message": f"Retraining failed: {str(e)}"
            }
    
    async def _get_baseline_accuracy(self) -> float:
        """Get baseline model accuracy."""
        # This would typically evaluate the current model
        # For now, return a simulated baseline
        return 0.75
    
    async def _assess_image_quality(self, image_path: str) -> float:
        """Assess image quality score."""
        # Use the advanced training system's quality assessment
        try:
            quality_metrics = self.advanced_system.analyze_image_quality(image_path)
            return quality_metrics.get("overall_quality", 0.5) if quality_metrics else 0.5
        except:
            return 0.5
    
    async def _should_trigger_retraining(self, session_id: str) -> bool:
        """Determine if retraining should be triggered."""
        progress = await self.get_session_progress(session_id)
        
        # Trigger retraining every 10 samples or when target is reached
        return (
            progress["current_samples"] % 10 == 0 or 
            progress["current_samples"] >= progress["target_samples"]
        )
    
    async def _get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM training_sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(zip([col[0] for col in cursor.description], row)) if row else None
    
    async def _update_session_accuracy(self, session_id: str, accuracy: float):
        """Update session accuracy."""
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE training_sessions 
            SET current_accuracy = ?, last_updated = ?
            WHERE session_id = ?
        """, (accuracy, datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    async def list_sessions(self, status: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """List training sessions."""
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT * FROM training_sessions 
                WHERE status = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT * FROM training_sessions 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def delete_session(self, session_id: str):
        """Delete a training session."""
        # Remove from database
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM session_samples WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM session_metrics WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM training_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        # Remove session directory
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        
        logger.info(f"Deleted training session: {session_id}")
    
    async def validate_model_for_deployment(self, session_id: str) -> Dict[str, Any]:
        """Validate if model is ready for deployment."""
        progress = await self.get_session_progress(session_id)
        
        ready = (
            progress["current_accuracy"] >= progress["target_accuracy"] and
            progress["current_samples"] >= 10  # Minimum samples
        )
        
        reason = ""
        if not ready:
            if progress["current_accuracy"] < progress["target_accuracy"]:
                reason = f"Accuracy {progress['current_accuracy']:.2f} below target {progress['target_accuracy']:.2f}"
            elif progress["current_samples"] < 10:
                reason = f"Insufficient samples: {progress['current_samples']} (minimum 10)"
        
        return {
            "ready": ready,
            "reason": reason,
            "current_accuracy": progress["current_accuracy"],
            "target_accuracy": progress["target_accuracy"]
        }
    
    async def deploy_model(self, session_id: str, model_name: str):
        """Deploy a trained model."""
        logger.info(f"Deploying model {model_name} from session {session_id}")
        
        # This would implement actual model deployment
        # For now, simulate deployment
        await asyncio.sleep(2)  # Simulate deployment time
        
        # Mark session as completed
        db_path = self.data_dir / "sessions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE training_sessions 
            SET status = 'completed', last_updated = ?
            WHERE session_id = ?
        """, (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Model {model_name} deployed successfully")
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available trained models."""
        # This would list actual deployed models
        # For now, return simulated models
        return [
            {
                "model_id": "cambodian_id_v1",
                "name": "Cambodian ID OCR v1.0",
                "accuracy": 0.89,
                "created_at": "2024-01-15T10:30:00",
                "status": "active"
            },
            {
                "model_id": "cambodian_id_v2",
                "name": "Cambodian ID OCR v2.0",
                "accuracy": 0.93,
                "created_at": "2024-01-20T14:15:00",
                "status": "active"
            }
        ]
