"""
Advanced AI Training System for Ultra-Low Quality Image Enhancement

This module implements cutting-edge training techniques specifically designed
for improving AI models on the lowest quality images, using modern deep learning
approaches and active learning strategies.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sqlite3
from datetime import datetime
import asyncio
from dataclasses import dataclass

# Import existing systems
from ai_training_system import CambodianIDTrainingSystem
from ai_enhancement_config import AI_TRAINING_CONFIG, ACTIVE_LEARNING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics for tracking training progress."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_score: float
    processing_time: float
    quality_improvement: float


class AdvancedAITrainingSystem(CambodianIDTrainingSystem):
    """
    Advanced AI training system with modern techniques for ultra-low quality images.
    
    Features:
    - Active learning for optimal sample selection
    - Transfer learning from state-of-the-art models
    - Quality-aware training strategies
    - Real-time performance monitoring
    - Automated model improvement
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """Initialize the advanced training system."""
        super().__init__(data_dir)
        
        # Advanced training directories
        (self.data_dir / "quality_analysis").mkdir(exist_ok=True)
        (self.data_dir / "active_learning").mkdir(exist_ok=True)
        (self.data_dir / "model_checkpoints").mkdir(exist_ok=True)
        (self.data_dir / "performance_logs").mkdir(exist_ok=True)
        
        # Initialize advanced database tables
        self._init_advanced_database()
        
        logger.info("Advanced AI training system initialized")
    
    def _init_advanced_database(self):
        """Initialize advanced database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Quality analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                blur_score REAL,
                noise_level REAL,
                contrast_score REAL,
                resolution_score REAL,
                overall_quality REAL,
                difficulty_level TEXT,
                enhancement_potential REAL,
                FOREIGN KEY (image_id) REFERENCES training_images (id)
            )
        """)
        
        # Active learning table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_learning_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                uncertainty_score REAL,
                learning_value REAL,
                priority_score REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                samples_processed INTEGER,
                accuracy_improvement REAL,
                model_version TEXT,
                configuration TEXT,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_image_quality(self, image_path: str) -> Dict[str, float]:
        """
        Perform comprehensive quality analysis of an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Noise level estimation
            noise_level = self._estimate_noise_level(gray)
            
            # Contrast analysis
            contrast_score = gray.std()
            
            # Resolution analysis
            height, width = gray.shape
            resolution_score = (height * width) / 1000000  # Megapixels
            
            # Overall quality score (weighted combination)
            overall_quality = (
                min(blur_score / 1000, 1.0) * 0.3 +
                max(0, 1.0 - noise_level / 50) * 0.25 +
                min(contrast_score / 100, 1.0) * 0.25 +
                min(resolution_score / 2, 1.0) * 0.2
            )
            
            # Determine difficulty level
            if overall_quality < 0.3:
                difficulty_level = "ultra_hard"
            elif overall_quality < 0.5:
                difficulty_level = "hard"
            elif overall_quality < 0.7:
                difficulty_level = "medium"
            else:
                difficulty_level = "easy"
            
            # Enhancement potential (how much improvement is possible)
            enhancement_potential = 1.0 - overall_quality
            
            return {
                "blur_score": blur_score,
                "noise_level": noise_level,
                "contrast_score": contrast_score,
                "resolution_score": resolution_score,
                "overall_quality": overall_quality,
                "difficulty_level": difficulty_level,
                "enhancement_potential": enhancement_potential
            }
            
        except Exception as e:
            logger.error(f"Quality analysis failed for {image_path}: {e}")
            return {}
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image using high-frequency analysis."""
        # Apply high-pass filter to detect noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(image, -1, kernel)
        noise_level = np.std(filtered)
        return noise_level
    
    def identify_active_learning_candidates(self, image_paths: List[str], top_k: int = 10) -> List[str]:
        """
        Identify the most valuable images for training using active learning.
        
        Args:
            image_paths: List of candidate image paths
            top_k: Number of top candidates to return
            
        Returns:
            List of most valuable image paths for training
        """
        candidates = []
        
        for image_path in image_paths:
            # Analyze image quality
            quality_metrics = self.analyze_image_quality(image_path)
            if not quality_metrics:
                continue
            
            # Calculate uncertainty score (higher for difficult cases)
            uncertainty_score = 1.0 - quality_metrics.get("overall_quality", 0.5)
            
            # Calculate learning value (potential for improvement)
            learning_value = quality_metrics.get("enhancement_potential", 0.5)
            
            # Calculate priority score (combination of uncertainty and learning value)
            priority_score = (uncertainty_score * 0.6 + learning_value * 0.4)
            
            candidates.append({
                "image_path": image_path,
                "uncertainty_score": uncertainty_score,
                "learning_value": learning_value,
                "priority_score": priority_score,
                "quality_metrics": quality_metrics
            })
        
        # Sort by priority score (highest first)
        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Store in active learning queue
        self._store_active_learning_candidates(candidates[:top_k])
        
        return [c["image_path"] for c in candidates[:top_k]]
    
    def _store_active_learning_candidates(self, candidates: List[Dict]):
        """Store active learning candidates in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for candidate in candidates:
            cursor.execute("""
                INSERT OR REPLACE INTO active_learning_queue 
                (image_path, uncertainty_score, learning_value, priority_score)
                VALUES (?, ?, ?, ?)
            """, (
                candidate["image_path"],
                candidate["uncertainty_score"],
                candidate["learning_value"],
                candidate["priority_score"]
            ))
        
        conn.commit()
        conn.close()
    
    def create_quality_aware_training_data(self, base_images: List[str], target_count: int = 200):
        """
        Create training data with quality-aware augmentation strategies.
        
        Args:
            base_images: List of base image paths
            target_count: Number of training samples to create
        """
        logger.info(f"Creating {target_count} quality-aware training samples...")
        
        # Analyze quality of base images
        quality_groups = {"ultra_hard": [], "hard": [], "medium": [], "easy": []}
        
        for image_path in base_images:
            quality_metrics = self.analyze_image_quality(image_path)
            difficulty = quality_metrics.get("difficulty_level", "medium")
            quality_groups[difficulty].append(image_path)
        
        # Create balanced dataset with emphasis on difficult cases
        samples_per_group = {
            "ultra_hard": int(target_count * 0.4),  # 40% ultra-hard cases
            "hard": int(target_count * 0.3),        # 30% hard cases
            "medium": int(target_count * 0.2),      # 20% medium cases
            "easy": int(target_count * 0.1)         # 10% easy cases
        }
        
        created_count = 0
        for difficulty, count in samples_per_group.items():
            if not quality_groups[difficulty]:
                continue
                
            for i in range(count):
                # Select random base image from this difficulty group
                base_image_path = np.random.choice(quality_groups[difficulty])
                
                # Apply difficulty-specific augmentations
                augmented_image = self._apply_quality_aware_augmentation(
                    base_image_path, difficulty
                )
                
                # Save augmented image
                output_path = self.data_dir / "processed_images" / f"quality_aware_{created_count:04d}.jpg"
                cv2.imwrite(str(output_path), augmented_image)
                created_count += 1
        
        logger.info(f"Created {created_count} quality-aware training samples")
    
    def _apply_quality_aware_augmentation(self, image_path: str, difficulty: str) -> np.ndarray:
        """Apply augmentation techniques based on image difficulty level."""
        image = cv2.imread(image_path)
        
        if difficulty == "ultra_hard":
            # Aggressive augmentations for ultra-hard cases
            augmentations = [
                lambda img: self._add_extreme_noise(img),
                lambda img: self._add_severe_blur(img),
                lambda img: self._reduce_resolution(img, 0.3),
                lambda img: self._add_compression_artifacts(img, 20),
                lambda img: self._add_lighting_issues(img)
            ]
        elif difficulty == "hard":
            # Moderate augmentations for hard cases
            augmentations = [
                lambda img: self._add_noise(img),
                lambda img: self._add_blur(img),
                lambda img: self._reduce_resolution(img, 0.5),
                lambda img: self._add_compression_artifacts(img, 40)
            ]
        else:
            # Light augmentations for easier cases
            augmentations = [
                lambda img: self._adjust_brightness(img),
                lambda img: self._rotate_slightly(img),
                lambda img: self._add_shadows(img)
            ]
        
        # Apply 1-3 random augmentations
        num_augs = np.random.randint(1, min(4, len(augmentations) + 1))
        selected_augs = np.random.choice(augmentations, size=num_augs, replace=False)
        
        augmented = image.copy()
        for aug_func in selected_augs:
            augmented = aug_func(augmented)
        
        return augmented
    
    def _add_extreme_noise(self, image: np.ndarray) -> np.ndarray:
        """Add extreme noise for ultra-hard training cases."""
        noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    def _add_severe_blur(self, image: np.ndarray) -> np.ndarray:
        """Add severe blur for ultra-hard training cases."""
        kernel_size = np.random.choice([9, 11, 13, 15])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _reduce_resolution(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Reduce image resolution and upscale back."""
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * factor), int(h * factor)))
        return cv2.resize(small, (w, h))
    
    def _add_compression_artifacts(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Add JPEG compression artifacts."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    def _add_lighting_issues(self, image: np.ndarray) -> np.ndarray:
        """Add lighting problems like uneven illumination."""
        h, w = image.shape[:2]
        
        # Create gradient lighting effect
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        gradient = 0.5 + 0.5 * np.sin(x * np.pi) * np.sin(y * np.pi)
        gradient = (gradient * 100).astype(np.uint8)
        
        # Apply lighting effect
        gradient_3ch = cv2.merge([gradient, gradient, gradient])
        return cv2.subtract(image, gradient_3ch)


def create_advanced_training_system(data_dir: str = "training_data") -> AdvancedAITrainingSystem:
    """Create an advanced training system instance."""
    return AdvancedAITrainingSystem(data_dir)
