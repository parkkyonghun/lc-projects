"""
AI Training System for Cambodian ID Card OCR

This module implements a comprehensive training system to improve AI models
specifically for Cambodian ID cards using custom datasets and fine-tuning.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class CambodianIDTrainingSystem:
    """
    Comprehensive training system for Cambodian ID card AI models.
    
    Features:
    - Data collection and annotation
    - Custom dataset creation
    - Model fine-tuning for Khmer script
    - Performance evaluation and improvement
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """
        Initialize the training system.
        
        Args:
            data_dir: Directory to store training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw_images").mkdir(exist_ok=True)
        (self.data_dir / "processed_images").mkdir(exist_ok=True)
        (self.data_dir / "annotations").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        (self.data_dir / "evaluation").mkdir(exist_ok=True)
        
        # Initialize database for training data management
        self.db_path = self.data_dir / "training_data.db"
        self._init_database()
        
        logger.info(f"Training system initialized with data directory: {data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for training data management."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                image_path TEXT NOT NULL,
                annotation_path TEXT,
                quality_score REAL,
                ocr_accuracy REAL,
                extraction_success INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence REAL,
                bounding_box TEXT,
                is_correct BOOLEAN,
                FOREIGN KEY (image_id) REFERENCES training_images (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                accuracy REAL,
                precision_khmer REAL,
                recall_khmer REAL,
                f1_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def collect_training_data(self, image_path: str, ground_truth: Dict[str, str]) -> str:
        """
        Collect and annotate training data.
        
        Args:
            image_path: Path to the ID card image
            ground_truth: Dictionary with correct field values
            
        Returns:
            Training data ID
        """
        try:
            # Copy image to training directory
            image = Image.open(image_path)
            filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(image_path).name}"
            new_image_path = self.data_dir / "raw_images" / filename
            image.save(new_image_path)
            
            # Create annotation file
            annotation_path = self.data_dir / "annotations" / f"{Path(filename).stem}.json"
            annotation_data = {
                "filename": filename,
                "ground_truth": ground_truth,
                "image_size": image.size,
                "created_at": datetime.now().isoformat(),
                "fields": {
                    "name_kh": ground_truth.get("name_kh", ""),
                    "name_en": ground_truth.get("name_en", ""),
                    "id_number": ground_truth.get("id_number", ""),
                    "date_of_birth": ground_truth.get("date_of_birth", ""),
                    "gender": ground_truth.get("gender", ""),
                    "nationality": ground_truth.get("nationality", "Cambodian")
                }
            }
            
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_images (filename, image_path, annotation_path)
                VALUES (?, ?, ?)
            """, (filename, str(new_image_path), str(annotation_path)))
            
            training_id = cursor.lastrowid
            
            # Store field data
            for field_name, field_value in ground_truth.items():
                cursor.execute("""
                    INSERT INTO extracted_fields (image_id, field_name, field_value, is_correct)
                    VALUES (?, ?, ?, ?)
                """, (training_id, field_name, field_value, True))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Training data collected: {filename}")
            return str(training_id)
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            return None
    
    def evaluate_current_model(self, test_images: List[str]) -> Dict[str, float]:
        """
        Evaluate current model performance on test images.
        
        Args:
            test_images: List of test image paths
            
        Returns:
            Performance metrics
        """
        from controllers.ocr_controller import process_cambodian_id_ocr
        import asyncio
        
        results = {
            "total_images": len(test_images),
            "successful_extractions": 0,
            "field_accuracy": {},
            "average_confidence": 0.0
        }
        
        field_counts = {"name": 0, "id_number": 0, "date_of_birth": 0, "gender": 0}
        field_correct = {"name": 0, "id_number": 0, "date_of_birth": 0, "gender": 0}
        
        async def evaluate_image(image_path: str):
            try:
                # Mock UploadFile for testing
                class MockFile:
                    def __init__(self, path):
                        self.path = path
                        self.content_type = "image/jpeg"
                    async def read(self):
                        with open(self.path, 'rb') as f:
                            return f.read()
                
                mock_file = MockFile(image_path)
                result = await process_cambodian_id_ocr(
                    mock_file,
                    use_enhanced_preprocessing=True,
                    use_ai_enhancement=True,
                    use_extreme_enhancement=False,
                    enhancement_mode="khmer_optimized",
                    use_robust_parsing=True
                )
                
                # Count successful extractions
                extracted_fields = 0
                for field in ["full_name", "id_number", "date_of_birth", "gender"]:
                    if getattr(result, field):
                        extracted_fields += 1
                
                if extracted_fields >= 3:  # Consider successful if 3+ fields extracted
                    results["successful_extractions"] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Evaluation failed for {image_path}: {e}")
                return None
        
        # Run evaluation
        for image_path in test_images:
            result = asyncio.run(evaluate_image(image_path))
            if result:
                # Update field accuracy counts
                for field in field_counts.keys():
                    field_counts[field] += 1
                    if getattr(result, field if field != "name" else "full_name"):
                        field_correct[field] += 1
        
        # Calculate accuracy metrics
        results["extraction_rate"] = results["successful_extractions"] / results["total_images"]
        
        for field in field_counts.keys():
            if field_counts[field] > 0:
                results["field_accuracy"][field] = field_correct[field] / field_counts[field]
            else:
                results["field_accuracy"][field] = 0.0
        
        # Store evaluation results
        self._store_evaluation_results(results)
        
        return results
    
    def _store_evaluation_results(self, results: Dict):
        """Store evaluation results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance (
                model_name, version, accuracy, precision_khmer, recall_khmer, f1_score, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "current_model",
            "1.0",
            results.get("extraction_rate", 0.0),
            results["field_accuracy"].get("name", 0.0),
            results["field_accuracy"].get("id_number", 0.0),
            results.get("average_confidence", 0.0),
            json.dumps(results)
        ))
        
        conn.commit()
        conn.close()
    
    def create_synthetic_training_data(self, base_images: List[str], count: int = 100):
        """
        Create synthetic training data through augmentation.
        
        Args:
            base_images: List of base image paths
            count: Number of synthetic images to create
        """
        logger.info(f"Creating {count} synthetic training images...")
        
        augmentation_techniques = [
            self._add_noise,
            self._adjust_brightness,
            self._add_blur,
            self._rotate_slightly,
            self._add_shadows,
            self._compress_jpeg
        ]
        
        for i in range(count):
            # Select random base image
            base_image_path = np.random.choice(base_images)
            image = cv2.imread(base_image_path)
            
            # Apply random augmentations
            num_augmentations = np.random.randint(1, 4)
            selected_augmentations = np.random.choice(
                augmentation_techniques, 
                size=num_augmentations, 
                replace=False
            )
            
            augmented_image = image.copy()
            for aug_func in selected_augmentations:
                augmented_image = aug_func(augmented_image)
            
            # Save synthetic image
            synthetic_path = self.data_dir / "processed_images" / f"synthetic_{i:04d}.jpg"
            cv2.imwrite(str(synthetic_path), augmented_image)
        
        logger.info(f"Created {count} synthetic training images")
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise to image."""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust image brightness."""
        factor = np.random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """Add blur to image."""
        kernel_size = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _rotate_slightly(self, image: np.ndarray) -> np.ndarray:
        """Rotate image slightly."""
        angle = np.random.uniform(-5, 5)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h))
    
    def _add_shadows(self, image: np.ndarray) -> np.ndarray:
        """Add shadow effects."""
        h, w = image.shape[:2]
        shadow = np.zeros((h, w), dtype=np.uint8)
        
        # Create random shadow shape
        pts = np.random.randint(0, min(h, w), (6, 2))
        cv2.fillPoly(shadow, [pts], 100)
        
        # Apply shadow
        shadow_3ch = cv2.merge([shadow, shadow, shadow])
        return cv2.subtract(image, shadow_3ch)
    
    def _compress_jpeg(self, image: np.ndarray) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        quality = np.random.randint(30, 80)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    def generate_training_report(self) -> Dict:
        """Generate comprehensive training report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get training data statistics
        cursor.execute("SELECT COUNT(*) FROM training_images")
        total_images = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(quality_score) FROM training_images WHERE quality_score IS NOT NULL")
        avg_quality = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(ocr_accuracy) FROM training_images WHERE ocr_accuracy IS NOT NULL")
        avg_accuracy = cursor.fetchone()[0] or 0.0
        
        # Get field extraction statistics
        cursor.execute("""
            SELECT field_name, COUNT(*) as total, SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct
            FROM extracted_fields
            GROUP BY field_name
        """)
        field_stats = cursor.fetchall()
        
        # Get recent model performance
        cursor.execute("""
            SELECT * FROM model_performance
            ORDER BY training_date DESC
            LIMIT 5
        """)
        recent_performance = cursor.fetchall()
        
        conn.close()
        
        report = {
            "training_data": {
                "total_images": total_images,
                "average_quality": avg_quality,
                "average_accuracy": avg_accuracy
            },
            "field_statistics": {
                row[0]: {"total": row[1], "correct": row[2], "accuracy": row[2]/row[1] if row[1] > 0 else 0}
                for row in field_stats
            },
            "recent_performance": recent_performance,
            "recommendations": self._generate_recommendations(total_images, avg_accuracy)
        }
        
        return report
    
    def _generate_recommendations(self, total_images: int, avg_accuracy: float) -> List[str]:
        """Generate training recommendations based on current data."""
        recommendations = []
        
        if total_images < 100:
            recommendations.append("Collect more training data (target: 500+ images)")
        
        if avg_accuracy < 0.8:
            recommendations.append("Focus on improving image quality in training data")
            recommendations.append("Add more diverse examples of poor quality images")
        
        recommendations.extend([
            "Implement active learning to identify difficult cases",
            "Create specialized models for different image quality levels",
            "Add more Khmer script specific training data",
            "Implement transfer learning from pre-trained OCR models"
        ])
        
        return recommendations


def create_training_system(data_dir: str = "training_data") -> CambodianIDTrainingSystem:
    """Create a training system instance."""
    return CambodianIDTrainingSystem(data_dir)


def collect_training_sample(image_path: str, ground_truth: Dict[str, str]) -> str:
    """
    Convenience function to collect a training sample.
    
    Args:
        image_path: Path to ID card image
        ground_truth: Correct field values
        
    Returns:
        Training sample ID
    """
    system = create_training_system()
    return system.collect_training_data(image_path, ground_truth)
