#!/usr/bin/env python3
"""
Real-time Model Improvement System

This module implements a continuous learning system that automatically
improves AI models in real-time based on user feedback and new data.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_ai_training import AdvancedAITrainingSystem
from smart_training_orchestrator import SmartTrainingOrchestrator

logger = logging.getLogger(__name__)


class RealTimeModelImprovement:
    """
    Real-time model improvement system with continuous learning capabilities.
    
    Features:
    - Continuous monitoring of model performance
    - Automatic data collection from user interactions
    - Real-time feedback processing
    - Automated model retraining triggers
    - Performance drift detection
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """Initialize the real-time improvement system."""
        self.data_dir = Path(data_dir)
        self.training_system = AdvancedAITrainingSystem(data_dir)
        self.orchestrator = SmartTrainingOrchestrator(data_dir)
        
        # Real-time monitoring directories
        (self.data_dir / "realtime_feedback").mkdir(exist_ok=True)
        (self.data_dir / "performance_monitoring").mkdir(exist_ok=True)
        (self.data_dir / "auto_retraining").mkdir(exist_ok=True)
        
        # Performance monitoring
        self.performance_buffer = []
        self.last_retrain_time = datetime.now()
        self.monitoring_active = False
        
        # Thresholds for automatic retraining
        self.accuracy_threshold = 0.85  # Retrain if accuracy drops below this
        self.feedback_threshold = 10    # Retrain after this many feedback samples
        self.time_threshold = timedelta(hours=24)  # Retrain at least daily
        
        logger.info("Real-time model improvement system initialized")
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            print("⚠️  Monitoring already active")
            return
        
        self.monitoring_active = True
        print("🔄 Starting Continuous Model Monitoring")
        print("=" * 50)
        print("   📊 Performance tracking: ACTIVE")
        print("   🔄 Auto-retraining: ENABLED")
        print("   📝 Feedback collection: ACTIVE")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        print("   ✅ Continuous monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        print("⏹️  Continuous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check if retraining is needed
                if self._should_retrain():
                    print("\n🚨 Automatic Retraining Triggered!")
                    asyncio.run(self._trigger_automatic_retraining())
                
                # Sleep for monitoring interval
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _should_retrain(self) -> bool:
        """Determine if automatic retraining should be triggered."""
        # Check time since last retrain
        time_since_retrain = datetime.now() - self.last_retrain_time
        if time_since_retrain > self.time_threshold:
            return True
        
        # Check performance degradation
        if len(self.performance_buffer) >= 5:
            recent_accuracy = sum(self.performance_buffer[-5:]) / 5
            if recent_accuracy < self.accuracy_threshold:
                return True
        
        # Check feedback accumulation
        feedback_files = list((self.data_dir / "realtime_feedback").glob("*.json"))
        if len(feedback_files) >= self.feedback_threshold:
            return True
        
        return False
    
    async def _trigger_automatic_retraining(self):
        """Trigger automatic model retraining."""
        print("   🤖 Starting automatic retraining...")
        
        try:
            # Run intelligent training cycle
            results = await self.orchestrator.run_intelligent_training_cycle(
                target_accuracy=0.92,
                max_iterations=3
            )
            
            # Update last retrain time
            self.last_retrain_time = datetime.now()
            
            # Clear feedback buffer
            self._clear_feedback_buffer()
            
            print(f"   ✅ Automatic retraining complete: {results['final_accuracy']:.1%}")
            
        except Exception as e:
            logger.error(f"Automatic retraining failed: {e}")
    
    def collect_user_feedback(self, image_path: str, ocr_result: Dict[str, Any], 
                            user_corrections: Dict[str, str], confidence: float):
        """
        Collect user feedback for continuous learning.
        
        Args:
            image_path: Path to the processed image
            ocr_result: Original OCR results
            user_corrections: User-provided corrections
            confidence: User confidence in corrections (0.0-1.0)
        """
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "original_ocr": ocr_result,
            "user_corrections": user_corrections,
            "user_confidence": confidence,
            "feedback_type": "correction"
        }
        
        # Save feedback
        feedback_file = (self.data_dir / "realtime_feedback" / 
                        f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        print(f"📝 User feedback collected: {feedback_file.name}")
        
        # Add to training data if confidence is high
        if confidence >= 0.8:
            self._add_feedback_to_training_data(image_path, user_corrections)
    
    def _add_feedback_to_training_data(self, image_path: str, corrections: Dict[str, str]):
        """Add high-confidence feedback to training data."""
        try:
            # Create ground truth from corrections
            ground_truth = {
                "name_kh": corrections.get("name_kh", ""),
                "name_en": corrections.get("name_en", ""),
                "id_number": corrections.get("id_number", ""),
                "date_of_birth": corrections.get("date_of_birth", ""),
                "gender": corrections.get("gender", ""),
                "nationality": corrections.get("nationality", "Cambodian")
            }
            
            # Filter out empty values
            ground_truth = {k: v for k, v in ground_truth.items() if v}
            
            if ground_truth:
                training_id = self.training_system.collect_training_data(image_path, ground_truth)
                if training_id:
                    print(f"   ✅ Added to training data: ID {training_id}")
                
        except Exception as e:
            logger.error(f"Failed to add feedback to training data: {e}")
    
    def _clear_feedback_buffer(self):
        """Clear processed feedback files."""
        feedback_files = list((self.data_dir / "realtime_feedback").glob("*.json"))
        processed_dir = self.data_dir / "realtime_feedback" / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        for feedback_file in feedback_files:
            # Move to processed directory
            new_path = processed_dir / feedback_file.name
            feedback_file.rename(new_path)
        
        print(f"   📁 Moved {len(feedback_files)} feedback files to processed")
    
    def record_performance_metric(self, accuracy: float, processing_time: float, 
                                quality_score: float):
        """Record a performance metric for monitoring."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "processing_time": processing_time,
            "quality_score": quality_score
        }
        
        # Add to buffer
        self.performance_buffer.append(accuracy)
        
        # Keep buffer size manageable
        if len(self.performance_buffer) > 100:
            self.performance_buffer = self.performance_buffer[-50:]
        
        # Save metric
        metric_file = (self.data_dir / "performance_monitoring" / 
                      f"metric_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(metric_file, 'w') as f:
            json.dump(metric, f, indent=2)
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load recent metrics
        metrics = []
        metric_files = list((self.data_dir / "performance_monitoring").glob("*.json"))
        
        for metric_file in metric_files:
            try:
                with open(metric_file, 'r') as f:
                    metric = json.load(f)
                
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff_date:
                    metrics.append(metric)
                    
            except Exception as e:
                logger.error(f"Failed to load metric {metric_file}: {e}")
        
        if not metrics:
            return {"error": "No metrics available for the specified period"}
        
        # Calculate trends
        accuracies = [m["accuracy"] for m in metrics]
        processing_times = [m["processing_time"] for m in metrics]
        quality_scores = [m["quality_score"] for m in metrics]
        
        trends = {
            "period_days": days,
            "total_samples": len(metrics),
            "accuracy": {
                "average": sum(accuracies) / len(accuracies),
                "min": min(accuracies),
                "max": max(accuracies),
                "trend": "improving" if accuracies[-1] > accuracies[0] else "declining"
            },
            "processing_time": {
                "average": sum(processing_times) / len(processing_times),
                "min": min(processing_times),
                "max": max(processing_times)
            },
            "quality_score": {
                "average": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores)
            }
        }
        
        return trends
    
    def generate_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive improvement report."""
        print("\n📊 Real-time Improvement Report")
        print("=" * 50)
        
        # Get performance trends
        trends_7d = self.get_performance_trends(7)
        trends_30d = self.get_performance_trends(30)
        
        # Count feedback samples
        feedback_files = list((self.data_dir / "realtime_feedback").glob("*.json"))
        processed_feedback = list((self.data_dir / "realtime_feedback" / "processed").glob("*.json"))
        
        # Get training data stats
        base_report = self.training_system.generate_training_report()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "performance_trends": {
                "7_days": trends_7d,
                "30_days": trends_30d
            },
            "feedback_stats": {
                "pending_feedback": len(feedback_files),
                "processed_feedback": len(processed_feedback),
                "total_feedback": len(feedback_files) + len(processed_feedback)
            },
            "training_stats": base_report["training_data"],
            "last_retrain": self.last_retrain_time.isoformat(),
            "next_retrain_trigger": {
                "time_based": (self.last_retrain_time + self.time_threshold).isoformat(),
                "feedback_based": f"{len(feedback_files)}/{self.feedback_threshold} samples",
                "performance_based": f"accuracy < {self.accuracy_threshold:.1%}"
            },
            "recommendations": self._generate_realtime_recommendations(trends_7d, len(feedback_files))
        }
        
        return report
    
    def _generate_realtime_recommendations(self, trends: Dict[str, Any], 
                                         pending_feedback: int) -> List[str]:
        """Generate recommendations based on real-time data."""
        recommendations = []
        
        if isinstance(trends, dict) and "accuracy" in trends:
            if trends["accuracy"]["trend"] == "declining":
                recommendations.append("🚨 Performance declining - consider immediate retraining")
            
            if trends["accuracy"]["average"] < 0.85:
                recommendations.append("📈 Average accuracy below threshold - increase training data")
        
        if pending_feedback >= 5:
            recommendations.append("📝 High feedback volume - valuable for next training cycle")
        
        if not recommendations:
            recommendations.extend([
                "✅ System performing well - continue monitoring",
                "🔄 Regular retraining schedule is optimal",
                "📊 Consider expanding to new image types"
            ])
        
        return recommendations


def demo_realtime_system():
    """Demonstrate the real-time improvement system."""
    print("🤖 Real-time Model Improvement System Demo")
    print("=" * 60)
    
    # Initialize system
    rt_system = RealTimeModelImprovement()
    
    # Start monitoring
    rt_system.start_continuous_monitoring()
    
    # Simulate some performance metrics
    print("\n📊 Simulating Performance Metrics...")
    import random
    for i in range(5):
        accuracy = random.uniform(0.80, 0.95)
        processing_time = random.uniform(1.5, 3.0)
        quality_score = random.uniform(0.6, 0.9)
        
        rt_system.record_performance_metric(accuracy, processing_time, quality_score)
        print(f"   Metric {i+1}: {accuracy:.1%} accuracy, {processing_time:.1f}s")
    
    # Simulate user feedback
    print("\n📝 Simulating User Feedback...")
    sample_corrections = {
        "name_en": "CORRECTED NAME",
        "id_number": "12345678",
        "date_of_birth": "01.01.2000"
    }
    
    rt_system.collect_user_feedback(
        "sample_image.jpg",
        {"name_en": "WRONG NAME", "id_number": "87654321"},
        sample_corrections,
        confidence=0.9
    )
    
    # Generate report
    print("\n📋 Generating Improvement Report...")
    report = rt_system.generate_improvement_report()
    
    print(f"\n📊 Report Summary:")
    print(f"   Monitoring: {report['monitoring_status']}")
    print(f"   Pending Feedback: {report['feedback_stats']['pending_feedback']}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    # Stop monitoring
    rt_system.stop_continuous_monitoring()
    
    return report


if __name__ == "__main__":
    demo_realtime_system()
