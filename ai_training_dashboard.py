#!/usr/bin/env python3
"""
AI Training Dashboard - Master Control Center

This module provides a comprehensive dashboard for managing all aspects
of AI model training, from data collection to deployment.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_ai_training import AdvancedAITrainingSystem
from smart_training_orchestrator import SmartTrainingOrchestrator
from realtime_model_improvement import RealTimeModelImprovement
from training_data_collector import TrainingDataCollector


class AITrainingDashboard:
    """
    Master dashboard for AI training system management.
    
    Provides unified interface for:
    - Training data management
    - Model training orchestration
    - Real-time performance monitoring
    - System configuration and optimization
    """
    
    def __init__(self, data_dir: str = "training_data"):
        """Initialize the AI training dashboard."""
        self.data_dir = Path(data_dir)
        
        # Initialize all subsystems
        self.training_system = AdvancedAITrainingSystem(data_dir)
        self.orchestrator = SmartTrainingOrchestrator(data_dir)
        self.realtime_system = RealTimeModelImprovement(data_dir)
        self.data_collector = TrainingDataCollector()
        
        print("🎛️  AI Training Dashboard Initialized")
        print("=" * 60)
        print("   🎓 Advanced Training System: READY")
        print("   🤖 Smart Orchestrator: READY")
        print("   🔄 Real-time Improvement: READY")
        print("   📊 Data Collector: READY")
    
    def show_main_menu(self):
        """Display the main dashboard menu."""
        while True:
            print("\n🎛️  AI Training Dashboard - Master Control")
            print("=" * 60)
            print("📊 MONITORING & ANALYSIS")
            print("   1. 📈 System Status Overview")
            print("   2. 📋 Comprehensive Training Report")
            print("   3. 📊 Performance Trends Analysis")
            print("   4. 🎯 Quality Distribution Analysis")
            print()
            print("🎓 TRAINING OPERATIONS")
            print("   5. 🚀 Run Smart Training Cycle")
            print("   6. 📸 Collect Training Data")
            print("   7. 🔬 Generate Synthetic Data")
            print("   8. 🎯 Active Learning Session")
            print()
            print("🔄 REAL-TIME SYSTEMS")
            print("   9. ▶️  Start Real-time Monitoring")
            print("  10. ⏹️  Stop Real-time Monitoring")
            print("  11. 📝 Process User Feedback")
            print("  12. 🔄 Trigger Manual Retraining")
            print()
            print("⚙️  SYSTEM MANAGEMENT")
            print("  13. 🛠️  System Configuration")
            print("  14. 💾 Export Training Data")
            print("  15. 📤 Backup System State")
            print("  16. 🚪 Exit Dashboard")
            
            choice = input("\nSelect option (1-16): ").strip()
            
            try:
                if choice == "1":
                    self.show_system_status()
                elif choice == "2":
                    self.show_comprehensive_report()
                elif choice == "3":
                    self.show_performance_trends()
                elif choice == "4":
                    self.show_quality_analysis()
                elif choice == "5":
                    asyncio.run(self.run_smart_training())
                elif choice == "6":
                    self.collect_training_data()
                elif choice == "7":
                    self.generate_synthetic_data()
                elif choice == "8":
                    self.run_active_learning()
                elif choice == "9":
                    self.start_realtime_monitoring()
                elif choice == "10":
                    self.stop_realtime_monitoring()
                elif choice == "11":
                    self.process_user_feedback()
                elif choice == "12":
                    asyncio.run(self.trigger_manual_retraining())
                elif choice == "13":
                    self.system_configuration()
                elif choice == "14":
                    self.export_training_data()
                elif choice == "15":
                    self.backup_system_state()
                elif choice == "16":
                    print("👋 Goodbye! AI Training Dashboard shutting down...")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-16.")
                    
            except Exception as e:
                print(f"❌ Error executing option {choice}: {e}")
                input("Press Enter to continue...")
    
    def show_system_status(self):
        """Display comprehensive system status."""
        print("\n📈 AI Training System Status Overview")
        print("=" * 60)
        
        # Training data statistics
        report = self.training_system.generate_training_report()
        data_stats = report['training_data']
        
        print("📊 TRAINING DATA STATUS")
        print(f"   Total Images: {data_stats['total_images']}")
        print(f"   Average Quality: {data_stats['average_quality']:.2f}")
        print(f"   Average Accuracy: {data_stats['average_accuracy']:.2f}")
        
        # Field statistics
        print("\n🎯 FIELD EXTRACTION STATUS")
        for field, stats in report['field_statistics'].items():
            accuracy = stats['accuracy']
            status = "🟢" if accuracy > 0.8 else "🟡" if accuracy > 0.6 else "🔴"
            print(f"   {status} {field}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
        
        # Real-time system status
        rt_report = self.realtime_system.generate_improvement_report()
        print(f"\n🔄 REAL-TIME MONITORING")
        print(f"   Status: {rt_report['monitoring_status'].upper()}")
        print(f"   Pending Feedback: {rt_report['feedback_stats']['pending_feedback']}")
        print(f"   Last Retrain: {rt_report['last_retrain'][:19]}")
        
        # System health indicators
        print("\n💚 SYSTEM HEALTH")
        health_score = self._calculate_health_score(data_stats, rt_report)
        health_status = "🟢 EXCELLENT" if health_score > 0.8 else "🟡 GOOD" if health_score > 0.6 else "🔴 NEEDS ATTENTION"
        print(f"   Overall Health: {health_status} ({health_score:.1%})")
        
        input("\nPress Enter to continue...")
    
    def _calculate_health_score(self, data_stats: Dict, rt_report: Dict) -> float:
        """Calculate overall system health score."""
        # Data quality component (40%)
        data_score = min(data_stats['average_quality'], 1.0) * 0.4
        
        # Training data volume component (30%)
        volume_score = min(data_stats['total_images'] / 100, 1.0) * 0.3
        
        # Real-time monitoring component (30%)
        monitoring_score = 0.3 if rt_report['monitoring_status'] == 'active' else 0.15
        
        return data_score + volume_score + monitoring_score
    
    def show_comprehensive_report(self):
        """Show comprehensive training report."""
        print("\n📋 Comprehensive AI Training Report")
        print("=" * 60)
        
        # Get reports from all subsystems
        training_report = self.training_system.generate_training_report()
        orchestrator_report = self.orchestrator.generate_training_report()
        realtime_report = self.realtime_system.generate_improvement_report()
        
        # Display key metrics
        print("📊 KEY PERFORMANCE INDICATORS")
        data_stats = training_report['training_data']
        print(f"   Training Samples: {data_stats['total_images']}")
        print(f"   Data Quality: {data_stats['average_quality']:.2f}/1.0")
        print(f"   Model Accuracy: {data_stats['average_accuracy']:.1%}")
        
        # Recommendations
        print("\n💡 INTELLIGENT RECOMMENDATIONS")
        all_recommendations = (
            training_report.get('recommendations', []) +
            orchestrator_report.get('orchestrator_metrics', {}).get('improvement_recommendations', []) +
            realtime_report.get('recommendations', [])
        )
        
        for i, rec in enumerate(all_recommendations[:10], 1):
            print(f"   {i}. {rec}")
        
        input("\nPress Enter to continue...")
    
    def show_performance_trends(self):
        """Show performance trends analysis."""
        print("\n📊 Performance Trends Analysis")
        print("=" * 60)
        
        # Get trends from real-time system
        trends_7d = self.realtime_system.get_performance_trends(7)
        trends_30d = self.realtime_system.get_performance_trends(30)
        
        if "error" not in trends_7d:
            print("📈 7-DAY TRENDS")
            acc = trends_7d['accuracy']
            print(f"   Average Accuracy: {acc['average']:.1%}")
            print(f"   Trend: {acc['trend'].upper()}")
            print(f"   Range: {acc['min']:.1%} - {acc['max']:.1%}")
        
        if "error" not in trends_30d:
            print("\n📈 30-DAY TRENDS")
            acc = trends_30d['accuracy']
            print(f"   Average Accuracy: {acc['average']:.1%}")
            print(f"   Trend: {acc['trend'].upper()}")
            print(f"   Range: {acc['min']:.1%} - {acc['max']:.1%}")
        
        if "error" in trends_7d and "error" in trends_30d:
            print("⚠️  No performance data available yet.")
            print("   Start real-time monitoring to collect trends.")
        
        input("\nPress Enter to continue...")
    
    def show_quality_analysis(self):
        """Show image quality distribution analysis."""
        print("\n🎯 Image Quality Distribution Analysis")
        print("=" * 60)
        
        # Analyze available images
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            all_images.extend(Path('.').glob(f'*{ext}'))
            all_images.extend((self.data_dir / "raw_images").glob(f'*{ext}'))
        
        if not all_images:
            print("⚠️  No images found for analysis")
            input("Press Enter to continue...")
            return
        
        # Analyze quality distribution
        quality_distribution = {"ultra_hard": 0, "hard": 0, "medium": 0, "easy": 0}
        total_analyzed = 0
        
        print(f"🔍 Analyzing {min(len(all_images), 20)} images...")
        
        for image_path in all_images[:20]:  # Limit for performance
            quality_metrics = self.training_system.analyze_image_quality(str(image_path))
            if quality_metrics:
                difficulty = quality_metrics.get("difficulty_level", "medium")
                quality_distribution[difficulty] += 1
                total_analyzed += 1
        
        if total_analyzed > 0:
            print("\n📊 QUALITY DISTRIBUTION")
            for difficulty, count in quality_distribution.items():
                percentage = (count / total_analyzed) * 100
                bar = "█" * int(percentage / 5)  # Visual bar
                print(f"   {difficulty.upper():12} {count:3d} ({percentage:5.1f}%) {bar}")
            
            # Recommendations based on distribution
            print("\n💡 QUALITY-BASED RECOMMENDATIONS")
            if quality_distribution["ultra_hard"] > total_analyzed * 0.3:
                print("   🔴 High proportion of ultra-hard images - focus on extreme enhancement")
            if quality_distribution["easy"] > total_analyzed * 0.5:
                print("   🟢 Many high-quality images - consider more challenging test cases")
            if quality_distribution["medium"] + quality_distribution["hard"] > total_analyzed * 0.6:
                print("   🟡 Balanced distribution - good for comprehensive training")
        
        input("\nPress Enter to continue...")
    
    async def run_smart_training(self):
        """Run smart training cycle."""
        print("\n🚀 Smart Training Cycle")
        print("=" * 60)
        
        target_accuracy = input("Target accuracy (0.85-0.99) [0.92]: ").strip()
        target_accuracy = float(target_accuracy) if target_accuracy else 0.92
        
        max_iterations = input("Max iterations (1-10) [5]: ").strip()
        max_iterations = int(max_iterations) if max_iterations else 5
        
        print(f"\n🎯 Starting training cycle...")
        print(f"   Target: {target_accuracy:.1%}")
        print(f"   Max iterations: {max_iterations}")
        
        results = await self.orchestrator.run_intelligent_training_cycle(
            target_accuracy=target_accuracy,
            max_iterations=max_iterations
        )
        
        print(f"\n✅ Training cycle complete!")
        print(f"   Final accuracy: {results['final_accuracy']:.1%}")
        print(f"   Improvement: +{results['accuracy_improvement']:.1%}")
        
        input("Press Enter to continue...")
    
    def collect_training_data(self):
        """Collect training data using the data collector."""
        print("\n📸 Training Data Collection")
        print("=" * 60)
        
        print("1. Collect from current image (id_card.jpg)")
        print("2. Collect from directory")
        print("3. Quick collection with known data")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            self.data_collector.collect_from_current_image()
        elif choice == "2":
            directory = input("Enter directory path: ").strip()
            if directory:
                self.data_collector.collect_batch_data(directory)
        elif choice == "3":
            # Use the quick collection from training_data_collector
            from training_data_collector import collect_sample_data
            collect_sample_data()
        
        input("Press Enter to continue...")
    
    def generate_synthetic_data(self):
        """Generate synthetic training data."""
        print("\n🔬 Synthetic Data Generation")
        print("=" * 60)
        
        count = input("Number of synthetic images [100]: ").strip()
        count = int(count) if count.isdigit() else 100
        
        # Find base images
        base_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            base_images.extend(str(p) for p in Path('.').glob(f'*{ext}'))
        
        if not base_images:
            print("❌ No base images found")
            input("Press Enter to continue...")
            return
        
        print(f"🔬 Creating {count} synthetic images from {len(base_images)} base images...")
        
        self.training_system.create_quality_aware_training_data(base_images, count)
        
        print(f"✅ Created {count} synthetic training images")
        input("Press Enter to continue...")
    
    def run_active_learning(self):
        """Run active learning session."""
        print("\n🎯 Active Learning Session")
        print("=" * 60)
        
        # Find candidate images
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            all_images.extend(str(p) for p in Path('.').glob(f'*{ext}'))
        
        if not all_images:
            print("❌ No images found for active learning")
            input("Press Enter to continue...")
            return
        
        top_k = input(f"Number of top candidates to identify [10]: ").strip()
        top_k = int(top_k) if top_k.isdigit() else 10
        
        print(f"🔍 Analyzing {len(all_images)} images for learning value...")
        
        candidates = self.training_system.identify_active_learning_candidates(all_images, top_k)
        
        print(f"\n🎓 Top {len(candidates)} learning candidates identified:")
        for i, candidate in enumerate(candidates, 1):
            print(f"   {i}. {Path(candidate).name}")
        
        input("Press Enter to continue...")
    
    def start_realtime_monitoring(self):
        """Start real-time monitoring."""
        print("\n▶️  Starting Real-time Monitoring")
        print("=" * 60)
        
        self.realtime_system.start_continuous_monitoring()
        input("Press Enter to continue...")
    
    def stop_realtime_monitoring(self):
        """Stop real-time monitoring."""
        print("\n⏹️  Stopping Real-time Monitoring")
        print("=" * 60)
        
        self.realtime_system.stop_continuous_monitoring()
        input("Press Enter to continue...")
    
    def process_user_feedback(self):
        """Process user feedback."""
        print("\n📝 User Feedback Processing")
        print("=" * 60)
        
        # Simulate feedback processing
        print("This would integrate with your OCR API to collect real user feedback.")
        print("For now, simulating feedback collection...")
        
        sample_corrections = {
            "name_en": "CORRECTED NAME",
            "id_number": "12345678"
        }
        
        self.realtime_system.collect_user_feedback(
            "sample_image.jpg",
            {"name_en": "WRONG NAME"},
            sample_corrections,
            confidence=0.9
        )
        
        print("✅ Sample feedback processed")
        input("Press Enter to continue...")
    
    async def trigger_manual_retraining(self):
        """Trigger manual retraining."""
        print("\n🔄 Manual Retraining")
        print("=" * 60)
        
        confirm = input("Start manual retraining? (y/n): ").strip().lower()
        if confirm == 'y':
            await self.realtime_system._trigger_automatic_retraining()
        
        input("Press Enter to continue...")
    
    def system_configuration(self):
        """System configuration menu."""
        print("\n🛠️  System Configuration")
        print("=" * 60)
        print("Configuration options would be implemented here.")
        print("This could include:")
        print("   • Training thresholds")
        print("   • Quality parameters")
        print("   • Monitoring intervals")
        print("   • Model selection")
        
        input("Press Enter to continue...")
    
    def export_training_data(self):
        """Export training data."""
        print("\n💾 Export Training Data")
        print("=" * 60)
        
        export_path = f"training_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get comprehensive data
        report = self.training_system.generate_training_report()
        
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Training data exported to: {export_path}")
        input("Press Enter to continue...")
    
    def backup_system_state(self):
        """Backup system state."""
        print("\n📤 System State Backup")
        print("=" * 60)
        
        backup_dir = Path(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(exist_ok=True)
        
        print(f"📁 Creating backup in: {backup_dir}")
        print("   This would backup all training data, models, and configurations")
        
        # In a real implementation, this would copy all relevant files
        print("✅ Backup completed (simulated)")
        input("Press Enter to continue...")


def main():
    """Main function to run the AI training dashboard."""
    print("🎛️  AI Training Dashboard - Master Control Center")
    print("=" * 60)
    print("Welcome to the comprehensive AI training management system!")
    print()
    
    dashboard = AITrainingDashboard()
    dashboard.show_main_menu()


if __name__ == "__main__":
    main()
