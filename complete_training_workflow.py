#!/usr/bin/env python3
"""
Complete AI Training Workflow for Cambodian ID Card OCR

This script implements a comprehensive training pipeline that includes:
1. Data collection and annotation
2. Model training and fine-tuning
3. Performance evaluation
4. Continuous improvement
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_training_system import create_training_system
from khmer_model_trainer import create_khmer_trainer
from training_data_collector import collect_sample_data


class CompleteTrainingWorkflow:
    """Complete training workflow for Cambodian ID card OCR."""
    
    def __init__(self):
        """Initialize the training workflow."""
        self.training_system = create_training_system()
        self.khmer_trainer = create_khmer_trainer()
        
        print("🎓 Complete AI Training Workflow for Cambodian ID Cards")
        print("=" * 70)
    
    def step1_collect_initial_data(self):
        """Step 1: Collect initial training data."""
        print("\n📊 STEP 1: Initial Data Collection")
        print("=" * 50)
        
        # Collect data from current successful OCR result
        print("🎯 Collecting training data from successful OCR result...")
        
        ground_truth = {
            "name_kh": "ស្រី ពៅ",
            "name_en": "SREY POV",
            "id_number": "34323458",
            "date_of_birth": "03.08.1999",
            "gender": "Female",
            "nationality": "Cambodian"
        }
        
        training_id = self.training_system.collect_training_data("id_card.jpg", ground_truth)
        
        if training_id:
            print(f"✅ Initial training data collected (ID: {training_id})")
            print("📋 Ground truth data:")
            for field, value in ground_truth.items():
                print(f"   {field}: {value}")
        else:
            print("❌ Failed to collect initial training data")
        
        return training_id is not None
    
    def step2_create_synthetic_data(self):
        """Step 2: Create synthetic training data."""
        print("\n🔬 STEP 2: Synthetic Data Generation")
        print("=" * 50)
        
        try:
            # Create synthetic variations of the base image
            base_images = ["id_card.jpg"]
            synthetic_count = 20  # Start with a small set
            
            print(f"🔬 Creating {synthetic_count} synthetic training images...")
            self.training_system.create_synthetic_training_data(base_images, synthetic_count)
            
            print(f"✅ Created {synthetic_count} synthetic training images")
            print("   These include variations with:")
            print("   • Different noise levels")
            print("   • Brightness adjustments")
            print("   • Blur effects")
            print("   • Slight rotations")
            print("   • Shadow effects")
            print("   • JPEG compression artifacts")
            
            return True
            
        except Exception as e:
            print(f"❌ Synthetic data generation failed: {e}")
            return False
    
    def step3_train_models(self):
        """Step 3: Train specialized models."""
        print("\n🤖 STEP 3: Model Training")
        print("=" * 50)
        
        try:
            # Create datasets
            print("📊 Creating training datasets...")
            
            char_dataset = self.khmer_trainer.create_khmer_character_dataset("training_data")
            field_dataset = self.khmer_trainer.create_field_extraction_dataset("training_data")
            
            print(f"   Character dataset: {len(char_dataset.get('characters', []))} samples")
            print(f"   Field dataset: {len(field_dataset.get('images', []))} samples")
            
            # Train Khmer recognition model
            print("\n🔤 Training Khmer character recognition model...")
            khmer_model_path = self.khmer_trainer.train_khmer_recognition_model(char_dataset, epochs=30)
            print(f"✅ Khmer model trained: {khmer_model_path}")
            
            # Train field extraction model
            print("\n🎯 Training field extraction model...")
            field_model_path = self.khmer_trainer.train_field_extraction_model(field_dataset, epochs=25)
            print(f"✅ Field extraction model trained: {field_model_path}")
            
            return True, khmer_model_path, field_model_path
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False, None, None
    
    def step4_evaluate_performance(self):
        """Step 4: Evaluate model performance."""
        print("\n📈 STEP 4: Performance Evaluation")
        print("=" * 50)
        
        try:
            # Evaluate current system
            test_images = ["id_card.jpg"]  # In practice, you'd have a separate test set
            
            print("🧪 Evaluating current OCR system...")
            results = self.training_system.evaluate_current_model(test_images)
            
            print("\n📊 Current Performance:")
            print(f"   Extraction Rate: {results['extraction_rate']:.1%}")
            print(f"   Total Images: {results['total_images']}")
            print(f"   Successful Extractions: {results['successful_extractions']}")
            
            print("\n🎯 Field-wise Accuracy:")
            for field, accuracy in results['field_accuracy'].items():
                status = "✅" if accuracy > 0.8 else "⚠️" if accuracy > 0.5 else "❌"
                print(f"   {status} {field}: {accuracy:.1%}")
            
            return results
            
        except Exception as e:
            print(f"❌ Performance evaluation failed: {e}")
            return None
    
    def step5_generate_recommendations(self, performance_results):
        """Step 5: Generate improvement recommendations."""
        print("\n💡 STEP 5: Improvement Recommendations")
        print("=" * 50)
        
        try:
            # Generate training report
            report = self.training_system.generate_training_report()
            
            print("📋 Training Data Status:")
            data_stats = report['training_data']
            print(f"   Total Images: {data_stats['total_images']}")
            print(f"   Average Quality: {data_stats.get('average_quality', 0):.2f}")
            
            print("\n🎯 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
            
            # Additional specific recommendations based on performance
            if performance_results:
                extraction_rate = performance_results['extraction_rate']
                
                print(f"\n🔍 Performance-Based Recommendations:")
                if extraction_rate < 0.7:
                    print("   • Focus on image quality improvement")
                    print("   • Collect more diverse training examples")
                    print("   • Implement more aggressive preprocessing")
                elif extraction_rate < 0.9:
                    print("   • Fine-tune existing models")
                    print("   • Add edge case examples to training data")
                    print("   • Optimize OCR parameters")
                else:
                    print("   • System performing well!")
                    print("   • Focus on maintaining performance")
                    print("   • Consider deployment optimization")
            
            return True
            
        except Exception as e:
            print(f"❌ Recommendation generation failed: {e}")
            return False
    
    def step6_continuous_improvement_plan(self):
        """Step 6: Create continuous improvement plan."""
        print("\n🔄 STEP 6: Continuous Improvement Plan")
        print("=" * 50)
        
        improvement_plan = {
            "immediate_actions": [
                "Collect 50+ more real ID card images",
                "Annotate ground truth for all collected images",
                "Implement active learning for difficult cases",
                "Set up automated performance monitoring"
            ],
            "short_term_goals": [
                "Achieve 95%+ field extraction accuracy",
                "Reduce processing time to <2 seconds",
                "Handle 10+ different image quality levels",
                "Support multiple ID card layouts"
            ],
            "long_term_vision": [
                "Real-time OCR processing",
                "Multi-language support (Khmer + English + others)",
                "Edge deployment capabilities",
                "Automated quality assessment and routing"
            ],
            "data_collection_strategy": [
                "Partner with organizations for diverse data",
                "Implement user feedback collection",
                "Create data augmentation pipeline",
                "Establish data quality standards"
            ]
        }
        
        print("🎯 Immediate Actions (Next 2 weeks):")
        for action in improvement_plan["immediate_actions"]:
            print(f"   • {action}")
        
        print("\n📈 Short-term Goals (Next 3 months):")
        for goal in improvement_plan["short_term_goals"]:
            print(f"   • {goal}")
        
        print("\n🚀 Long-term Vision (Next year):")
        for vision in improvement_plan["long_term_vision"]:
            print(f"   • {vision}")
        
        # Save improvement plan
        plan_file = Path("ai_improvement_plan.json")
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(improvement_plan, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 Improvement plan saved: {plan_file}")
        
        return improvement_plan
    
    def run_complete_workflow(self):
        """Run the complete training workflow."""
        print("🚀 Starting Complete AI Training Workflow")
        print("=" * 70)
        
        workflow_results = {
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "success": False
        }
        
        try:
            # Step 1: Data Collection
            if self.step1_collect_initial_data():
                workflow_results["steps_completed"].append("data_collection")
            
            # Step 2: Synthetic Data
            if self.step2_create_synthetic_data():
                workflow_results["steps_completed"].append("synthetic_data")
            
            # Step 3: Model Training
            success, khmer_model, field_model = self.step3_train_models()
            if success:
                workflow_results["steps_completed"].append("model_training")
                workflow_results["khmer_model"] = khmer_model
                workflow_results["field_model"] = field_model
            
            # Step 4: Performance Evaluation
            performance_results = self.step4_evaluate_performance()
            if performance_results:
                workflow_results["steps_completed"].append("performance_evaluation")
                workflow_results["performance"] = performance_results
            
            # Step 5: Recommendations
            if self.step5_generate_recommendations(performance_results):
                workflow_results["steps_completed"].append("recommendations")
            
            # Step 6: Improvement Plan
            improvement_plan = self.step6_continuous_improvement_plan()
            if improvement_plan:
                workflow_results["steps_completed"].append("improvement_plan")
                workflow_results["improvement_plan"] = improvement_plan
            
            workflow_results["success"] = len(workflow_results["steps_completed"]) >= 4
            workflow_results["end_time"] = datetime.now().isoformat()
            
            # Save workflow results
            results_file = Path("training_workflow_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✨ WORKFLOW COMPLETE!")
            print(f"📊 Steps completed: {len(workflow_results['steps_completed'])}/6")
            print(f"📋 Results saved: {results_file}")
            
            if workflow_results["success"]:
                print("🎉 Training workflow successful!")
                print("\n🚀 Next Steps:")
                print("   1. Start collecting more real ID card images")
                print("   2. Implement the improvement recommendations")
                print("   3. Monitor performance in production")
                print("   4. Set up automated retraining pipeline")
            else:
                print("⚠️  Workflow partially completed. Review results for next steps.")
            
            return workflow_results
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            workflow_results["error"] = str(e)
            workflow_results["end_time"] = datetime.now().isoformat()
            return workflow_results


def main():
    """Main function to run the training workflow."""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick data collection only
        collect_sample_data()
    else:
        # Full workflow
        workflow = CompleteTrainingWorkflow()
        results = workflow.run_complete_workflow()
        
        if results["success"]:
            print("\n🎓 Your AI training system is now set up and ready for continuous improvement!")
        else:
            print("\n📚 Training system initialized. Follow the recommendations to improve performance.")


if __name__ == "__main__":
    main()
