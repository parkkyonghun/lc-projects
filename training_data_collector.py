#!/usr/bin/env python3
"""
Interactive Training Data Collector for Cambodian ID Cards

This script provides an easy interface to collect and annotate training data
for improving the AI models.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import asyncio

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_training_system import create_training_system


class TrainingDataCollector:
    """Interactive training data collection interface."""
    
    def __init__(self):
        """Initialize the collector."""
        self.training_system = create_training_system()
        print("🎓 Cambodian ID Card Training Data Collector")
        print("=" * 60)
    
    def collect_from_current_image(self, image_path: str = "id_card.jpg"):
        """Collect training data from the current test image."""
        print(f"📁 Processing: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Show image info
        image = Image.open(image_path)
        print(f"📏 Image size: {image.size}")
        
        # Get ground truth from user
        print("\n📝 Please provide the correct field values:")
        print("(Press Enter to skip a field)")
        
        ground_truth = {}
        
        # Collect Khmer name
        name_kh = input("👤 Name (Khmer): ").strip()
        if name_kh:
            ground_truth["name_kh"] = name_kh
        
        # Collect English name
        name_en = input("👤 Name (English): ").strip()
        if name_en:
            ground_truth["name_en"] = name_en
        
        # Collect ID number
        id_number = input("🆔 ID Number: ").strip()
        if id_number:
            ground_truth["id_number"] = id_number
        
        # Collect date of birth
        dob = input("📅 Date of Birth (DD.MM.YYYY): ").strip()
        if dob:
            ground_truth["date_of_birth"] = dob
        
        # Collect gender
        print("⚧  Gender:")
        print("   1. Male")
        print("   2. Female")
        gender_choice = input("   Choice (1/2): ").strip()
        if gender_choice == "1":
            ground_truth["gender"] = "Male"
        elif gender_choice == "2":
            ground_truth["gender"] = "Female"
        
        # Collect nationality
        nationality = input("🏳️  Nationality [Cambodian]: ").strip()
        ground_truth["nationality"] = nationality if nationality else "Cambodian"
        
        # Save training data
        if ground_truth:
            training_id = self.training_system.collect_training_data(image_path, ground_truth)
            if training_id:
                print(f"\n✅ Training data saved with ID: {training_id}")
                print("📊 Ground truth collected:")
                for field, value in ground_truth.items():
                    print(f"   {field}: {value}")
            else:
                print("❌ Failed to save training data")
        else:
            print("⚠️  No ground truth provided, skipping...")
    
    def collect_batch_data(self, image_dir: str):
        """Collect training data from a directory of images."""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return
        
        print(f"📁 Found {len(image_files)} images in {image_dir}")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n📸 Processing image {i}/{len(image_files)}: {image_file.name}")
            
            # Ask if user wants to process this image
            choice = input("Process this image? (y/n/q): ").strip().lower()
            if choice == 'q':
                break
            elif choice != 'y':
                continue
            
            self.collect_from_current_image(str(image_file))
    
    def evaluate_current_model(self):
        """Evaluate the current model performance."""
        print("\n📊 Evaluating Current Model Performance")
        print("=" * 50)
        
        # Use available test images
        test_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            test_images.extend(Path('.').glob(f'*{ext}'))
        
        if not test_images:
            print("❌ No test images found in current directory")
            return
        
        print(f"🧪 Testing with {len(test_images)} images...")
        
        try:
            results = self.training_system.evaluate_current_model([str(img) for img in test_images])
            
            print("\n📈 Evaluation Results:")
            print(f"   📊 Extraction Rate: {results['extraction_rate']:.1%}")
            print(f"   📁 Total Images: {results['total_images']}")
            print(f"   ✅ Successful: {results['successful_extractions']}")
            
            print("\n🎯 Field Accuracy:")
            for field, accuracy in results['field_accuracy'].items():
                print(f"   {field}: {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    def show_training_report(self):
        """Show comprehensive training report."""
        print("\n📋 Training Data Report")
        print("=" * 50)
        
        try:
            report = self.training_system.generate_training_report()
            
            # Training data statistics
            data_stats = report['training_data']
            print(f"📊 Training Data:")
            print(f"   Total Images: {data_stats['total_images']}")
            print(f"   Average Quality: {data_stats['average_quality']:.2f}")
            print(f"   Average Accuracy: {data_stats['average_accuracy']:.2f}")
            
            # Field statistics
            print(f"\n🎯 Field Statistics:")
            for field, stats in report['field_statistics'].items():
                accuracy = stats['accuracy']
                print(f"   {field}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
            
            # Recommendations
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
                
        except Exception as e:
            print(f"❌ Failed to generate report: {e}")
    
    def create_synthetic_data(self):
        """Create synthetic training data."""
        print("\n🔬 Creating Synthetic Training Data")
        print("=" * 50)
        
        # Find base images
        base_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            base_images.extend(str(p) for p in Path('.').glob(f'*{ext}'))
        
        if not base_images:
            print("❌ No base images found for synthetic data generation")
            return
        
        count = input("How many synthetic images to create? [50]: ").strip()
        count = int(count) if count.isdigit() else 50
        
        print(f"🔬 Creating {count} synthetic images from {len(base_images)} base images...")
        
        try:
            self.training_system.create_synthetic_training_data(base_images, count)
            print(f"✅ Created {count} synthetic training images")
        except Exception as e:
            print(f"❌ Synthetic data creation failed: {e}")
    
    def interactive_menu(self):
        """Show interactive menu."""
        while True:
            print("\n🎓 Training Data Collector Menu")
            print("=" * 40)
            print("1. 📸 Collect data from current image (id_card.jpg)")
            print("2. 📁 Collect batch data from directory")
            print("3. 📊 Evaluate current model")
            print("4. 📋 Show training report")
            print("5. 🔬 Create synthetic data")
            print("6. 🚪 Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                self.collect_from_current_image()
            elif choice == "2":
                directory = input("Enter image directory path: ").strip()
                if directory:
                    self.collect_batch_data(directory)
            elif choice == "3":
                self.evaluate_current_model()
            elif choice == "4":
                self.show_training_report()
            elif choice == "5":
                self.create_synthetic_data()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")


def collect_sample_data():
    """Collect sample training data from the current id_card.jpg."""
    print("🎯 Quick Training Data Collection")
    print("=" * 50)
    
    # Based on our successful OCR results, let's collect this as training data
    ground_truth = {
        "name_kh": "ស្រី ពៅ",  # From OCR results
        "name_en": "SREY POV",  # From OCR results  
        "id_number": "34323458",  # From OCR results
        "date_of_birth": "03.08.1999",  # From OCR results (converted format)
        "gender": "Female",  # From OCR results
        "nationality": "Cambodian"
    }
    
    print("📝 Using extracted data as ground truth:")
    for field, value in ground_truth.items():
        print(f"   {field}: {value}")
    
    confirm = input("\nSave this as training data? (y/n): ").strip().lower()
    if confirm == 'y':
        system = create_training_system()
        training_id = system.collect_training_data("id_card.jpg", ground_truth)
        if training_id:
            print(f"✅ Training data saved with ID: {training_id}")
            print("🎓 This will help improve future OCR accuracy!")
        else:
            print("❌ Failed to save training data")
    else:
        print("⚠️  Training data not saved")


def main():
    """Main function."""
    print("🤖 AI Training System for Cambodian ID Cards")
    print("=" * 60)
    print("This system helps improve OCR accuracy by collecting training data")
    print("and fine-tuning AI models specifically for Cambodian ID cards.")
    print()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            collect_sample_data()
        elif command == "interactive":
            collector = TrainingDataCollector()
            collector.interactive_menu()
        elif command == "evaluate":
            collector = TrainingDataCollector()
            collector.evaluate_current_model()
        elif command == "report":
            collector = TrainingDataCollector()
            collector.show_training_report()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: quick, interactive, evaluate, report")
    else:
        # Default to interactive mode
        collector = TrainingDataCollector()
        collector.interactive_menu()


if __name__ == "__main__":
    main()
