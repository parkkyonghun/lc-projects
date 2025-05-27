#!/usr/bin/env python3
"""
Model Performance Comparison Tool

Compare performance between default OCR and deployed trained models.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import OCR processing
from controllers.ocr_controller import process_cambodian_id_ocr, get_active_model_info
from fastapi import UploadFile
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceComparator:
    """Compare performance between default and deployed models."""

    def __init__(self, test_data_dir: str = "training_data/raw_images"):
        self.test_data_dir = Path(test_data_dir)
        self.results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "test_images": [],
            "default_model_results": {},
            "deployed_model_results": {},
            "performance_comparison": {}
        }

    async def run_comprehensive_comparison(self, max_images: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive comparison between default and deployed models.

        Args:
            max_images: Maximum number of test images to use

        Returns:
            Comprehensive comparison results
        """
        print("🔍 Model Performance Comparison")
        print("=" * 50)

        # Get test images
        test_images = self._get_test_images(max_images)
        if not test_images:
            print("❌ No test images found!")
            return self.results

        print(f"📸 Testing with {len(test_images)} images")

        # Test with default model (disable deployed model temporarily)
        print("\n🔧 Testing Default OCR Model...")
        await self._backup_active_model()
        default_results = await self._test_model_performance(test_images, "default")

        # Test with deployed model
        print("\n🚀 Testing Deployed Model...")
        await self._restore_active_model()
        deployed_results = await self._test_model_performance(test_images, "deployed")

        # Compare results
        comparison = self._compare_results(default_results, deployed_results)

        # Store results
        self.results.update({
            "test_images": [str(img) for img in test_images],
            "default_model_results": default_results,
            "deployed_model_results": deployed_results,
            "performance_comparison": comparison
        })

        # Display results
        self._display_comparison_results()

        # Save results
        self._save_results()

        return self.results

    def _get_test_images(self, max_images: int) -> List[Path]:
        """Get list of test images."""
        if not self.test_data_dir.exists():
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [
            f for f in self.test_data_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        return images[:max_images]

    async def _backup_active_model(self):
        """Backup active model configuration."""
        active_model_path = Path("model_registry/active_model.json")
        backup_path = Path("model_registry/active_model_backup.json")

        if active_model_path.exists():
            import shutil
            shutil.copy2(active_model_path, backup_path)
            # Remove active model to test default
            active_model_path.unlink()

    async def _restore_active_model(self):
        """Restore active model configuration."""
        active_model_path = Path("model_registry/active_model.json")
        backup_path = Path("model_registry/active_model_backup.json")

        if backup_path.exists():
            import shutil
            shutil.copy2(backup_path, active_model_path)
            backup_path.unlink()

    async def _test_model_performance(self, test_images: List[Path], model_type: str) -> Dict[str, Any]:
        """Test model performance on given images."""
        results = {
            "model_type": model_type,
            "total_images": len(test_images),
            "successful_extractions": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "field_extraction_rates": {
                "name": 0,
                "id_number": 0,
                "date_of_birth": 0,
                "gender": 0,
                "nationality": 0
            },
            "detailed_results": []
        }

        # Check active model
        active_model = await get_active_model_info()
        model_info = {
            "has_active_model": active_model is not None,
            "model_id": active_model.get("model_id", "default") if active_model else "default",
            "accuracy": active_model.get("accuracy", "unknown") if active_model else "baseline"
        }
        results["model_info"] = model_info

        print(f"   Model: {model_info['model_id']} (Accuracy: {model_info['accuracy']})")

        for i, image_path in enumerate(test_images, 1):
            print(f"   Processing image {i}/{len(test_images)}: {image_path.name}")

            try:
                # Process image
                start_time = time.time()
                result = await self._process_single_image(image_path)
                processing_time = time.time() - start_time

                # Analyze result
                analysis = self._analyze_ocr_result(result)
                analysis["processing_time"] = processing_time
                analysis["image_name"] = image_path.name

                results["detailed_results"].append(analysis)
                results["total_processing_time"] += processing_time

                # Update field extraction rates
                if analysis["extraction_success"]:
                    results["successful_extractions"] += 1

                for field in results["field_extraction_rates"]:
                    if analysis["extracted_fields"].get(field):
                        results["field_extraction_rates"][field] += 1

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results["detailed_results"].append({
                    "image_name": image_path.name,
                    "error": str(e),
                    "extraction_success": False,
                    "processing_time": 0.0
                })

        # Calculate averages
        if results["total_images"] > 0:
            results["average_processing_time"] = results["total_processing_time"] / results["total_images"]
            results["extraction_success_rate"] = results["successful_extractions"] / results["total_images"]

            for field in results["field_extraction_rates"]:
                results["field_extraction_rates"][field] = results["field_extraction_rates"][field] / results["total_images"]

        return results

    async def _process_single_image(self, image_path: Path) -> Any:
        """Process a single image through OCR."""
        # Create UploadFile-like object
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Create a mock UploadFile
        class MockUploadFile:
            def __init__(self, content: bytes, filename: str):
                self.content = content
                self.filename = filename
                self.content_type = "image/jpeg"

            async def read(self):
                return self.content

        mock_file = MockUploadFile(image_bytes, image_path.name)

        # Process through OCR
        result = await process_cambodian_id_ocr(
            file=mock_file,
            use_enhanced_preprocessing=True,
            use_ai_enhancement=False,
            use_extreme_enhancement=False,
            enhancement_mode="auto",
            use_robust_parsing=True
        )

        return result

    def _analyze_ocr_result(self, result: Any) -> Dict[str, Any]:
        """Analyze OCR result for performance metrics."""
        extracted_fields = {
            "name": bool(result.full_name and result.full_name.strip()),
            "id_number": bool(result.id_number and result.id_number.strip()),
            "date_of_birth": bool(result.date_of_birth and result.date_of_birth.strip()),
            "gender": bool(result.gender and result.gender.strip()),
            "nationality": bool(result.nationality and result.nationality.strip())
        }

        extraction_success = any(extracted_fields.values())
        field_count = sum(extracted_fields.values())

        return {
            "extraction_success": extraction_success,
            "extracted_fields": extracted_fields,
            "field_count": field_count,
            "field_extraction_rate": field_count / len(extracted_fields),
            "raw_result": {
                "full_name": result.full_name,
                "id_number": result.id_number,
                "date_of_birth": result.date_of_birth,
                "gender": result.gender,
                "nationality": result.nationality
            }
        }

    def _compare_results(self, default_results: Dict, deployed_results: Dict) -> Dict[str, Any]:
        """Compare results between default and deployed models."""
        comparison = {
            "overall_improvement": {},
            "field_improvements": {},
            "performance_metrics": {},
            "summary": {}
        }

        # Overall improvements
        default_success_rate = default_results.get("extraction_success_rate", 0)
        deployed_success_rate = deployed_results.get("extraction_success_rate", 0)
        success_improvement = deployed_success_rate - default_success_rate

        comparison["overall_improvement"] = {
            "default_success_rate": default_success_rate,
            "deployed_success_rate": deployed_success_rate,
            "improvement": success_improvement,
            "improvement_percentage": (success_improvement / default_success_rate * 100) if default_success_rate > 0 else 0
        }

        # Field-wise improvements
        for field in default_results["field_extraction_rates"]:
            default_rate = default_results["field_extraction_rates"][field]
            deployed_rate = deployed_results["field_extraction_rates"][field]
            improvement = deployed_rate - default_rate

            comparison["field_improvements"][field] = {
                "default_rate": default_rate,
                "deployed_rate": deployed_rate,
                "improvement": improvement,
                "improvement_percentage": (improvement / default_rate * 100) if default_rate > 0 else 0
            }

        # Performance metrics
        default_time = default_results.get("average_processing_time", 0)
        deployed_time = deployed_results.get("average_processing_time", 0)
        time_difference = deployed_time - default_time

        comparison["performance_metrics"] = {
            "default_processing_time": default_time,
            "deployed_processing_time": deployed_time,
            "time_difference": time_difference,
            "speed_improvement": -time_difference  # Negative means faster
        }

        # Summary
        comparison["summary"] = {
            "overall_better": deployed_success_rate > default_success_rate,
            "faster": deployed_time < default_time,
            "significant_improvement": success_improvement > 0.05,  # 5% improvement threshold
            "recommendation": self._generate_recommendation(comparison)
        }

        return comparison

    def _generate_recommendation(self, comparison: Dict) -> str:
        """Generate recommendation based on comparison results."""
        overall_improvement = comparison["overall_improvement"]["improvement"]
        speed_improvement = comparison["performance_metrics"]["speed_improvement"]

        if overall_improvement > 0.1:  # 10% improvement
            return "🚀 Excellent! Deployed model shows significant improvement. Recommended for production."
        elif overall_improvement > 0.05:  # 5% improvement
            return "✅ Good improvement. Deployed model is better than default. Consider deployment."
        elif overall_improvement > 0:  # Any improvement
            return "📈 Slight improvement. Deployed model is marginally better."
        elif overall_improvement == 0:
            return "🔄 No significant difference. Both models perform similarly."
        else:
            return "⚠️ Default model performs better. Consider more training data or different approach."

    def _display_comparison_results(self):
        """Display comprehensive comparison results."""
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE COMPARISON RESULTS")
        print("=" * 60)

        # Model information
        default_info = self.results["default_model_results"]["model_info"]
        deployed_info = self.results["deployed_model_results"]["model_info"]

        print(f"\n🔧 Default Model: {default_info['model_id']}")
        print(f"🚀 Deployed Model: {deployed_info['model_id']} (Accuracy: {deployed_info['accuracy']:.1%})")

        # Overall performance
        comparison = self.results["performance_comparison"]
        overall = comparison["overall_improvement"]

        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Default Success Rate:  {overall['default_success_rate']:.1%}")
        print(f"   Deployed Success Rate: {overall['deployed_success_rate']:.1%}")
        print(f"   Improvement:           {overall['improvement']:+.1%} ({overall['improvement_percentage']:+.1f}%)")

        # Field-wise performance
        print(f"\n🎯 FIELD EXTRACTION RATES:")
        for field, data in comparison["field_improvements"].items():
            status = "✅" if data["improvement"] > 0 else "🔄" if data["improvement"] == 0 else "❌"
            print(f"   {status} {field.replace('_', ' ').title():<15}: "
                  f"{data['default_rate']:.1%} → {data['deployed_rate']:.1%} "
                  f"({data['improvement']:+.1%})")

        # Processing time
        metrics = comparison["performance_metrics"]
        speed_status = "⚡" if metrics["speed_improvement"] > 0 else "🐌"
        print(f"\n{speed_status} PROCESSING SPEED:")
        print(f"   Default Time:  {metrics['default_processing_time']:.2f}s")
        print(f"   Deployed Time: {metrics['deployed_processing_time']:.2f}s")
        print(f"   Difference:    {metrics['time_difference']:+.2f}s")

        # Recommendation
        summary = comparison["summary"]
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {summary['recommendation']}")

        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print(f"   Test Images: {len(self.results['test_images'])}")
        print(f"   Default Successful: {self.results['default_model_results']['successful_extractions']}")
        print(f"   Deployed Successful: {self.results['deployed_model_results']['successful_extractions']}")

    def _save_results(self):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main function to run the comparison."""
    comparator = ModelPerformanceComparator()

    print("🤖 AI OCR Model Performance Comparison Tool")
    print("=" * 50)

    # Run comparison
    results = await comparator.run_comprehensive_comparison(max_images=6)

    print("\n✅ Comparison completed!")
    return results


if __name__ == "__main__":
    asyncio.run(main())
