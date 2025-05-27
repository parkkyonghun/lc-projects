#!/usr/bin/env python3
"""
Simple Model Performance Comparison

Compare OCR performance with and without deployed model using API calls.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import os

class SimpleModelComparator:
    """Simple comparison using API calls."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_images_dir = Path("training_data/raw_images")
    
    def run_comparison(self, max_images: int = 3) -> Dict[str, Any]:
        """Run simple comparison between models."""
        print("🔍 Simple Model Performance Comparison")
        print("=" * 50)
        
        # Get test images
        test_images = self._get_test_images(max_images)
        if not test_images:
            print("❌ No test images found!")
            return {}
        
        print(f"📸 Testing with {len(test_images)} images")
        
        # Test with current model (whatever is active)
        print("\n🚀 Testing Current Active Model...")
        current_results = self._test_current_model(test_images)
        
        # Display results
        self._display_results(current_results)
        
        return current_results
    
    def _get_test_images(self, max_images: int) -> List[Path]:
        """Get list of test images."""
        if not self.test_images_dir.exists():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [
            f for f in self.test_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        return images[:max_images]
    
    def _test_current_model(self, test_images: List[Path]) -> Dict[str, Any]:
        """Test current active model."""
        results = {
            "model_info": self._get_model_info(),
            "total_images": len(test_images),
            "successful_extractions": 0,
            "field_extraction_counts": {
                "name": 0,
                "id_number": 0,
                "date_of_birth": 0,
                "gender": 0,
                "nationality": 0
            },
            "processing_times": [],
            "detailed_results": []
        }
        
        print(f"   Model: {results['model_info']['model_id']}")
        print(f"   Status: {results['model_info']['status']}")
        
        for i, image_path in enumerate(test_images, 1):
            print(f"   Processing image {i}/{len(test_images)}: {image_path.name}")
            
            try:
                # Process image through OCR API
                start_time = time.time()
                ocr_result = self._process_image_via_api(image_path)
                processing_time = time.time() - start_time
                
                # Analyze result
                analysis = self._analyze_result(ocr_result, image_path.name, processing_time)
                results["detailed_results"].append(analysis)
                results["processing_times"].append(processing_time)
                
                # Update counters
                if analysis["extraction_success"]:
                    results["successful_extractions"] += 1
                
                for field, extracted in analysis["extracted_fields"].items():
                    if extracted:
                        results["field_extraction_counts"][field] += 1
                        
            except Exception as e:
                print(f"     ❌ Error: {e}")
                results["detailed_results"].append({
                    "image_name": image_path.name,
                    "error": str(e),
                    "extraction_success": False,
                    "processing_time": 0.0
                })
        
        # Calculate rates
        if results["total_images"] > 0:
            results["success_rate"] = results["successful_extractions"] / results["total_images"]
            results["average_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"]) if results["processing_times"] else 0
            
            results["field_extraction_rates"] = {}
            for field, count in results["field_extraction_counts"].items():
                results["field_extraction_rates"][field] = count / results["total_images"]
        
        return results
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        try:
            response = requests.get(f"{self.base_url}/ui/api/models/comparison")
            if response.status_code == 200:
                data = response.json()
                active_model = data.get("active_model", {})
                return {
                    "model_id": active_model.get("model_id", "unknown"),
                    "status": active_model.get("status", "unknown"),
                    "has_active_model": active_model.get("has_active_model", False),
                    "accuracy": active_model.get("accuracy", "unknown")
                }
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")
        
        return {
            "model_id": "unknown",
            "status": "unknown",
            "has_active_model": False,
            "accuracy": "unknown"
        }
    
    def _process_image_via_api(self, image_path: Path) -> Dict[str, Any]:
        """Process image through OCR API."""
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/ocr/idcard", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    def _analyze_result(self, ocr_result: Dict, image_name: str, processing_time: float) -> Dict[str, Any]:
        """Analyze OCR result."""
        extracted_fields = {
            "name": bool(ocr_result.get("full_name", "").strip()),
            "id_number": bool(ocr_result.get("id_number", "").strip()),
            "date_of_birth": bool(ocr_result.get("date_of_birth", "").strip()),
            "gender": bool(ocr_result.get("gender", "").strip()),
            "nationality": bool(ocr_result.get("nationality", "").strip())
        }
        
        extraction_success = any(extracted_fields.values())
        field_count = sum(extracted_fields.values())
        
        return {
            "image_name": image_name,
            "extraction_success": extraction_success,
            "extracted_fields": extracted_fields,
            "field_count": field_count,
            "processing_time": processing_time,
            "ocr_result": ocr_result
        }
    
    def _display_results(self, results: Dict[str, Any]):
        """Display comparison results."""
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE RESULTS")
        print("=" * 60)
        
        # Model info
        model_info = results["model_info"]
        print(f"\n🤖 Active Model: {model_info['model_id']}")
        print(f"   Status: {model_info['status']}")
        print(f"   Has Deployed Model: {model_info['has_active_model']}")
        if model_info.get('accuracy') != 'unknown':
            print(f"   Model Accuracy: {model_info['accuracy']:.1%}")
        
        # Overall performance
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Total Images Tested: {results['total_images']}")
        print(f"   Successful Extractions: {results['successful_extractions']}")
        print(f"   Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"   Average Processing Time: {results.get('average_processing_time', 0):.2f}s")
        
        # Field extraction rates
        print(f"\n🎯 FIELD EXTRACTION RATES:")
        field_rates = results.get("field_extraction_rates", {})
        for field, rate in field_rates.items():
            status = "✅" if rate > 0.7 else "⚠️" if rate > 0.3 else "❌"
            print(f"   {status} {field.replace('_', ' ').title():<15}: {rate:.1%}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in results["detailed_results"]:
            if result.get("error"):
                print(f"   ❌ {result['image_name']}: {result['error']}")
            else:
                status = "✅" if result["extraction_success"] else "❌"
                print(f"   {status} {result['image_name']}: {result['field_count']}/5 fields "
                      f"({result['processing_time']:.2f}s)")
        
        # Performance assessment
        success_rate = results.get('success_rate', 0)
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        if success_rate >= 0.8:
            print("   🚀 Excellent performance! Model is working well.")
        elif success_rate >= 0.6:
            print("   ✅ Good performance. Some room for improvement.")
        elif success_rate >= 0.4:
            print("   ⚠️ Moderate performance. Consider more training.")
        else:
            print("   ❌ Poor performance. Needs significant improvement.")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if model_info['has_active_model']:
            print("   • Deployed model is active and being used")
            if success_rate < 0.95:  # Your target accuracy
                print("   • Consider collecting more training data to reach 95% accuracy goal")
                print("   • Focus on images with failed extractions for training")
        else:
            print("   • No deployed model detected - using default OCR")
            print("   • Consider training and deploying a custom model")
        
        print(f"\n✅ Analysis completed!")


def main():
    """Main function."""
    comparator = SimpleModelComparator()
    
    print("🤖 Simple OCR Model Performance Analysis")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server first.")
            return
    except Exception:
        print("❌ Cannot connect to server. Please start the server first.")
        return
    
    # Run comparison
    results = comparator.run_comparison(max_images=6)
    
    return results


if __name__ == "__main__":
    main()
