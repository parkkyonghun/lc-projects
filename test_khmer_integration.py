#!/usr/bin/env python3
"""
Khmer Integration Test Script

This script tests the Khmer language integration to ensure everything
is working correctly with your Cambodian ID card OCR system.

Usage:
    python test_khmer_integration.py
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KhmerIntegrationTester:
    """Test suite for Khmer language integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Returns:
            Test results summary
        """
        logger.info("🧪 Starting Khmer Integration Tests")
        logger.info("=" * 50)
        
        # Test 1: Basic imports
        await self._test_imports()
        
        # Test 2: Khmer text processing
        await self._test_text_processing()
        
        # Test 3: Database initialization
        await self._test_database_setup()
        
        # Test 4: Synthetic data generation
        await self._test_synthetic_data()
        
        # Test 5: OCR integration
        await self._test_ocr_integration()
        
        # Print summary
        self._print_test_summary()
        
        return self.test_results
    
    async def _test_imports(self):
        """Test that all Khmer modules can be imported."""
        test_name = "Module Imports"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test imports
            from khmer_language_integration import create_khmer_integration
            from khmer_text_processor import create_khmer_processor
            from khmer_training_enhancer import create_khmer_enhancer
            
            logger.info("   ✅ All modules imported successfully")
            self._record_test_pass(test_name, "All required modules imported")
            
        except ImportError as e:
            logger.error(f"   ❌ Import failed: {e}")
            self._record_test_fail(test_name, f"Import error: {e}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")
            self._record_test_fail(test_name, f"Unexpected error: {e}")
    
    async def _test_text_processing(self):
        """Test Khmer text processing functionality."""
        test_name = "Khmer Text Processing"
        logger.info(f"Testing: {test_name}")
        
        try:
            from khmer_text_processor import create_khmer_processor
            
            processor = create_khmer_processor()
            
            # Test text normalization
            test_text = "ឈ្មោះ: សុខ សុភាព"
            normalized = processor.normalize_khmer_text(test_text)
            
            if normalized:
                logger.info("   ✅ Text normalization working")
            else:
                raise ValueError("Text normalization returned empty result")
            
            # Test validation
            validation = processor.validate_khmer_text(test_text)
            
            if validation and 'is_valid' in validation:
                logger.info("   ✅ Text validation working")
            else:
                raise ValueError("Text validation failed")
            
            # Test word segmentation
            words = processor.segment_khmer_words(test_text)
            
            if words and len(words) > 0:
                logger.info("   ✅ Word segmentation working")
            else:
                raise ValueError("Word segmentation failed")
            
            self._record_test_pass(test_name, "All text processing functions working")
            
        except Exception as e:
            logger.error(f"   ❌ Text processing failed: {e}")
            self._record_test_fail(test_name, f"Error: {e}")
    
    async def _test_database_setup(self):
        """Test database initialization."""
        test_name = "Database Setup"
        logger.info(f"Testing: {test_name}")
        
        try:
            from khmer_language_integration import create_khmer_integration
            from khmer_training_enhancer import create_khmer_enhancer
            
            # Test Khmer integration database
            integration = create_khmer_integration()
            if integration.db_path.exists():
                logger.info("   ✅ Khmer integration database created")
            else:
                raise FileNotFoundError("Khmer integration database not found")
            
            # Test training enhancer database
            enhancer = create_khmer_enhancer()
            if enhancer.db_path.exists():
                logger.info("   ✅ Training enhancer database created")
            else:
                raise FileNotFoundError("Training enhancer database not found")
            
            self._record_test_pass(test_name, "All databases initialized correctly")
            
        except Exception as e:
            logger.error(f"   ❌ Database setup failed: {e}")
            self._record_test_fail(test_name, f"Error: {e}")
    
    async def _test_synthetic_data(self):
        """Test synthetic data generation."""
        test_name = "Synthetic Data Generation"
        logger.info(f"Testing: {test_name}")
        
        try:
            from khmer_training_enhancer import create_khmer_enhancer
            
            enhancer = create_khmer_enhancer()
            
            # Generate a small test sample
            results = await enhancer.generate_synthetic_khmer_data(
                num_samples=2,
                quality_levels=["high", "medium"]
            )
            
            if results['generated_samples'] > 0:
                logger.info(f"   ✅ Generated {results['generated_samples']} synthetic samples")
                self._record_test_pass(test_name, f"Generated {results['generated_samples']} samples")
            else:
                raise ValueError("No synthetic samples generated")
            
        except Exception as e:
            logger.error(f"   ❌ Synthetic data generation failed: {e}")
            self._record_test_fail(test_name, f"Error: {e}")
    
    async def _test_ocr_integration(self):
        """Test OCR integration with Khmer processing."""
        test_name = "OCR Integration"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Check if OCR controller has been updated
            ocr_controller_path = Path("controllers/ocr_controller.py")
            
            if not ocr_controller_path.exists():
                raise FileNotFoundError("OCR controller not found")
            
            # Read the controller file and check for Khmer imports
            with open(ocr_controller_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "khmer_text_processor" in content and "khmer_language_integration" in content:
                logger.info("   ✅ OCR controller updated with Khmer integration")
                self._record_test_pass(test_name, "OCR controller properly integrated")
            else:
                raise ValueError("OCR controller not properly integrated")
            
        except Exception as e:
            logger.error(f"   ❌ OCR integration test failed: {e}")
            self._record_test_fail(test_name, f"Error: {e}")
    
    def _record_test_pass(self, test_name: str, details: str):
        """Record a passed test."""
        self.test_results["tests_passed"] += 1
        self.test_results["test_details"].append({
            "test": test_name,
            "status": "PASS",
            "details": details
        })
    
    def _record_test_fail(self, test_name: str, details: str):
        """Record a failed test."""
        self.test_results["tests_failed"] += 1
        self.test_results["test_details"].append({
            "test": test_name,
            "status": "FAIL",
            "details": details
        })
    
    def _print_test_summary(self):
        """Print test results summary."""
        total_tests = self.test_results["tests_passed"] + self.test_results["tests_failed"]
        
        print("\n" + "=" * 50)
        print("🧪 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {self.test_results['tests_passed']}")
        print(f"❌ Failed: {self.test_results['tests_failed']}")
        
        if total_tests > 0:
            success_rate = (self.test_results["tests_passed"] / total_tests) * 100
            print(f"📊 Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test in self.test_results["test_details"]:
            status_icon = "✅" if test["status"] == "PASS" else "❌"
            print(f"   {status_icon} {test['test']}: {test['details']}")
        
        if self.test_results["tests_failed"] == 0:
            print("\n🎉 All tests passed! Khmer integration is working correctly.")
            print("\n🚀 Next steps:")
            print("   • Run your training data collector")
            print("   • Test with real Cambodian ID card images")
            print("   • Monitor performance improvements")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
            print("   • Make sure all dependencies are installed")
            print("   • Run setup_khmer_integration.py if needed")
            print("   • Check file permissions and paths")


async def main():
    """Main test function."""
    tester = KhmerIntegrationTester()
    results = await tester.run_all_tests()
    
    # Return appropriate exit code
    if results["tests_failed"] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
