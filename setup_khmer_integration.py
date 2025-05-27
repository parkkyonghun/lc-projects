#!/usr/bin/env python3
"""
Khmer Integration Setup Script

This script sets up the complete Khmer language integration for your
Cambodian ID card OCR system, including all resources, tools, and enhancements.

Usage:
    python setup_khmer_integration.py
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KhmerIntegrationSetup:
    """Complete setup manager for Khmer language integration."""

    def __init__(self):
        """Initialize the setup manager."""
        self.project_root = Path.cwd()
        self.setup_results = {
            "steps_completed": [],
            "steps_failed": [],
            "warnings": [],
            "next_steps": []
        }

    async def run_complete_setup(self) -> Dict[str, Any]:
        """
        Run the complete Khmer integration setup.

        Returns:
            Setup results and status
        """
        logger.info("🇰🇭 Starting Khmer Language Integration Setup")
        logger.info("=" * 60)

        try:
            # Step 1: Install Python dependencies
            await self._install_dependencies()

            # Step 2: Set up Khmer language resources
            await self._setup_khmer_resources()

            # Step 3: Initialize databases
            await self._initialize_databases()

            # Step 4: Generate initial synthetic data
            await self._generate_initial_data()

            # Step 5: Test the integration
            await self._test_integration()

            # Step 6: Update existing training system
            await self._update_training_system()

            logger.info("✅ Khmer integration setup completed successfully!")
            self._print_summary()

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            self.setup_results["steps_failed"].append(f"Critical error: {e}")

        return self.setup_results

    async def _install_dependencies(self):
        """Install required Python dependencies."""
        logger.info("📦 Installing Python dependencies...")

        try:
            import subprocess

            # Required packages for Khmer processing
            packages = [
                "Pillow>=9.0.0",
                "opencv-python>=4.5.0",
                "numpy>=1.21.0",
                "pytesseract>=0.3.8",
            ]

            # Optional packages for enhanced functionality
            optional_packages = [
                "transformers",  # For pre-trained models
                "torch",  # For deep learning
                "datasets",  # For HuggingFace datasets
            ]

            # Install required packages
            for package in packages:
                try:
                    # Use DEVNULL for compatibility with older Python versions
                    with open(os.devnull, 'w') as devnull:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package
                        ], stdout=devnull, stderr=devnull)
                    logger.info(f"   ✅ Installed: {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"   ⚠️  Failed to install: {package}")
                    self.setup_results["warnings"].append(f"Failed to install {package}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Error installing {package}: {e}")
                    self.setup_results["warnings"].append(f"Error installing {package}: {e}")

            # Try to install optional packages
            for package in optional_packages:
                try:
                    with open(os.devnull, 'w') as devnull:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package
                        ], stdout=devnull, stderr=devnull)
                    logger.info(f"   ✅ Installed (optional): {package}")
                except subprocess.CalledProcessError:
                    logger.info(f"   ⏭️  Skipped (optional): {package}")
                except Exception as e:
                    logger.info(f"   ⏭️  Skipped (optional) {package}: {e}")

            self.setup_results["steps_completed"].append("Python dependencies installed")

        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            self.setup_results["steps_failed"].append("Python dependencies installation")
            raise

    async def _setup_khmer_resources(self):
        """Set up Khmer language resources."""
        logger.info("🇰🇭 Setting up Khmer language resources...")

        try:
            from khmer_language_integration import create_khmer_integration

            # Create Khmer integration system
            integration = create_khmer_integration()

            # Install resources
            results = await integration.install_khmer_resources()

            logger.info(f"   ✅ Installed {len(results['installed'])} resources")
            if results['failed']:
                logger.warning(f"   ⚠️  Failed to install {len(results['failed'])} resources")
                self.setup_results["warnings"].extend(results['failed'])

            self.setup_results["steps_completed"].append("Khmer resources setup")

        except Exception as e:
            logger.error(f"Failed to setup Khmer resources: {e}")
            self.setup_results["steps_failed"].append("Khmer resources setup")
            raise

    async def _initialize_databases(self):
        """Initialize all required databases."""
        logger.info("🗄️  Initializing databases...")

        try:
            from khmer_text_processor import create_khmer_processor
            from khmer_training_enhancer import create_khmer_enhancer

            # Initialize Khmer processor (creates database)
            processor = create_khmer_processor()
            logger.info("   ✅ Khmer text processor database initialized")

            # Initialize Khmer training enhancer (creates database)
            enhancer = create_khmer_enhancer()
            logger.info("   ✅ Khmer training enhancer database initialized")

            self.setup_results["steps_completed"].append("Database initialization")

        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            self.setup_results["steps_failed"].append("Database initialization")
            raise

    async def _generate_initial_data(self):
        """Generate initial synthetic training data."""
        logger.info("🎨 Generating initial synthetic training data...")

        try:
            from khmer_training_enhancer import create_khmer_enhancer

            enhancer = create_khmer_enhancer()

            # Generate a small set of synthetic data for testing
            results = await enhancer.generate_synthetic_khmer_data(
                num_samples=20,
                quality_levels=["high", "medium", "low"]
            )

            logger.info(f"   ✅ Generated {results['generated_samples']} synthetic samples")
            if results['failed_samples'] > 0:
                logger.warning(f"   ⚠️  Failed to generate {results['failed_samples']} samples")

            self.setup_results["steps_completed"].append("Initial synthetic data generation")

        except Exception as e:
            logger.error(f"Failed to generate initial data: {e}")
            self.setup_results["steps_failed"].append("Initial synthetic data generation")
            # Don't raise - this is not critical for basic functionality

    async def _test_integration(self):
        """Test the Khmer integration."""
        logger.info("🧪 Testing Khmer integration...")

        try:
            from khmer_text_processor import process_khmer_text_quick

            # Test with sample Khmer text
            test_text = "ឈ្មោះ: សុខ សុភាព"
            results = process_khmer_text_quick(test_text)

            # Verify results
            if results['normalized'] and results['validation']:
                logger.info("   ✅ Khmer text processing test passed")
                self.setup_results["steps_completed"].append("Integration testing")
            else:
                logger.warning("   ⚠️  Khmer text processing test had issues")
                self.setup_results["warnings"].append("Integration test had issues")

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.setup_results["steps_failed"].append("Integration testing")
            # Don't raise - system might still work

    async def _update_training_system(self):
        """Update the existing training system with Khmer enhancements."""
        logger.info("🔄 Updating training system with Khmer enhancements...")

        try:
            # Update AI improvement plan
            self._update_improvement_plan()

            # Create integration documentation
            self._create_documentation()

            logger.info("   ✅ Training system updated")
            self.setup_results["steps_completed"].append("Training system update")

        except Exception as e:
            logger.error(f"Failed to update training system: {e}")
            self.setup_results["steps_failed"].append("Training system update")
            # Don't raise - this is not critical

    def _update_improvement_plan(self):
        """Update the AI improvement plan with Khmer integration status."""
        try:
            import json

            plan_path = self.project_root / "ai_improvement_plan.json"
            if plan_path.exists():
                with open(plan_path, 'r') as f:
                    plan = json.load(f)

                # Add integration status
                plan["khmer_integration_status"] = {
                    "setup_completed": True,
                    "setup_date": str(Path().cwd()),
                    "components_installed": [
                        "khmer_language_integration",
                        "khmer_text_processor",
                        "khmer_training_enhancer"
                    ],
                    "next_actions": [
                        "Run training data collection with Khmer enhancements",
                        "Test OCR with Khmer text processing",
                        "Evaluate performance improvements"
                    ]
                }

                with open(plan_path, 'w') as f:
                    json.dump(plan, f, indent=2, ensure_ascii=False)

                logger.info("   ✅ AI improvement plan updated")

        except Exception as e:
            logger.warning(f"Failed to update improvement plan: {e}")

    def _create_documentation(self):
        """Create documentation for the Khmer integration."""
        try:
            doc_content = """# Khmer Language Integration

## Overview
Your AI training system has been enhanced with comprehensive Khmer language support for improved Cambodian ID card OCR accuracy.

## New Components

### 1. Khmer Language Integration (`khmer_language_integration.py`)
- Manages Khmer language resources
- Handles installation and setup of Khmer tools
- Provides resource management and tracking

### 2. Khmer Text Processor (`khmer_text_processor.py`)
- Unicode normalization for Khmer text
- Character validation and correction
- OCR error detection and correction
- Field extraction for Cambodian ID cards

### 3. Khmer Training Enhancer (`khmer_training_enhancer.py`)
- Synthetic Khmer data generation
- Khmer-specific model training
- Performance evaluation with Khmer metrics
- Integration with existing training pipeline

## Usage

### Quick Start
```python
# Process Khmer text
from khmer_text_processor import process_khmer_text_quick
results = process_khmer_text_quick("ឈ្មោះ: សុខ សុភាព")

# Generate synthetic training data
from khmer_training_enhancer import enhance_training_with_khmer_resources
results = await enhance_training_with_khmer_resources()
```

### Integration with OCR
The OCR controller has been automatically enhanced with Khmer processing:
- Automatic text normalization
- Error correction
- Improved field extraction

## Next Steps
1. Collect more real Cambodian ID card images
2. Run training with enhanced Khmer capabilities
3. Evaluate performance improvements
4. Fine-tune models for specific use cases

## Resources
- Khmer OCR benchmark datasets
- Synthetic data generation tools
- Pre-trained Khmer language models
- Comprehensive evaluation metrics
"""

            doc_path = self.project_root / "KHMER_INTEGRATION.md"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)

            logger.info("   ✅ Documentation created")

        except Exception as e:
            logger.warning(f"Failed to create documentation: {e}")

    def _print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 60)
        print("🎉 KHMER INTEGRATION SETUP SUMMARY")
        print("=" * 60)

        print(f"\n✅ Completed Steps ({len(self.setup_results['steps_completed'])}):")
        for step in self.setup_results['steps_completed']:
            print(f"   • {step}")

        if self.setup_results['steps_failed']:
            print(f"\n❌ Failed Steps ({len(self.setup_results['steps_failed'])}):")
            for step in self.setup_results['steps_failed']:
                print(f"   • {step}")

        if self.setup_results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.setup_results['warnings'])}):")
            for warning in self.setup_results['warnings']:
                print(f"   • {warning}")

        print(f"\n🚀 Next Steps:")
        print(f"   • Run your existing training data collector with enhanced Khmer support")
        print(f"   • Test OCR processing with Cambodian ID cards")
        print(f"   • Monitor performance improvements")
        print(f"   • Generate more synthetic training data as needed")

        print(f"\n📚 Documentation:")
        print(f"   • See KHMER_INTEGRATION.md for detailed usage instructions")
        print(f"   • Check ai_improvement_plan.json for updated integration plan")


async def main():
    """Main setup function."""
    setup = KhmerIntegrationSetup()
    results = await setup.run_complete_setup()

    # Return exit code based on results
    if results['steps_failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
