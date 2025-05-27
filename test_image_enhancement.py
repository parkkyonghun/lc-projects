#!/usr/bin/env python3
"""
Test script for image enhancement utilities.

This script demonstrates the image enhancement capabilities and provides
tools for testing and validating OCR preprocessing improvements.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_enhancement_utils import (
    create_preprocessing_pipeline,
    assess_image_quality_comprehensive
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhancement_pipeline(image_path: str, output_dir: str = "enhanced_output", debug: bool = True):
    """
    Test the complete image enhancement pipeline.

    Args:
        image_path: Path to input image
        output_dir: Directory to save enhanced images
        debug: Enable debug mode to save intermediate steps
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load image
    try:
        image = Image.open(image_path)
        logger.info(f"Loaded image: {image_path} ({image.size})")
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return

    # Create enhancer
    enhancer = create_preprocessing_pipeline(target_dpi=300, debug=debug)

    # Assess original image quality
    original_cv = np.array(image.convert('L'))
    original_metrics = assess_image_quality_comprehensive(original_cv)
    logger.info("Original image quality metrics:")
    for key, value in original_metrics.items():
        logger.info(f"  {key}: {value:.3f}")

    # Apply enhancement
    try:
        enhanced_image = enhancer.enhance_for_ocr(image)

        # Save enhanced image
        output_path = os.path.join(output_dir, f"enhanced_{Path(image_path).name}")
        enhanced_image.save(output_path)
        logger.info(f"Enhanced image saved: {output_path}")

        # Assess enhanced image quality
        enhanced_cv = np.array(enhanced_image)
        enhanced_metrics = assess_image_quality_comprehensive(enhanced_cv)
        logger.info("Enhanced image quality metrics:")
        for key, value in enhanced_metrics.items():
            logger.info(f"  {key}: {value:.3f}")

        # Compare metrics
        logger.info("Quality improvement:")
        for key in original_metrics:
            if key in enhanced_metrics:
                diff = enhanced_metrics[key] - original_metrics[key]
                logger.info(f"  {key}: {diff:+.3f}")

        return enhanced_image, original_metrics, enhanced_metrics

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        return None, None, None


def batch_test_images(input_dir: str, output_dir: str = "batch_enhanced"):
    """
    Test enhancement on a batch of images.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Find all image files
    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in extensions]

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    results = []
    for image_file in image_files:
        logger.info(f"Processing: {image_file.name}")

        result = test_enhancement_pipeline(
            str(image_file),
            str(output_path),
            debug=False  # Disable debug for batch processing
        )

        if result and result[0] is not None:
            results.append({
                'filename': image_file.name,
                'original_metrics': result[1],
                'enhanced_metrics': result[2]
            })

    # Generate summary report
    generate_batch_report(results, output_path)


def generate_batch_report(results: list, output_dir: Path):
    """Generate a summary report for batch processing."""
    if not results:
        return

    report_path = output_dir / "enhancement_report.txt"

    with open(report_path, 'w') as f:
        f.write("Image Enhancement Batch Report\n")
        f.write("=" * 50 + "\n\n")

        # Calculate average improvements
        metrics_keys = list(results[0]['original_metrics'].keys())
        avg_improvements = {}

        for key in metrics_keys:
            improvements = []
            for result in results:
                original = result['original_metrics'][key]
                enhanced = result['enhanced_metrics'][key]
                improvements.append(enhanced - original)
            avg_improvements[key] = np.mean(improvements)

        f.write("Average Quality Improvements:\n")
        f.write("-" * 30 + "\n")
        for key, improvement in avg_improvements.items():
            f.write(f"{key}: {improvement:+.3f}\n")

        f.write("\n\nIndividual Results:\n")
        f.write("-" * 30 + "\n")

        for result in results:
            f.write(f"\nFile: {result['filename']}\n")
            f.write("Improvements:\n")
            for key in metrics_keys:
                original = result['original_metrics'][key]
                enhanced = result['enhanced_metrics'][key]
                improvement = enhanced - original
                f.write(f"  {key}: {original:.3f} -> {enhanced:.3f} ({improvement:+.3f})\n")

    logger.info(f"Batch report saved: {report_path}")


def compare_with_tesseract(image_path: str, output_dir: str = "ocr_comparison"):
    """
    Compare OCR results before and after enhancement.

    Args:
        image_path: Path to input image
        output_dir: Directory to save comparison results
    """
    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not available for OCR comparison")
        return

    Path(output_dir).mkdir(exist_ok=True)

    # Load and process image
    image = Image.open(image_path)
    enhancer = create_preprocessing_pipeline(debug=False)
    enhanced_image = enhancer.enhance_for_ocr(image)

    # OCR configuration optimized for Khmer
    config = '--oem 1 --psm 11 -c preserve_interword_spaces=1'

    # Run OCR on original image
    original_text_khm = pytesseract.image_to_string(image, lang='khm', config=config)
    original_text_eng = pytesseract.image_to_string(image, lang='eng', config=config)

    # Run OCR on enhanced image
    enhanced_text_khm = pytesseract.image_to_string(enhanced_image, lang='khm', config=config)
    enhanced_text_eng = pytesseract.image_to_string(enhanced_image, lang='eng', config=config)

    # Save results
    comparison_file = Path(output_dir) / f"ocr_comparison_{Path(image_path).stem}.txt"

    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("OCR Comparison Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("ORIGINAL IMAGE - Khmer OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(original_text_khm)
        f.write("\n\n")

        f.write("ORIGINAL IMAGE - English OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(original_text_eng)
        f.write("\n\n")

        f.write("ENHANCED IMAGE - Khmer OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(enhanced_text_khm)
        f.write("\n\n")

        f.write("ENHANCED IMAGE - English OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(enhanced_text_eng)
        f.write("\n\n")

        # Simple comparison metrics
        f.write("COMPARISON METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Original Khmer text length: {len(original_text_khm.strip())}\n")
        f.write(f"Enhanced Khmer text length: {len(enhanced_text_khm.strip())}\n")
        f.write(f"Original English text length: {len(original_text_eng.strip())}\n")
        f.write(f"Enhanced English text length: {len(enhanced_text_eng.strip())}\n")

    logger.info(f"OCR comparison saved: {comparison_file}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Test image enhancement for OCR")
    parser.add_argument("command", choices=["single", "batch", "ocr"],
                       help="Test mode: single image, batch processing, or OCR comparison")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", "-o", default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    if args.command == "single":
        test_enhancement_pipeline(args.input, args.output, args.debug)
    elif args.command == "batch":
        batch_test_images(args.input, args.output)
    elif args.command == "ocr":
        compare_with_tesseract(args.input, args.output)


if __name__ == "__main__":
    main()
