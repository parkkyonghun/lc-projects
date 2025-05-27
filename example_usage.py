#!/usr/bin/env python3
"""
Example usage of the enhanced image preprocessing pipeline for Tesseract OCR.

This script demonstrates how to use the new image enhancement features
to improve OCR accuracy on Cambodian ID cards and other documents.
"""

import os
import sys
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_enhancement_utils import create_preprocessing_pipeline, assess_image_quality_comprehensive
from enhancement_config import get_config_by_name


def demonstrate_enhancement(image_path: str):
    """
    Demonstrate the image enhancement pipeline with before/after comparison.
    
    Args:
        image_path: Path to input image
    """
    print("🔍 Image Enhancement Demonstration")
    print("=" * 50)
    
    try:
        # Load the image
        image = Image.open(image_path)
        print(f"📁 Loaded image: {image_path}")
        print(f"📏 Original size: {image.size}")
        print(f"🎨 Mode: {image.mode}")
        
        # Convert to grayscale for quality assessment
        original_gray = np.array(image.convert('L'))
        
        # Assess original image quality
        print("\n📊 Original Image Quality:")
        original_metrics = assess_image_quality_comprehensive(original_gray)
        for key, value in original_metrics.items():
            print(f"   {key}: {value:.3f}")
        
        # Create enhanced preprocessing pipeline
        print("\n🔧 Applying Enhanced Preprocessing...")
        enhancer = create_preprocessing_pipeline(target_dpi=300, debug=True)
        
        # Apply enhancement
        enhanced_image = enhancer.enhance_for_ocr(image)
        
        # Assess enhanced image quality
        enhanced_gray = np.array(enhanced_image)
        print("\n📊 Enhanced Image Quality:")
        enhanced_metrics = assess_image_quality_comprehensive(enhanced_gray)
        for key, value in enhanced_metrics.items():
            print(f"   {key}: {value:.3f}")
        
        # Show improvements
        print("\n📈 Quality Improvements:")
        for key in original_metrics:
            if key in enhanced_metrics:
                diff = enhanced_metrics[key] - original_metrics[key]
                improvement = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"   {improvement} {key}: {diff:+.3f}")
        
        # Save enhanced image
        output_path = f"enhanced_{os.path.basename(image_path)}"
        enhanced_image.save(output_path)
        print(f"\n💾 Enhanced image saved: {output_path}")
        
        # List debug images if created
        debug_files = [f for f in os.listdir('.') if f.startswith('debug_')]
        if debug_files:
            print(f"\n🐛 Debug images created: {len(debug_files)} files")
            for debug_file in sorted(debug_files):
                print(f"   📄 {debug_file}")
        
        return enhanced_image, original_metrics, enhanced_metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None


def demonstrate_configurations():
    """Demonstrate different configuration options."""
    print("\n🛠️  Available Configurations")
    print("=" * 50)
    
    configs = {
        "khmer_id": "Optimized for Cambodian ID cards",
        "general": "General document processing",
        "low_quality": "For poor quality/noisy images",
        "high_resolution": "For high-resolution scanned documents"
    }
    
    for config_name, description in configs.items():
        try:
            config = get_config_by_name(config_name)
            print(f"\n📋 {config_name.upper()}:")
            print(f"   📝 {description}")
            print(f"   🎯 Target DPI: {config.resize.target_dpi}")
            print(f"   📐 Min size: {config.resize.min_width}x{config.resize.min_height}")
            print(f"   🔧 CLAHE clip: {config.contrast.clahe_clip_limit}")
            print(f"   🌟 Gamma: {config.contrast.gamma_correction}")
        except Exception as e:
            print(f"   ❌ Error loading config: {e}")


def demonstrate_tesseract_integration(image_path: str):
    """
    Demonstrate OCR integration with enhanced preprocessing.
    
    Args:
        image_path: Path to input image
    """
    print("\n🔤 Tesseract OCR Integration")
    print("=" * 50)
    
    try:
        import pytesseract
        
        # Load and enhance image
        image = Image.open(image_path)
        enhancer = create_preprocessing_pipeline(debug=False)
        enhanced_image = enhancer.enhance_for_ocr(image)
        
        # OCR configuration optimized for Khmer
        config = (
            '--oem 1 --psm 11 '
            '-c preserve_interword_spaces=1 '
            '--dpi 300 '
            '-c load_system_dawg=0 '
            '-c load_freq_dawg=0'
        )
        
        print("🔍 Running OCR on original image...")
        original_khmer = pytesseract.image_to_string(image, lang='khm', config=config)
        original_english = pytesseract.image_to_string(image, lang='eng', config=config)
        
        print("🔍 Running OCR on enhanced image...")
        enhanced_khmer = pytesseract.image_to_string(enhanced_image, lang='khm', config=config)
        enhanced_english = pytesseract.image_to_string(enhanced_image, lang='eng', config=config)
        
        print("\n📊 OCR Results Comparison:")
        print(f"   Original Khmer text length: {len(original_khmer.strip())} chars")
        print(f"   Enhanced Khmer text length: {len(enhanced_khmer.strip())} chars")
        print(f"   Original English text length: {len(original_english.strip())} chars")
        print(f"   Enhanced English text length: {len(enhanced_english.strip())} chars")
        
        # Show sample text (first 100 chars)
        if enhanced_khmer.strip():
            print(f"\n📝 Sample Khmer text: {enhanced_khmer.strip()[:100]}...")
        if enhanced_english.strip():
            print(f"📝 Sample English text: {enhanced_english.strip()[:100]}...")
            
    except ImportError:
        print("⚠️  pytesseract not available. Install with: pip install pytesseract")
    except Exception as e:
        print(f"❌ OCR Error: {e}")


def main():
    """Main demonstration function."""
    print("🚀 Enhanced Image Preprocessing for Tesseract OCR")
    print("=" * 60)
    print("This demonstration shows the improved image preprocessing pipeline")
    print("designed to optimize images for better OCR accuracy.\n")
    
    # Check if an image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Demonstrate enhancement
            result = demonstrate_enhancement(image_path)
            
            # Demonstrate OCR integration if enhancement was successful
            if result[0] is not None:
                demonstrate_tesseract_integration(image_path)
        else:
            print(f"❌ Image file not found: {image_path}")
    else:
        print("💡 Usage: python example_usage.py <image_path>")
        print("   Example: python example_usage.py cambodian_id.jpg")
    
    # Always show available configurations
    demonstrate_configurations()
    
    print("\n✨ Enhancement pipeline demonstration complete!")
    print("\n📚 For more information, see IMAGE_ENHANCEMENT_README.md")


if __name__ == "__main__":
    main()
