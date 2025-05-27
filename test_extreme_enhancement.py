#!/usr/bin/env python3
"""
Test script for extreme image enhancement system.

This script tests the most aggressive enhancement techniques
for severely damaged, ultra-low quality images.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extreme_enhancement import (
    ExtremeImageEnhancer,
    enhance_with_multiple_approaches,
    get_best_enhanced_image
)
from robust_ocr_parser import parse_ocr_robust

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_extreme_enhancement(image_path: str, output_dir: str = "extreme_test_output"):
    """
    Test extreme enhancement on a severely damaged image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    print("💥 Extreme Image Enhancement Test")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"📁 Loaded image: {image_path}")
        print(f"📏 Original size: {image.size}")
        print(f"🎨 Mode: {image.mode}")
        
        # Test extreme enhancement
        print("\n💥 Applying Extreme Enhancement...")
        start_time = time.time()
        
        # Get multiple enhanced versions
        enhanced_versions = enhance_with_multiple_approaches(image, debug=True)
        
        processing_time = time.time() - start_time
        print(f"⏱️  Generated {len(enhanced_versions)} versions in {processing_time:.2f} seconds")
        
        # Save all versions
        version_names = [
            "extreme_contrast",
            "extreme_denoising", 
            "extreme_super_resolution",
            "extreme_text_enhancement",
            "extreme_multi_scale",
            "extreme_frequency"
        ]
        
        for i, (enhanced, name) in enumerate(zip(enhanced_versions, version_names)):
            output_path = Path(output_dir) / f"{name}_{Path(image_path).name}"
            enhanced.save(output_path)
            print(f"💾 Saved {name}: {output_path}")
        
        # Get best version
        print("\n🏆 Selecting Best Enhanced Version...")
        best_enhanced = get_best_enhanced_image(image)
        best_path = Path(output_dir) / f"best_enhanced_{Path(image_path).name}"
        best_enhanced.save(best_path)
        print(f"🥇 Best enhanced image saved: {best_path}")
        
        # Quality comparison
        print("\n📊 Quality Analysis:")
        original_gray = np.array(image.convert('L'))
        best_gray = np.array(best_enhanced.convert('L'))
        
        # Calculate improvements
        original_contrast = np.std(original_gray)
        enhanced_contrast = np.std(best_gray)
        contrast_improvement = ((enhanced_contrast - original_contrast) / original_contrast) * 100
        
        print(f"   📈 Contrast Improvement: {contrast_improvement:+.1f}%")
        print(f"   📏 Size Change: {image.size} → {best_enhanced.size}")
        
        return enhanced_versions, best_enhanced
        
    except Exception as e:
        print(f"❌ Error during extreme enhancement: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_ocr_with_extreme_enhancement(image_path: str, output_dir: str = "extreme_ocr_test"):
    """
    Test OCR improvement with extreme enhancement.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    print("📝 OCR Test with Extreme Enhancement")
    print("=" * 60)
    
    try:
        import pytesseract
    except ImportError:
        print("⚠️  pytesseract not available. Install with: pip install pytesseract")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load image
    image = Image.open(image_path)
    print(f"📁 Testing OCR on: {image_path}")
    
    # OCR configuration
    config = '--oem 1 --psm 11 -c preserve_interword_spaces=1 --dpi 300'
    
    # Test original image
    print("\n🔍 OCR on original image...")
    try:
        original_khmer = pytesseract.image_to_string(image, lang='khm', config=config)
        original_english = pytesseract.image_to_string(image, lang='eng', config=config)
        print(f"   Khmer: {len(original_khmer)} chars")
        print(f"   English: {len(original_english)} chars")
    except Exception as e:
        print(f"   ❌ Original OCR failed: {e}")
        original_khmer = ""
        original_english = ""
    
    # Test extreme enhanced image
    print("\n💥 Applying extreme enhancement and running OCR...")
    try:
        extreme_enhanced = get_best_enhanced_image(image)
        extreme_enhanced.save(Path(output_dir) / f"extreme_enhanced_{Path(image_path).name}")
        
        extreme_khmer = pytesseract.image_to_string(extreme_enhanced, lang='khm', config=config)
        extreme_english = pytesseract.image_to_string(extreme_enhanced, lang='eng', config=config)
        
        print(f"   Khmer: {len(extreme_khmer)} chars")
        print(f"   English: {len(extreme_english)} chars")
        
        # Test robust parsing
        print("\n🧠 Testing Robust Parsing...")
        parsed_data = parse_ocr_robust(extreme_khmer, extreme_english)
        
        print("   Extracted Fields:")
        for field, value in parsed_data.items():
            if value:
                print(f"     ✅ {field}: {value}")
            else:
                print(f"     ❌ {field}: Not found")
        
    except Exception as e:
        print(f"   ❌ Extreme enhancement OCR failed: {e}")
        extreme_khmer = ""
        extreme_english = ""
        parsed_data = {}
    
    # Compare results
    print("\n📊 OCR Results Comparison:")
    print("-" * 40)
    print(f"{'Method':<20} | {'Khmer':<8} | {'English':<8}")
    print("-" * 40)
    print(f"{'Original':<20} | {len(original_khmer):<8} | {len(original_english):<8}")
    print(f"{'Extreme Enhanced':<20} | {len(extreme_khmer):<8} | {len(extreme_english):<8}")
    
    # Save detailed results
    results_file = Path(output_dir) / f"extreme_ocr_results_{Path(image_path).stem}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Extreme Enhancement OCR Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ORIGINAL IMAGE - Khmer OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(original_khmer)
        f.write("\n\n")
        
        f.write("ORIGINAL IMAGE - English OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(original_english)
        f.write("\n\n")
        
        f.write("EXTREME ENHANCED - Khmer OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(extreme_khmer)
        f.write("\n\n")
        
        f.write("EXTREME ENHANCED - English OCR:\n")
        f.write("-" * 30 + "\n")
        f.write(extreme_english)
        f.write("\n\n")
        
        f.write("ROBUST PARSING RESULTS:\n")
        f.write("-" * 30 + "\n")
        for field, value in parsed_data.items():
            f.write(f"{field}: {value}\n")
    
    print(f"📋 Detailed results saved: {results_file}")
    return parsed_data


def test_api_extreme_enhancement(image_path: str):
    """
    Test extreme enhancement via API.
    
    Args:
        image_path: Path to input image
    """
    print("🌐 API Test with Extreme Enhancement")
    print("=" * 60)
    
    try:
        import requests
        
        # Test API endpoint
        url = "http://localhost:8000/ocr/idcard"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'extreme_enhancement': True,
                'robust_parsing': True,
                'enhancement_mode': 'ultra_low_quality'
            }
            
            print(f"📡 Sending request to {url}")
            print(f"🔧 Parameters: {params}")
            
            response = requests.post(url, files=files, params=params)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API request successful!")
                print("\n📋 Extracted Fields:")
                for field, value in result.items():
                    if value and field != 'raw_khmer' and field != 'raw_english':
                        print(f"   ✅ {field}: {value}")
                    elif not value and field != 'raw_khmer' and field != 'raw_english':
                        print(f"   ❌ {field}: Not found")
                
                print(f"\n📝 Raw OCR lengths:")
                print(f"   Khmer: {len(result.get('raw_khmer', ''))} chars")
                print(f"   English: {len(result.get('raw_english', ''))} chars")
                
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
    except ImportError:
        print("⚠️  requests library not available. Install with: pip install requests")
    except Exception as e:
        print(f"❌ API test failed: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Test extreme image enhancement")
    parser.add_argument("command", choices=["enhance", "ocr", "api"], 
                       help="Test mode: enhance only, OCR test, or API test")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("--output", "-o", default="extreme_test_output", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    if args.command == "enhance":
        test_extreme_enhancement(args.input, args.output)
    elif args.command == "ocr":
        test_ocr_with_extreme_enhancement(args.input, args.output)
    elif args.command == "api":
        test_api_extreme_enhancement(args.input)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("💥 Extreme Enhancement Test Suite")
        print("=" * 50)
        print("Usage examples:")
        print("  python test_extreme_enhancement.py enhance image.jpg")
        print("  python test_extreme_enhancement.py ocr image.jpg")
        print("  python test_extreme_enhancement.py api image.jpg")
        print("\nThis tests the most aggressive enhancement techniques")
        print("for severely damaged, ultra-low quality images.")
