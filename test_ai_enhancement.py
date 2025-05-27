#!/usr/bin/env python3
"""
Test script for AI-powered image enhancement system.

This script demonstrates and tests the advanced AI enhancement capabilities
for ultra-low quality images, with comprehensive evaluation and comparison.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_image_enhancement import (
    create_ai_enhancer, 
    enhance_ultra_low_quality_image,
    assess_enhancement_potential
)
from ai_enhancement_config import (
    get_config_by_name,
    auto_select_config,
    print_ai_config_summary
)
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


def test_ai_enhancement_single(image_path: str, output_dir: str = "ai_enhanced_output", 
                              enhancement_mode: str = "auto", use_gpu: bool = True):
    """
    Test AI enhancement on a single image with comprehensive analysis.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save enhanced images
        enhancement_mode: Enhancement mode to use
        use_gpu: Whether to use GPU acceleration
    """
    print("🤖 AI-Powered Image Enhancement Test")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"📁 Loaded image: {image_path}")
        print(f"📏 Original size: {image.size}")
        print(f"🎨 Mode: {image.mode}")
        
        # Step 1: Assess enhancement potential
        print("\n🔍 AI Quality Assessment:")
        assessment = assess_enhancement_potential(image)
        for key, value in assessment.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Step 2: Select and display configuration
        print(f"\n⚙️  Enhancement Configuration ({enhancement_mode}):")
        if enhancement_mode == "auto":
            config = auto_select_config(
                assessment['quality_score'], 
                image.size, 
                processing_priority='quality'
            )
            print("   Auto-selected configuration based on image analysis")
        else:
            config = get_config_by_name(enhancement_mode)
            print(f"   Using predefined configuration: {enhancement_mode}")
        
        print_ai_config_summary(config)
        
        # Step 3: Apply AI enhancement
        print("\n🚀 Applying AI Enhancement...")
        start_time = time.time()
        
        ai_enhancer = create_ai_enhancer(use_gpu=use_gpu)
        enhanced_image = ai_enhancer.enhance_ultra_low_quality(image)
        
        enhancement_time = time.time() - start_time
        print(f"⏱️  Enhancement completed in {enhancement_time:.2f} seconds")
        
        # Step 4: Save enhanced image
        output_path = Path(output_dir) / f"ai_enhanced_{Path(image_path).name}"
        enhanced_image.save(output_path)
        print(f"💾 Enhanced image saved: {output_path}")
        
        # Step 5: Quality comparison
        print("\n📊 Quality Comparison:")
        original_gray = np.array(image.convert('L'))
        enhanced_gray = np.array(enhanced_image.convert('L'))
        
        original_metrics = assess_image_quality_comprehensive(original_gray)
        enhanced_metrics = assess_image_quality_comprehensive(enhanced_gray)
        
        print("   Original → Enhanced:")
        for key in original_metrics:
            if key in enhanced_metrics:
                orig_val = original_metrics[key]
                enh_val = enhanced_metrics[key]
                diff = enh_val - orig_val
                improvement = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"   {improvement} {key}: {orig_val:.3f} → {enh_val:.3f} ({diff:+.3f})")
        
        # Step 6: Compare with regular enhancement
        print("\n🔄 Comparison with Regular Enhancement:")
        regular_enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)
        regular_enhanced = regular_enhancer.enhance_for_ocr(image)
        
        regular_output_path = Path(output_dir) / f"regular_enhanced_{Path(image_path).name}"
        regular_enhanced.save(regular_output_path)
        
        regular_gray = np.array(regular_enhanced)
        regular_metrics = assess_image_quality_comprehensive(regular_gray)
        
        print("   AI vs Regular Enhancement:")
        for key in enhanced_metrics:
            if key in regular_metrics:
                ai_val = enhanced_metrics[key]
                reg_val = regular_metrics[key]
                diff = ai_val - reg_val
                winner = "🤖 AI" if diff > 0 else "🔧 Regular" if diff < 0 else "🤝 Tie"
                print(f"   {winner} {key}: AI={ai_val:.3f}, Regular={reg_val:.3f} (Δ{diff:+.3f})")
        
        # Step 7: Save comparison report
        report = {
            'image_path': image_path,
            'enhancement_mode': enhancement_mode,
            'processing_time': enhancement_time,
            'original_metrics': original_metrics,
            'ai_enhanced_metrics': enhanced_metrics,
            'regular_enhanced_metrics': regular_metrics,
            'assessment': assessment,
            'config_used': config.__dict__ if hasattr(config, '__dict__') else str(config)
        }
        
        report_path = Path(output_dir) / f"ai_enhancement_report_{Path(image_path).stem}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📋 Detailed report saved: {report_path}")
        
        return enhanced_image, report
        
    except Exception as e:
        print(f"❌ Error during AI enhancement: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_ai_enhancement_modes(image_path: str, output_dir: str = "ai_modes_comparison"):
    """
    Test all AI enhancement modes on a single image for comparison.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    print("🔬 AI Enhancement Modes Comparison")
    print("=" * 60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    modes = ["ultra_low_quality", "low_quality", "khmer_optimized", "high_performance"]
    results = {}
    
    # Load original image
    image = Image.open(image_path)
    original_gray = np.array(image.convert('L'))
    original_metrics = assess_image_quality_comprehensive(original_gray)
    
    print(f"📁 Testing image: {image_path}")
    print(f"📏 Size: {image.size}")
    
    for mode in modes:
        print(f"\n🧪 Testing mode: {mode}")
        try:
            start_time = time.time()
            
            # Apply enhancement
            config = get_config_by_name(mode)
            ai_enhancer = create_ai_enhancer(use_gpu=config.use_gpu)
            enhanced_image = ai_enhancer.enhance_ultra_low_quality(image)
            
            processing_time = time.time() - start_time
            
            # Save result
            output_path = Path(output_dir) / f"{mode}_{Path(image_path).name}"
            enhanced_image.save(output_path)
            
            # Assess quality
            enhanced_gray = np.array(enhanced_image.convert('L'))
            enhanced_metrics = assess_image_quality_comprehensive(enhanced_gray)
            
            # Calculate improvement score
            improvement_score = 0
            for key in original_metrics:
                if key in enhanced_metrics:
                    if key in ['sharpness', 'std_contrast', 'edge_density']:  # Higher is better
                        improvement_score += (enhanced_metrics[key] - original_metrics[key]) / original_metrics[key]
                    elif key in ['noise_level']:  # Lower is better
                        improvement_score += (original_metrics[key] - enhanced_metrics[key]) / original_metrics[key]
            
            results[mode] = {
                'processing_time': processing_time,
                'improvement_score': improvement_score,
                'metrics': enhanced_metrics,
                'output_path': str(output_path)
            }
            
            print(f"   ⏱️  Time: {processing_time:.2f}s")
            print(f"   📈 Improvement score: {improvement_score:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[mode] = {'error': str(e)}
    
    # Generate comparison report
    print(f"\n📊 Mode Comparison Summary:")
    print("-" * 40)
    
    best_quality = max([r.get('improvement_score', -999) for r in results.values()])
    fastest = min([r.get('processing_time', 999) for r in results.values() if 'processing_time' in r])
    
    for mode, result in results.items():
        if 'error' not in result:
            score = result['improvement_score']
            time_taken = result['processing_time']
            
            quality_badge = "🏆" if score == best_quality else "⭐" if score > 0 else "📉"
            speed_badge = "⚡" if time_taken == fastest else ""
            
            print(f"{quality_badge}{speed_badge} {mode:20} | Score: {score:+.3f} | Time: {time_taken:.2f}s")
        else:
            print(f"❌ {mode:20} | Error: {result['error']}")
    
    # Save detailed comparison
    comparison_report = {
        'image_path': image_path,
        'original_metrics': original_metrics,
        'mode_results': results,
        'best_quality_mode': max(results.keys(), key=lambda k: results[k].get('improvement_score', -999)),
        'fastest_mode': min([k for k, v in results.items() if 'processing_time' in v], 
                           key=lambda k: results[k]['processing_time'])
    }
    
    report_path = Path(output_dir) / f"modes_comparison_{Path(image_path).stem}.json"
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    print(f"\n📋 Detailed comparison saved: {report_path}")
    return results


def test_ocr_improvement(image_path: str, output_dir: str = "ocr_improvement_test"):
    """
    Test OCR improvement with AI enhancement.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    print("📝 OCR Improvement Test with AI Enhancement")
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
    original_khmer = pytesseract.image_to_string(image, lang='khm', config=config)
    original_english = pytesseract.image_to_string(image, lang='eng', config=config)
    
    # Test AI enhanced image
    print("🤖 Applying AI enhancement and running OCR...")
    ai_enhanced = enhance_ultra_low_quality_image(image, use_gpu=True)
    ai_khmer = pytesseract.image_to_string(ai_enhanced, lang='khm', config=config)
    ai_english = pytesseract.image_to_string(ai_enhanced, lang='eng', config=config)
    
    # Test regular enhanced image
    print("🔧 Applying regular enhancement and running OCR...")
    regular_enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)
    regular_enhanced = regular_enhancer.enhance_for_ocr(image)
    regular_khmer = pytesseract.image_to_string(regular_enhanced, lang='khm', config=config)
    regular_english = pytesseract.image_to_string(regular_enhanced, lang='eng', config=config)
    
    # Save enhanced images
    ai_enhanced.save(Path(output_dir) / f"ai_enhanced_{Path(image_path).name}")
    regular_enhanced.save(Path(output_dir) / f"regular_enhanced_{Path(image_path).name}")
    
    # Compare results
    print("\n📊 OCR Results Comparison:")
    print("-" * 40)
    
    results = {
        'original': {'khmer': original_khmer, 'english': original_english},
        'ai_enhanced': {'khmer': ai_khmer, 'english': ai_english},
        'regular_enhanced': {'khmer': regular_khmer, 'english': regular_english}
    }
    
    for method, texts in results.items():
        khmer_len = len(texts['khmer'].strip())
        english_len = len(texts['english'].strip())
        print(f"{method:15} | Khmer: {khmer_len:3d} chars | English: {english_len:3d} chars")
    
    # Save OCR comparison report
    ocr_report_path = Path(output_dir) / f"ocr_comparison_{Path(image_path).stem}.txt"
    with open(ocr_report_path, 'w', encoding='utf-8') as f:
        f.write("OCR Improvement Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        for method, texts in results.items():
            f.write(f"{method.upper()} - KHMER OCR:\n")
            f.write("-" * 30 + "\n")
            f.write(texts['khmer'])
            f.write("\n\n")
            
            f.write(f"{method.upper()} - ENGLISH OCR:\n")
            f.write("-" * 30 + "\n")
            f.write(texts['english'])
            f.write("\n\n")
    
    print(f"📋 OCR comparison saved: {ocr_report_path}")
    return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Test AI-powered image enhancement")
    parser.add_argument("command", choices=["single", "modes", "ocr"], 
                       help="Test mode: single enhancement, mode comparison, or OCR improvement")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("--output", "-o", default="ai_test_output", 
                       help="Output directory")
    parser.add_argument("--mode", "-m", default="auto",
                       choices=["auto", "ultra_low_quality", "low_quality", "khmer_optimized", "high_performance"],
                       help="Enhancement mode for single test")
    parser.add_argument("--no-gpu", action="store_true", 
                       help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    use_gpu = not args.no_gpu
    
    if args.command == "single":
        test_ai_enhancement_single(args.input, args.output, args.mode, use_gpu)
    elif args.command == "modes":
        test_ai_enhancement_modes(args.input, args.output)
    elif args.command == "ocr":
        test_ocr_improvement(args.input, args.output)


if __name__ == "__main__":
    main()
