#!/usr/bin/env python3
"""
AI Enhancement Demo Script

This script demonstrates the AI-powered image enhancement capabilities
with a simple, easy-to-use interface for testing ultra-low quality images.
"""

import os
import sys
from PIL import Image
import numpy as np
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_image_enhancement import (
    create_ai_enhancer, 
    enhance_ultra_low_quality_image,
    assess_enhancement_potential
)
from ai_enhancement_config import get_config_by_name, print_ai_config_summary


def demo_ai_enhancement(image_path: str):
    """
    Demonstrate AI enhancement with step-by-step output.
    
    Args:
        image_path: Path to input image
    """
    print("🤖 AI-Powered Image Enhancement Demo")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        # Step 1: Load and analyze image
        print("📁 Loading image...")
        image = Image.open(image_path)
        print(f"   Size: {image.size}")
        print(f"   Mode: {image.mode}")
        
        # Step 2: AI quality assessment
        print("\n🔍 AI Quality Assessment...")
        assessment = assess_enhancement_potential(image)
        
        quality_score = assessment['quality_score']
        print(f"   📊 Quality Score: {quality_score:.3f}/1.0")
        
        if quality_score < 0.3:
            quality_level = "Ultra-Low Quality 🔴"
            recommended_mode = "ultra_low_quality"
        elif quality_score < 0.6:
            quality_level = "Low Quality 🟡"
            recommended_mode = "low_quality"
        else:
            quality_level = "Good Quality 🟢"
            recommended_mode = "khmer_optimized"
        
        print(f"   📈 Quality Level: {quality_level}")
        print(f"   🎯 Recommended Mode: {recommended_mode}")
        print(f"   🔧 Should Enhance: {'Yes' if assessment['should_enhance'] else 'No'}")
        print(f"   ⚡ Processing Priority: {assessment['enhancement_priority']}")
        
        # Step 3: Show configuration
        print(f"\n⚙️  AI Enhancement Configuration ({recommended_mode}):")
        config = get_config_by_name(recommended_mode)
        print(f"   🎯 Target DPI: {config.target_dpi}")
        print(f"   🔍 Super-Resolution: {'Enabled' if config.enable_super_resolution else 'Disabled'}")
        if config.enable_super_resolution:
            print(f"   📏 Scale Factor: {config.sr_scale_factor}x")
        print(f"   🧹 AI Denoising: {'Enabled' if config.enable_ai_denoising else 'Disabled'}")
        if config.enable_ai_denoising:
            print(f"   💪 Denoising Strength: {config.denoising_strength}")
        print(f"   🌟 Contrast Boost: {config.contrast_boost_factor}x")
        print(f"   🎨 Text Enhancement: {config.text_enhancement_strength}x")
        print(f"   🖥️  GPU Acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
        
        # Step 4: Apply AI enhancement
        print("\n🚀 Applying AI Enhancement...")
        start_time = time.time()
        
        # Create AI enhancer
        ai_enhancer = create_ai_enhancer(use_gpu=config.use_gpu)
        
        # Apply enhancement
        enhanced_image = ai_enhancer.enhance_ultra_low_quality(image)
        
        processing_time = time.time() - start_time
        print(f"   ⏱️  Processing completed in {processing_time:.2f} seconds")
        print(f"   📏 Enhanced size: {enhanced_image.size}")
        
        # Step 5: Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save enhanced image
        enhanced_path = f"ai_enhanced_{base_name}.png"
        enhanced_image.save(enhanced_path)
        print(f"   💾 Enhanced image saved: {enhanced_path}")
        
        # Step 6: Quality comparison
        print("\n📊 Quality Improvement Analysis:")
        
        # Convert to grayscale for analysis
        original_gray = np.array(image.convert('L'))
        enhanced_gray = np.array(enhanced_image.convert('L'))
        
        # Calculate improvements
        original_contrast = np.std(original_gray)
        enhanced_contrast = np.std(enhanced_gray)
        contrast_improvement = ((enhanced_contrast - original_contrast) / original_contrast) * 100
        
        original_sharpness = np.var(np.gradient(original_gray.astype(float)))
        enhanced_sharpness = np.var(np.gradient(enhanced_gray.astype(float)))
        sharpness_improvement = ((enhanced_sharpness - original_sharpness) / original_sharpness) * 100
        
        print(f"   📈 Contrast Improvement: {contrast_improvement:+.1f}%")
        print(f"   🔍 Sharpness Improvement: {sharpness_improvement:+.1f}%")
        
        # Step 7: OCR readiness assessment
        print("\n📝 OCR Readiness Assessment:")
        final_assessment = assess_enhancement_potential(enhanced_image)
        final_score = final_assessment['quality_score']
        improvement = final_score - quality_score
        
        print(f"   📊 Final Quality Score: {final_score:.3f}/1.0 ({improvement:+.3f})")
        
        if final_score > 0.7:
            ocr_readiness = "Excellent for OCR 🟢"
        elif final_score > 0.5:
            ocr_readiness = "Good for OCR 🟡"
        elif final_score > 0.3:
            ocr_readiness = "Fair for OCR 🟠"
        else:
            ocr_readiness = "May need additional processing 🔴"
        
        print(f"   🎯 OCR Readiness: {ocr_readiness}")
        
        # Step 8: Recommendations
        print("\n💡 Recommendations:")
        if final_score > 0.7:
            print("   ✅ Image is ready for high-accuracy OCR")
            print("   🎯 Use standard Tesseract settings")
        elif final_score > 0.5:
            print("   ✅ Image should work well with OCR")
            print("   🎯 Consider using PSM 11 for sparse text")
        elif final_score > 0.3:
            print("   ⚠️  Image may have some OCR challenges")
            print("   🎯 Try different PSM modes (6, 8, 11)")
            print("   🔧 Consider manual review of results")
        else:
            print("   ⚠️  Image still has quality issues")
            print("   🎯 Try ultra_low_quality mode if not used")
            print("   🔧 Manual preprocessing may be needed")
        
        # Step 9: Next steps
        print("\n🚀 Next Steps:")
        print("   1. Test OCR with enhanced image:")
        print(f"      pytesseract.image_to_string('{enhanced_path}', lang='khm+eng')")
        print("   2. Compare with original OCR results")
        print("   3. Fine-tune enhancement mode if needed")
        print("   4. Use via API for production:")
        print("      POST /ocr/idcard?ai_enhancement=true&enhancement_mode=" + recommended_mode)
        
        print(f"\n✨ AI Enhancement Demo Complete!")
        print(f"📁 Enhanced image: {enhanced_path}")
        
        return enhanced_image
        
    except Exception as e:
        print(f"❌ Error during AI enhancement: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_enhance(image_path: str, mode: str = "auto"):
    """
    Quick enhancement without detailed output.
    
    Args:
        image_path: Path to input image
        mode: Enhancement mode
        
    Returns:
        Enhanced PIL Image
    """
    try:
        image = Image.open(image_path)
        
        if mode == "auto":
            # Auto-select mode based on quality
            assessment = assess_enhancement_potential(image)
            if assessment['quality_score'] < 0.3:
                mode = "ultra_low_quality"
            elif assessment['quality_score'] < 0.6:
                mode = "low_quality"
            else:
                mode = "khmer_optimized"
        
        # Apply enhancement
        enhanced = enhance_ultra_low_quality_image(image, use_gpu=True)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"quick_enhanced_{base_name}.png"
        enhanced.save(output_path)
        
        print(f"✅ Quick enhancement complete: {output_path}")
        return enhanced
        
    except Exception as e:
        print(f"❌ Quick enhancement failed: {e}")
        return None


def main():
    """Main demo function."""
    print("🤖 AI Image Enhancement Demo")
    print("Choose an option:")
    print("1. Full demo with detailed analysis")
    print("2. Quick enhancement")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if image_path:
                demo_ai_enhancement(image_path)
            break
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            mode = input("Enter mode (auto/ultra_low_quality/low_quality/khmer_optimized) [auto]: ").strip() or "auto"
            if image_path:
                quick_enhance(image_path, mode)
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        image_path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
        
        if mode == "quick":
            quick_enhance(image_path)
        else:
            demo_ai_enhancement(image_path)
    else:
        # Interactive mode
        main()
