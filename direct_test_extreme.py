#!/usr/bin/env python3
"""
Direct test of extreme enhancement with OCR processing.
"""

import sys
import os
import asyncio
from PIL import Image
import io

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock UploadFile for testing
class MockUploadFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content_type = "image/jpeg"
    
    async def read(self):
        with open(self.file_path, 'rb') as f:
            return f.read()

async def test_extreme_ocr_processing():
    """Test extreme OCR processing directly."""
    print("🧪 Direct Extreme OCR Processing Test")
    print("=" * 60)
    
    try:
        # Import the OCR controller
        from controllers.ocr_controller import process_cambodian_id_ocr
        
        # Create mock upload file
        mock_file = MockUploadFile("id_card.jpg")
        
        print("📁 Testing with id_card.jpg")
        
        # Test 1: Standard enhanced preprocessing
        print("\n🔧 Test 1: Enhanced Preprocessing")
        result1 = await process_cambodian_id_ocr(
            mock_file,
            use_enhanced_preprocessing=True,
            use_ai_enhancement=False,
            use_extreme_enhancement=False,
            use_robust_parsing=True
        )
        
        print("Results:")
        for field, value in result1.dict().items():
            if value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ✅ {field}: {value}")
            elif not value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ❌ {field}: Not found")
        
        # Test 2: AI Enhancement
        print("\n🤖 Test 2: AI Enhancement")
        result2 = await process_cambodian_id_ocr(
            mock_file,
            use_enhanced_preprocessing=False,
            use_ai_enhancement=True,
            use_extreme_enhancement=False,
            enhancement_mode="ultra_low_quality",
            use_robust_parsing=True
        )
        
        print("Results:")
        for field, value in result2.dict().items():
            if value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ✅ {field}: {value}")
            elif not value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ❌ {field}: Not found")
        
        # Test 3: Extreme Enhancement
        print("\n💥 Test 3: Extreme Enhancement")
        result3 = await process_cambodian_id_ocr(
            mock_file,
            use_enhanced_preprocessing=False,
            use_ai_enhancement=False,
            use_extreme_enhancement=True,
            use_robust_parsing=True
        )
        
        print("Results:")
        for field, value in result3.dict().items():
            if value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ✅ {field}: {value}")
            elif not value and field not in ['raw_khmer', 'raw_english']:
                print(f"   ❌ {field}: Not found")
        
        # Compare results
        print("\n📊 Comparison Summary:")
        print("-" * 60)
        print(f"{'Field':<15} | {'Enhanced':<12} | {'AI':<12} | {'Extreme':<12}")
        print("-" * 60)
        
        fields = ['full_name', 'id_number', 'date_of_birth', 'gender']
        for field in fields:
            val1 = "✅" if getattr(result1, field) else "❌"
            val2 = "✅" if getattr(result2, field) else "❌"
            val3 = "✅" if getattr(result3, field) else "❌"
            print(f"{field:<15} | {val1:<12} | {val2:<12} | {val3:<12}")
        
        # Show raw OCR lengths
        print("\n📝 Raw OCR Text Lengths:")
        print(f"Enhanced:  Khmer={len(result1.raw_khmer)}, English={len(result1.raw_english)}")
        print(f"AI:        Khmer={len(result2.raw_khmer)}, English={len(result2.raw_english)}")
        print(f"Extreme:   Khmer={len(result3.raw_khmer)}, English={len(result3.raw_english)}")
        
        return result1, result2, result3
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

async def test_with_different_modes():
    """Test different enhancement modes."""
    print("\n🎯 Testing Different Enhancement Modes")
    print("=" * 60)
    
    try:
        from controllers.ocr_controller import process_cambodian_id_ocr
        mock_file = MockUploadFile("id_card.jpg")
        
        modes = ["auto", "ultra_low_quality", "khmer_optimized", "high_performance"]
        results = {}
        
        for mode in modes:
            print(f"\n🔧 Testing mode: {mode}")
            try:
                result = await process_cambodian_id_ocr(
                    mock_file,
                    use_enhanced_preprocessing=False,
                    use_ai_enhancement=True,
                    use_extreme_enhancement=False,
                    enhancement_mode=mode,
                    use_robust_parsing=True
                )
                
                extracted_fields = sum(1 for field in ['full_name', 'id_number', 'date_of_birth', 'gender'] 
                                     if getattr(result, field))
                results[mode] = {
                    'extracted_fields': extracted_fields,
                    'khmer_length': len(result.raw_khmer),
                    'english_length': len(result.raw_english),
                    'full_name': getattr(result, 'full_name', None)
                }
                
                print(f"   📊 Extracted {extracted_fields}/4 fields")
                if result.full_name:
                    print(f"   👤 Name: {result.full_name}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[mode] = {'error': str(e)}
        
        # Summary
        print("\n📈 Mode Comparison:")
        print("-" * 50)
        for mode, result in results.items():
            if 'error' not in result:
                fields = result['extracted_fields']
                print(f"{mode:<20} | {fields}/4 fields extracted")
            else:
                print(f"{mode:<20} | Error: {result['error']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during mode testing: {e}")
        return {}

def main():
    """Main test function."""
    print("🚀 Extreme Enhancement Direct Test Suite")
    print("=" * 70)
    
    # Test basic processing
    results = asyncio.run(test_extreme_ocr_processing())
    
    # Test different modes
    mode_results = asyncio.run(test_with_different_modes())
    
    print("\n✨ Testing Complete!")
    print("\n💡 Recommendations:")
    
    if results and results[0]:
        # Analyze which method worked best
        enhanced_fields = sum(1 for field in ['full_name', 'id_number', 'date_of_birth', 'gender'] 
                            if getattr(results[0], field))
        ai_fields = sum(1 for field in ['full_name', 'id_number', 'date_of_birth', 'gender'] 
                       if getattr(results[1], field)) if results[1] else 0
        extreme_fields = sum(1 for field in ['full_name', 'id_number', 'date_of_birth', 'gender'] 
                           if getattr(results[2], field)) if results[2] else 0
        
        best_method = "Enhanced"
        best_count = enhanced_fields
        
        if ai_fields > best_count:
            best_method = "AI"
            best_count = ai_fields
        
        if extreme_fields > best_count:
            best_method = "Extreme"
            best_count = extreme_fields
        
        print(f"   🏆 Best method: {best_method} ({best_count}/4 fields)")
        
        if best_count < 4:
            print("   🔧 For better results, try:")
            print("     - Use extreme_enhancement=true for severely damaged images")
            print("     - Try different enhancement_mode values")
            print("     - Ensure image quality is sufficient")
            print("     - Check if Tesseract Khmer language pack is properly installed")

if __name__ == "__main__":
    main()
