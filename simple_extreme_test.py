#!/usr/bin/env python3
"""
Simple test of extreme enhancement.
"""

import sys
import os
from PIL import Image
import pytesseract

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_extreme():
    """Simple test of extreme enhancement."""
    print("🧪 Simple Extreme Enhancement Test")
    print("=" * 50)
    
    try:
        # Import modules
        from extreme_enhancement import get_best_enhanced_image
        from robust_ocr_parser import parse_ocr_robust
        
        # Load image
        image = Image.open("id_card.jpg")
        print(f"📁 Loaded image: {image.size}")
        
        # Apply extreme enhancement
        print("💥 Applying extreme enhancement...")
        enhanced_image = get_best_enhanced_image(image)
        enhanced_image.save("simple_extreme_enhanced.jpg")
        print(f"💾 Saved enhanced image: simple_extreme_enhanced.jpg")
        
        # Run OCR on original
        print("\n🔍 OCR on original image...")
        config = '--oem 1 --psm 11 -c preserve_interword_spaces=1 --dpi 300'
        
        try:
            original_khmer = pytesseract.image_to_string(image, lang='khm', config=config)
            original_english = pytesseract.image_to_string(image, lang='eng', config=config)
            print(f"   Original - Khmer: {len(original_khmer)} chars")
            print(f"   Original - English: {len(original_english)} chars")
        except Exception as e:
            print(f"   ❌ Original OCR failed: {e}")
            original_khmer = ""
            original_english = ""
        
        # Run OCR on enhanced
        print("\n🔍 OCR on enhanced image...")
        try:
            enhanced_khmer = pytesseract.image_to_string(enhanced_image, lang='khm', config=config)
            enhanced_english = pytesseract.image_to_string(enhanced_image, lang='eng', config=config)
            print(f"   Enhanced - Khmer: {len(enhanced_khmer)} chars")
            print(f"   Enhanced - English: {len(enhanced_english)} chars")
        except Exception as e:
            print(f"   ❌ Enhanced OCR failed: {e}")
            enhanced_khmer = ""
            enhanced_english = ""
        
        # Test robust parsing on enhanced text
        print("\n🧠 Testing robust parsing...")
        if enhanced_khmer or enhanced_english:
            parsed_data = parse_ocr_robust(enhanced_khmer, enhanced_english)
            
            print("   Extracted fields:")
            for field, value in parsed_data.items():
                if value:
                    print(f"     ✅ {field}: {value}")
                else:
                    print(f"     ❌ {field}: Not found")
        else:
            print("   ⚠️  No OCR text to parse")
        
        # Save raw OCR results
        with open("simple_extreme_ocr_results.txt", "w", encoding="utf-8") as f:
            f.write("Simple Extreme Enhancement OCR Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("ORIGINAL KHMER:\n")
            f.write(original_khmer)
            f.write("\n\nORIGINAL ENGLISH:\n")
            f.write(original_english)
            f.write("\n\nENHANCED KHMER:\n")
            f.write(enhanced_khmer)
            f.write("\n\nENHANCED ENGLISH:\n")
            f.write(enhanced_english)
        
        print(f"\n📋 Raw OCR results saved: simple_extreme_ocr_results.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_psm_modes():
    """Test different PSM modes with extreme enhancement."""
    print("\n🎯 Testing Different PSM Modes")
    print("=" * 50)
    
    try:
        from extreme_enhancement import get_best_enhanced_image
        from robust_ocr_parser import parse_ocr_robust
        
        # Load and enhance image
        image = Image.open("id_card.jpg")
        enhanced_image = get_best_enhanced_image(image)
        
        # Test different PSM modes
        psm_modes = [6, 7, 8, 11, 12, 13]
        best_result = None
        best_fields = 0
        
        for psm in psm_modes:
            print(f"\n🔧 Testing PSM {psm}...")
            config = f'--oem 1 --psm {psm} -c preserve_interword_spaces=1 --dpi 300'
            
            try:
                khmer_text = pytesseract.image_to_string(enhanced_image, lang='khm', config=config)
                english_text = pytesseract.image_to_string(enhanced_image, lang='eng', config=config)
                
                # Parse with robust parser
                parsed_data = parse_ocr_robust(khmer_text, english_text)
                
                # Count extracted fields
                extracted_fields = sum(1 for field in ['name', 'id_number', 'dob', 'gender'] 
                                     if parsed_data.get(field))
                
                print(f"   📊 Extracted {extracted_fields}/4 fields")
                if parsed_data.get('name'):
                    print(f"   👤 Name: {parsed_data['name']}")
                
                if extracted_fields > best_fields:
                    best_fields = extracted_fields
                    best_result = (psm, parsed_data)
                
            except Exception as e:
                print(f"   ❌ PSM {psm} failed: {e}")
        
        if best_result:
            psm, data = best_result
            print(f"\n🏆 Best PSM mode: {psm} ({best_fields}/4 fields)")
            print("   Best results:")
            for field, value in data.items():
                if value:
                    print(f"     ✅ {field}: {value}")
        
        return best_result
        
    except Exception as e:
        print(f"❌ PSM testing failed: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Simple Extreme Enhancement Test")
    print("=" * 60)
    
    # Test basic extreme enhancement
    success = test_simple_extreme()
    
    if success:
        # Test different PSM modes
        best_psm = test_different_psm_modes()
        
        print("\n✨ Testing Complete!")
        
        if best_psm:
            psm, data = best_psm
            print(f"\n💡 Recommendation: Use PSM {psm} for best results")
            print("   API usage:")
            print(f"   curl -X POST 'http://localhost:8000/ocr/idcard' \\")
            print(f"        -F 'file=@id_card.jpg' \\")
            print(f"        -F 'extreme_enhancement=true' \\")
            print(f"        -F 'robust_parsing=true'")
        else:
            print("\n⚠️  No successful extractions found.")
            print("   Try adjusting image quality or OCR parameters.")

if __name__ == "__main__":
    main()
