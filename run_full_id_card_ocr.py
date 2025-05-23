"""
End-to-end script for Cambodian ID card OCR extraction with advanced preprocessing, debug visualization, and smart field extraction.
"""
from tesseract_ocr_utils import preprocess_image, extract_text_with_tesseract_simple
from controllers.ocr_controller import parse_cambodian_id_ocr
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_id_card_ocr.py <id_card_image.jpg> [--debug]")
        sys.exit(1)
    image_path = sys.argv[1]
    debug = '--debug' in sys.argv

    # Step 1: Preprocess image (with optional debug visualization)
    print("[INFO] Preprocessing image...")
    pre_img = preprocess_image(image_path, save_debug=debug)
    if debug:
        debug_path = image_path.rsplit('.', 1)[0] + '_preprocessed.png'
        print(f"[INFO] Debug image saved: {debug_path}")

    # Step 2: OCR extraction (Khmer + English)
    print("[INFO] Running Tesseract OCR...")
    khmer_ocr = extract_text_with_tesseract_simple(image_path, lang='khm', config='--oem 1 --psm 11 -c preserve_interword_spaces=1 --dpi 300')
    english_ocr = extract_text_with_tesseract_simple(image_path, lang='eng', config='--oem 1 --psm 11 --dpi 300')

    print("\n[INFO] Khmer OCR Output:\n", khmer_ocr)
    print("\n[INFO] English OCR Output:\n", english_ocr)

    # Step 3: Smart field extraction
    print("[INFO] Extracting ID fields...")
    result = parse_cambodian_id_ocr(khmer_ocr, english_ocr)

    print("\n--- Cambodian ID Card OCR Result ---")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n[INFO] Done.")
