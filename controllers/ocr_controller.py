from schemas.ocr import CambodianIDCardOCRResult
from fastapi import UploadFile, HTTPException
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import re
import cv2
import numpy as np
import os
import logging
import time
from typing import Optional, Dict, List, Tuple, Union

# Import enhanced OCR utilities
from controllers.enhanced_ocr_utils import (
    enhance_for_ocr,
    find_best_ocr_variant,
    extract_text_from_variants,
    extract_id_card_info,
    extract_mrz_info,
    khmer_digits_to_latin,
    get_optimized_variants
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KhmerIDOCR")

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# OCR configs
OCR_CONFIG_KHMER = '--oem 1 --psm 11 -c preserve_interword_spaces=1 --dpi 300'
OCR_CONFIG_ENGLISH = '--oem 1 --psm 11 --dpi 300'

async def process_cambodian_id_ocr(file: UploadFile) -> CambodianIDCardOCRResult:
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

        # Read image bytes
        image_bytes = await file.read()
        
        # Save the uploaded image temporarily to use with enhanced OCR functions
        temp_path = os.path.join(os.getcwd(), "temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # Use our enhanced processing pipeline
        logger.info(f"Processing ID card image with enhanced OCR techniques")
        
        # Generate optimized variants for better OCR quality and performance
        variants = get_optimized_variants(temp_path)
        logger.info(f"Generated {len(variants)} optimized variants for OCR processing")
        
        # Find the best variant for OCR processing
        best_image, best_variant_name = find_best_ocr_variant(variants[:10], temp_path)  # Limit to top 10 variants for speed
        logger.info(f"Selected best OCR variant: {best_variant_name}")
        
        # Extract text using multiple approaches for better results
        # Use only top variants for text extraction to improve performance
        combined_ocr_text = extract_text_from_variants(variants[:5], temp_path)
        
        # Run Tesseract OCR on the best image
        khmer_text = pytesseract.image_to_string(best_image, lang='khm', config=OCR_CONFIG_KHMER)
        english_text = pytesseract.image_to_string(best_image, lang='eng', config=OCR_CONFIG_ENGLISH)
        
        # Try with one additional PSM mode for improved results (reduced from 2 to 1 for performance)
        # PSM 3 is better for full-page layouts like ID cards
        config = f'--oem 1 --psm 3 -c preserve_interword_spaces=1 --dpi 300'
        khmer_text += "\n" + pytesseract.image_to_string(best_image, lang='khm', config=config)
        english_text += "\n" + pytesseract.image_to_string(best_image, lang='eng', config=config)

        # Add combined OCR text to help with pattern matching
        all_text = f"{khmer_text}\n{english_text}\n{combined_ocr_text}"
        
        # Parse using enhanced extraction techniques
        parsed_data = extract_id_card_info(all_text, khmer_text, english_text)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"OCR processing completed in {processing_time:.2f} seconds")
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        # Determine best name to use for full_name field (prefer English for consistency)
        full_name = parsed_data.get("name_en")
        if not full_name:
            full_name = parsed_data.get("name")
        if not full_name:
            full_name = parsed_data.get("name_kh")
            
        logger.info(f"Using {full_name} as full_name (prioritizing English name)")
        
        # Return structured result with all available fields
        return CambodianIDCardOCRResult(
            full_name=full_name,  # Prioritize English name
            name_kh=parsed_data.get("name_kh"),  # Specifically from Khmer OCR
            name_en=parsed_data.get("name_en"),  # Specifically from English OCR
            id_number=parsed_data.get("id_number"),
            date_of_birth=parsed_data.get("dob"),  # Schema field renamed
            gender=parsed_data.get("gender"),  # Schema field renamed
            nationality=parsed_data.get("nationality"),  # Schema field added
            height=parsed_data.get("height"),
            birth_place=parsed_data.get("birth_place"),
            address=parsed_data.get("address"),
            issue_date=parsed_data.get("issue_date"),
            expiry_date=parsed_data.get("expiry_date"),
            description=parsed_data.get("description"),
            raw_khmer=khmer_text,
            raw_english=english_text
        )

    except Exception as e:
        logger.error(f"Enhanced OCR processing failed: {str(e)}")
        # Fallback to original method if enhanced method fails
        try:
            # Restore image from bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Use traditional processing as fallback
            processed_image = preprocess_image(image)
            
            # Standard OCR approach
            khmer_text = pytesseract.image_to_string(processed_image, lang='khm', config=OCR_CONFIG_KHMER)
            english_text = pytesseract.image_to_string(processed_image, lang='eng', config=OCR_CONFIG_ENGLISH)
            
            # Standard parsing
            parsed_data = parse_cambodian_id_ocr(khmer_text, english_text)
            
            logger.info("Successfully processed with fallback method")
            
            return CambodianIDCardOCRResult(
                id_number=parsed_data.get("id_number"),
                name=parsed_data.get("name"),
                name_kh=parsed_data.get("name_kh"),
                name_en=parsed_data.get("name_en"),
                dob=parsed_data.get("dob"),
                gender=parsed_data.get("gender"),
                nationality=parsed_data.get("nationality"),
                height=parsed_data.get("height"),
                birth_place=parsed_data.get("birth_place"),
                address=parsed_data.get("address"),
                issue_date=parsed_data.get("issue_date"),
                expiry_date=parsed_data.get("expiry_date"),
                raw_khmer=khmer_text,
                raw_english=english_text
            )
        except Exception as fallback_error:
            logger.error(f"Fallback OCR processing also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}\nFallback also failed: {str(fallback_error)}")
        
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image quality for OCR with DPI, grayscale, adaptive threshold, contrast, and noise reduction."""
    # Resize to 300 DPI equivalent (Tesseract expects 300 DPI for best results)
    # If image has DPI info, use it; otherwise, assume 72 DPI and scale accordingly
    dpi = image.info.get('dpi', (72, 72))[0]
    scale_factor = 300 / dpi if dpi != 300 else 1
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    if scale_factor != 1:
        image = image.resize(new_size, resample=Image.BICUBIC)
    
    # Convert to grayscale
    image_l = image.convert("L") # Work on a grayscale copy for cv2 processing
    
    # Convert PIL Image to OpenCV format
    import numpy as np
    import cv2 # Ensure cv2 is imported

    img_cv = np.array(image_l)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cv = clahe.apply(img_cv)
    
    # Apply bilateral filter to reduce noise while preserving edges
    img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75) # Parameters (d, sigmaColor, sigmaSpace)
    
    # Apply adaptive thresholding (Gaussian)
    # Block size and C value might need tuning for Khmer script.
    # Using parameters similar to tesseract_ocr_utils for consistency, but these can be fine-tuned.
    img_cv = cv2.adaptiveThreshold(
        img_cv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 7 
    )

    # Apply morphological opening to remove small noise (optional, but can help)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel, iterations=1)

    # Convert OpenCV image back to PIL Image
    image = Image.fromarray(img_cv)
    
    # Optionally set DPI metadata for downstream tools (original image's DPI info was used for scaling)
    # The image is now processed, its effective DPI related to content size vs pixel dimensions.
    # Setting it to 300 can be a hint for Tesseract if it doesn't pick up from image properties.
    image.info['dpi'] = (300, 300)
    
    return image


def khmer_digits_to_latin(text: str) -> str:
    """Convert Khmer digits to Latin digits in a string."""
    khmer_digit_map = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
    return text.translate(khmer_digit_map)


def parse_cambodian_id_ocr(khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
    """Extract structured fields from OCR text using regex, with Khmer/English fallback and logging."""
    data = {
        "name": None,
        "name_kh": None,
        "name_en": None,
        "id_number": None,
        "dob": None,
        "nationality": "Cambodian",
        "gender": None,
        "issue_date": None,
        "expiry_date": None,
        "address": None,
        "birth_place": None,
        "height": None
    }
    # Normalize Khmer digits to Latin for easier regex extraction
    khmer_text_norm = khmer_digits_to_latin(khmer_text)
    english_text_norm = khmer_digits_to_latin(english_text)

    logger.info(f"Raw Khmer OCR: {khmer_text}")
    logger.info(f"Raw English OCR: {english_text}")

    # --- Name extraction ---
    # Try multiple Khmer labels for name
    name_patterns = [
        r"ឈ្មោះ\s*[:\s]*\s*([\u1780-\u17FF ]+)",
        r"គោត្តនាមនិងនាម\s*[:\s]*\s*([\u1780-\u17FF ]+)",
        r"ឈ្មោះ\s*[:\s]*\s*([A-Za-z ]+)",  # Sometimes English chars under Khmer label
    ]
    for pat in name_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["name"] = m.group(1).strip()
            # Assign to kh/en based on script
            if re.search(r"[\u1780-\u17FF]", data["name"]):
                data["name_kh"] = data["name"]
            else:
                data["name_en"] = data["name"]
            logger.info(f"Extracted name: {data['name']}")
            break
    # Fallback: English text
    if not data["name"]:
        m = re.search(r"Name\s*[:\s]*\s*([A-Za-z ]+)", english_text_norm)
        if m:
            data["name"] = m.group(1).strip()
            data["name_en"] = data["name"]
            logger.info(f"Fallback English name: {data['name']}")
    # Fallback: first non-empty line (if nothing found)
    if not data["name"]:
        lines = [l.strip() for l in khmer_text_norm.splitlines() if l.strip()]
        if lines:
            data["name"] = lines[0]
            logger.info(f"Fallback first line as name: {data['name']}")

    # --- ID number extraction ---
    id_patterns = [
        r"(?:លេខសម្គាល់|ID(?: Number)?|លេខអត្តសញ្ញាណ)\s*[:\s]*([\d ]{6,})",
        r"([\d]{8,})"
    ]
    for pat in id_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["id_number"] = m.group(1).replace(" ", "").strip()
            logger.info(f"Extracted Khmer ID: {data['id_number']}")
            break
    if not data["id_number"]:
        m = re.search(r"ID(?: Number)?\s*[:\s]*([\d ]{6,})", english_text_norm)
        if m:
            data["id_number"] = m.group(1).replace(" ", "").strip()
            logger.info(f"Fallback English ID: {data['id_number']}")

    # --- DOB extraction ---
    dob_patterns = [
        r"(?:ថ្ងៃកំណើត|ថ្ងៃខែឆ្នាំកំណើត|DOB|Date of Birth)\s*[:\s]*([\d./-]{6,})",
        r"([\d]{1,2}[./-][\d]{1,2}[./-][\d]{2,4})"
    ]
    for pat in dob_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["dob"] = m.group(1).strip()
            logger.info(f"Extracted DOB: {data['dob']}")
            break
    if not data["dob"]:
        m = re.search(r"(?:Date of Birth|DOB)\s*[:\s]*([\d./-]{6,})", english_text_norm, re.IGNORECASE)
        if m:
            data["dob"] = m.group(1).strip()
            logger.info(f"Fallback English DOB: {data['dob']}")

    # --- Gender extraction ---
    gender_patterns = [
        r"(?:ភេទ|Sex)\s*[:\s]*\s*(ប្រុស|ស្រី|Male|Female|M|F)",
    ]
    for pat in gender_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            val = m.group(1).strip()
            if val in ["ប្រុស", "Male", "M"]:
                data["gender"] = "Male"
            elif val in ["ស្រី", "Female", "F"]:
                data["gender"] = "Female"
            logger.info(f"Extracted gender: {data['gender']}")
            break
    if not data["gender"]:
        m = re.search(r"Sex\s*[:\s]*\s*(Male|Female|M|F)", english_text_norm, re.IGNORECASE)
        if m:
            val = m.group(1).strip().upper()
            data["gender"] = "Male" if val in ["M", "MALE"] else "Female"
            logger.info(f"Fallback English gender: {data['gender']}")

    # --- Nationality extraction ---
    nat_patterns = [
        r"សញ្ជាតិ\s*[:\s]*\s*([\u1780-\u17FF ]+)",
        r"Nationality\s*[:\s]*\s*([\u1780-\u17FF ]+)",
        r"Nationality\s*[:\s]*\s*([A-Za-z ]+)",
    ]
    for pat in nat_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["nationality"] = m.group(1).strip()
            logger.info(f"Extracted nationality: {data['nationality']}")
            break
        m = re.search(pat, english_text_norm)
        if m:
            data["nationality"] = m.group(1).strip()
            logger.info(f"Extracted nationality: {data['nationality']}")
            break

    # --- Issue/expiry date extraction ---
    # Look for: សុពលភាព: ០៤.០៤.២០១៥ ដល់ថ្ងៃ ០៣.០៨.២០២៥
    validity_match = re.search(r"សុពលភាព\s*[:\s]*([\d./-]{6,})\s*ដល់ថ្ងៃ\s*([\d./-]{6,})", khmer_text_norm)
    if validity_match:
        data["issue_date"] = validity_match.group(1).strip()
        data["expiry_date"] = validity_match.group(2).strip()
        logger.info(f"Extracted issue/expiry: {data['issue_date']} / {data['expiry_date']}")

    # --- Address extraction ---
    # Expanded Khmer labels for address to handle common OCR errors/misreads
    addr_patterns = [
        r"អាសយដ្ឋាន\s*[:\s]*([\u1780-\u17FF\d\s,.-]+)",         # Correct label
        r"អាសយដ្ឋានៈ\s*([\u1780-\u17FF\d\s,.-]+)",               # OCR variant
        r"អាសយផ្លានៈ\s*([\u1780-\u17FF\d\s,.-]+)",               # OCR variant
        r"Address\s*[:\s]*([A-Za-z\d\s,.-]+)"                     # English
    ]
    for pat in addr_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["address"] = m.group(1).strip()
            logger.info(f"Extracted address: {data['address']}")
            break
        m = re.search(pat, english_text_norm)
        if m:
            data["address"] = m.group(1).strip()
            logger.info(f"Extracted address: {data['address']}")
            break
    # Fallback: look for lines containing 'ភូមិ', 'សង្កាត់', 'ខណ្ឌ', or 'address' (common in Cambodian addresses)
    if not data["address"]:
        for line in khmer_text_norm.splitlines():
            if any(lbl in line for lbl in ["ភូមិ", "សង្កាត់", "ខណ្ឌ"]):
                data["address"] = line.strip()
                logger.info(f"Fallback address line: {data['address']}")
                break

    # --- Birth place extraction ---
    # Expanded Khmer labels for birth place to handle OCR errors/misreads
    bp_patterns = [
        r"ទីកន្លែងកំណើត\s*[:\s]*([\u1780-\u17FF\d\s,.-]+)",     # Correct label
        r"ទីកន្លែងកំណើតៈ\s*([\u1780-\u17FF\d\s,.-]+)",         # OCR variant
        r"ទីកន្លែងកំពេរីត\s*[:\s]*([\u1780-\u17FF\d\s,.-]+)",  # OCR variant
        r"Birth Place\s*[:\s]*([A-Za-z\d\s,.-]+)"                 # English
    ]
    for pat in bp_patterns:
        m = re.search(pat, khmer_text_norm)
        if m:
            data["birth_place"] = m.group(1).strip()
            logger.info(f"Extracted birth place: {data['birth_place']}")
            break
        m = re.search(pat, english_text_norm)
        if m:
            data["birth_place"] = m.group(1).strip()
            logger.info(f"Extracted birth place: {data['birth_place']}")
            break
    # Fallback: look for lines containing 'កំណើត' (birth)
    if not data["birth_place"]:
        for line in khmer_text_norm.splitlines():
            if "កំណើត" in line:
                data["birth_place"] = line.strip()
                logger.info(f"Fallback birth place line: {data['birth_place']}")
                break

    # --- Gender extraction improvement ---
    # Also try to extract gender from lines with both 'ស្រី' or 'ប្រុស' (female/male) even without label
    if not data["gender"]:
        for line in khmer_text_norm.splitlines():
            if "ប្រុស" in line:
                data["gender"] = "Male"
                logger.info(f"Fallback gender from line: {data['gender']}")
                break
            if "ស្រី" in line:
                data["gender"] = "Female"
                logger.info(f"Fallback gender from line: {data['gender']}")
                break

    return data

    # ID Number: Khmer first, fallback to English
    id_match_kh = re.search(r"លេខសម្គាល់\s*[:\s]*\s*(\d+[\s\d]*)", khmer_text) # Label: លេខសម្គាល់, allow spaces in numbers
    if id_match_kh:
        data["id_number"] = re.sub(r'\s+', '', id_match_kh.group(1).strip()) # Remove spaces
        logger.info(f"Extracted Khmer ID: {data['id_number']}")
    else:
        id_match_kh_alt = re.search(r"ID\s*[:\s]*\s*(\d+[\s\d]*)", khmer_text) # English "ID" label in Khmer text
        if id_match_kh_alt:
            data["id_number"] = re.sub(r'\s+', '', id_match_kh_alt.group(1).strip())
            logger.info(f"Extracted Khmer ID (using 'ID' label): {data['id_number']}")
        else:
            id_match_en = re.search(r"ID\sNumber\s*[:\s]*\s*(\d+[\s\d]*)", english_text, re.IGNORECASE) # Label: ID Number
            if not id_match_en: # Try simple ID fallback if "ID Number" not found
                id_match_en = re.search(r"\b(\d{9,12})\b", english_text) # Common ID lengths
            if id_match_en:
                data["id_number"] = re.sub(r'\s+', '', id_match_en.group(1).strip())
                logger.info(f"Fallback English ID: {data['id_number']}")
            else:
                logger.warning("ID number not found in OCR output.")


    # Date of Birth: Khmer first, fallback to English
    dob_match_kh = re.search(r"ថ្ងៃកំណើត\s*[:\s]*\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", khmer_text) # Label: ថ្ងៃកំណើត
    if dob_match_kh:
        data["dob"] = dob_match_kh.group(1).strip()
        logger.info(f"Extracted Khmer DOB: {data['dob']}")
    else:
        dob_match_kh_alt = re.search(r"DOB\s*[:\s]*\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", khmer_text) # English "DOB" label in Khmer text
        if dob_match_kh_alt:
            data["dob"] = dob_match_kh_alt.group(1).strip()
            logger.info(f"Extracted Khmer DOB (using 'DOB' label): {data['dob']}")
        else:
            # Fallback to English, prefer "Date of Birth" or "DOB" label
            dob_match_en = re.search(r"(?:Date\s*of\s*Birth|DOB)\s*[:\s]*\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", english_text, re.IGNORECASE)
            if not dob_match_en: # Fallback to any date pattern if specific labels not found
                dob_match_en = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", english_text)
            if dob_match_en:
                data["dob"] = dob_match_en.group(1).strip()
                logger.info(f"Fallback English DOB: {data['dob']}")
            else:
                logger.warning("DOB not found in OCR output.")

    # Gender: Khmer first (Corrected), fallback to English
    gender_match_kh = re.search(r"ភេទ\s*[:\s]*\s*(ប្រុស|ស្រី)", khmer_text) # Label: ភេទ, Value: ប្រុស or ស្រី
    if gender_match_kh:
        gender_kh_value = gender_match_kh.group(1).strip()
        if gender_kh_value == "ប្រុស": # Male
            data["gender"] = "Male"
        elif gender_kh_value == "ស្រី": # Female
            data["gender"] = "Female"
        logger.info(f"Extracted Khmer gender: {data['gender']}")
    else:
        gender_match_kh_alt = re.search(r"Sex\s*[:\s]*\s*(ប្រុស|ស្រី)", khmer_text) # English "Sex" label in Khmer text
        if gender_match_kh_alt:
            gender_kh_value = gender_match_kh_alt.group(1).strip()
            if gender_kh_value == "ប្រុស":
                data["gender"] = "Male"
            elif gender_kh_value == "ស្រី":
                data["gender"] = "Female"
            logger.info(f"Extracted Khmer gender (using 'Sex' label): {data['gender']}")
        else:
            gender_match_en = re.search(r"Sex\s*[:\s]*\s*([MF]|Male|Female)", english_text, re.IGNORECASE)
            if gender_match_en:
                gender_en_val = gender_match_en.group(1).strip().upper()
                if gender_en_val == "M" or gender_en_val == "MALE":
                    data["gender"] = "Male"
                elif gender_en_val == "F" or gender_en_val == "FEMALE":
                    data["gender"] = "Female"
                logger.info(f"Fallback English gender: {data['gender']}")
            else:
                logger.warning("Gender not found in OCR output.")
    
    # Nationality: Khmer first, fallback to English. Default to "Cambodian".
    nationality_match_kh = re.search(r"សញ្ជាតិ\s*[:\s]*\s*([\u1780-\u17FF ]+)", khmer_text) # Label: សញ្ជាតិ, capture Khmer chars and spaces
    if nationality_match_kh:
        data["nationality"] = nationality_match_kh.group(1).strip()
        logger.info(f"Extracted Khmer nationality: {data['nationality']}")
    else:
        nationality_match_kh_alt = re.search(r"Nationality\s*[:\s]*\s*([\u1780-\u17FF ]+)", khmer_text) # English "Nationality" label in Khmer text, capture Khmer chars and spaces
        if nationality_match_kh_alt:
            data["nationality"] = nationality_match_kh_alt.group(1).strip()
            logger.info(f"Extracted Khmer nationality (using 'Nationality' label): {data['nationality']}")
        else:
            nationality_match_en = re.search(r"Nationality\s*[:\s]*\s*([A-Za-z ]+)", english_text, re.IGNORECASE) # Capture English chars and spaces
            if nationality_match_en:
                data["nationality"] = nationality_match_en.group(1).strip()
                logger.info(f"Fallback English nationality: {data['nationality']}")
            else:
                # If no nationality found, keep the default "Cambodian"
                logger.info("Nationality not found in OCR output, using default 'Cambodian'.")

    return data