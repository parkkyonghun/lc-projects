from schemas.ocr import CambodianIDCardOCRResult
from fastapi import UploadFile, HTTPException
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import re
import logging
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KhmerIDOCR")

# Path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

async def process_cambodian_id_ocr(file: UploadFile) -> CambodianIDCardOCRResult:
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image for better OCR accuracy
        processed_image = preprocess_image(image)

        # Run Tesseract OCR with language-specific optimizations
        # Using PSM 11 (Sparse text) as it's generally better for ID cards with scattered text fields.
        # OEM 1 (LSTM engine) is already correctly used.
        khmer_text = pytesseract.image_to_string(processed_image, lang='khm', config='--oem 1 --psm 11')
        english_text = pytesseract.image_to_string(processed_image, lang='eng', config='--oem 1 --psm 11')

        # Parse Khmer and English text
        parsed_data = parse_cambodian_id_ocr(khmer_text, english_text)

        # Return structured result
        return CambodianIDCardOCRResult(
            full_name=parsed_data.get("name"), # Primary, prioritized name
            name_kh=parsed_data.get("name_kh"), # Specifically from Khmer OCR
            name_en=parsed_data.get("name_en"), # Specifically from English OCR (fallback)
            id_number=parsed_data.get("id_number"),
            date_of_birth=parsed_data.get("dob"), # Schema field renamed
            gender=parsed_data.get("gender"), # Schema field renamed
            nationality=parsed_data.get("nationality"), # Schema field added
            raw_khmer=khmer_text,
            raw_english=english_text
            # Other fields like height, address, etc., are not populated by current parsing
            # and will remain None as per schema defaults.
        )

    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
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


def parse_cambodian_id_ocr(khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
    """Extract structured fields from OCR text using regex, with Khmer/English fallback and logging."""
    data = {
        "name": None,
        "name_kh": None, # To store specifically Khmer-extracted name
        "name_en": None, # To store specifically English-extracted name
        "id_number": None,
        "dob": None,
        "nationality": "Cambodian",
        "gender": None
    }
    # Logging raw OCR for debugging
    logger.info(f"Raw Khmer OCR: {khmer_text}")
    logger.info(f"Raw English OCR: {english_text}")

    # Khmer name extraction (Unicode range) - Prioritize Khmer
    name_match_kh = re.search(r"ឈ្មោះ\s*[:\s]*\s*([\u1780-\u17FF ]+)", khmer_text) # Label: ឈ្មោះ, capture Khmer chars and spaces only (no newlines)
    if name_match_kh:
        extracted_name_kh = name_match_kh.group(1).strip()
        data["name"] = extracted_name_kh
        data["name_kh"] = extracted_name_kh
        logger.info(f"Extracted Khmer name: {data['name']}")
    else:
        # Fallback: try English label in Khmer text if Khmer label name not found
        name_match_kh_alt = re.search(r"Name\s*[:\s]*\s*([\u1780-\u17FF ]+)", khmer_text) # Capture Khmer chars and spaces
        if name_match_kh_alt:
             extracted_name_kh_alt = name_match_kh_alt.group(1).strip()
             data["name"] = extracted_name_kh_alt
             data["name_kh"] = extracted_name_kh_alt
             logger.info(f"Extracted Khmer name (using 'Name' label): {data['name']}")
        else:
            # Fallback: try English name from English text
            name_match_en = re.search(r"Name\s*[:\s]*\s*([A-Za-z ]+)", english_text) # Capture English chars and spaces
            if name_match_en:
                extracted_name_en = name_match_en.group(1).strip()
                data["name"] = extracted_name_en
                data["name_en"] = extracted_name_en
                logger.info(f"Fallback English name: {data['name']}")
            else:
                logger.warning("Name not found in OCR output.")

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