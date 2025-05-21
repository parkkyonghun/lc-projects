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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        khmer_text = pytesseract.image_to_string(processed_image, lang='khm', config='--oem 1 --psm 6')
        english_text = pytesseract.image_to_string(processed_image, lang='eng', config='--oem 1 --psm 6')

        # Parse Khmer and English text
        parsed_data = parse_cambodian_id_ocr(khmer_text, english_text)

        # Return structured result
        return CambodianIDCardOCRResult(
            full_name=parsed_data.get("name"),
            id_number=parsed_data.get("id_number"),
            date_of_birth=parsed_data.get("dob"),
            nationality=parsed_data.get("nationality", "Cambodian"),
            gender=parsed_data.get("gender"),
            raw_khmer=khmer_text,
            raw_english=english_text
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
    image = image.convert("L")
    
    # Adaptive thresholding for better binarization
    try:
        import numpy as np
        arr = np.array(image)
        from PIL import ImageOps
        threshold = arr.mean() * 0.9
        binarized = Image.fromarray((arr > threshold).astype('uint8') * 255)
        image = binarized
    except ImportError:
        # Fallback to simple threshold if numpy not available
        image = image.point(lambda p: p > 128 and 255)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    
    # Optionally set DPI metadata for downstream tools
    image.info['dpi'] = (300, 300)
    
    return image


def parse_cambodian_id_ocr(khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
    """Extract structured fields from OCR text using regex, with Khmer/English fallback and logging."""
    data = {
        "name": None,
        "id_number": None,
        "dob": None,
        "nationality": "Cambodian",
        "gender": None
    }
    # Logging raw OCR for debugging
    logger.info(f"Raw Khmer OCR: {khmer_text}")
    logger.info(f"Raw English OCR: {english_text}")

    # Khmer name extraction (Unicode range)
    name_match = re.search(r"(?:ឈ្មោះ|Name)[^\S\r\n:]*([\u1780-\u17FF\s]+)", khmer_text)
    if name_match:
        data["name"] = name_match.group(1).strip()
        logger.info(f"Extracted Khmer name: {data['name']}")
    else:
        # Fallback: try English
        name_match_en = re.search(r"(?:Name)[^\S\r\n:]*([A-Za-z\s]+)", english_text)
        if name_match_en:
            data["name"] = name_match_en.group(1).strip()
            logger.info(f"Fallback English name: {data['name']}")
        else:
            logger.warning("Name not found in OCR output.")

    # ID Number: Khmer first, fallback to English
    id_match_kh = re.search(r"(?:លេខសម្គាល់|ID)[^\d]*(\d{12})", khmer_text)
    id_match_en = re.search(r"\b\d{12}\b", english_text)
    if id_match_kh:
        data["id_number"] = id_match_kh.group(1)
        logger.info(f"Extracted Khmer ID: {data['id_number']}")
    elif id_match_en:
        data["id_number"] = id_match_en.group(0)
        logger.info(f"Fallback English ID: {data['id_number']}")
    else:
        logger.warning("ID number not found in OCR output.")

    # Date of Birth: Khmer first, fallback to English
    dob_match_kh = re.search(r"(?:ថ្ងៃកំណើត|DOB)[^\d]*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})", khmer_text)
    dob_match_en = re.search(r"(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})", english_text)
    if dob_match_kh:
        data["dob"] = dob_match_kh.group(1)
        logger.info(f"Extracted Khmer DOB: {data['dob']}")
    elif dob_match_en:
        data["dob"] = dob_match_en.group(1)
        logger.info(f"Fallback English DOB: {data['dob']}")
    else:
        logger.warning("DOB not found in OCR output.")

    # Gender: Khmer first, fallback to English
    gender_match_kh = re.search(r"(?:ភេទ|Sex)[^\S\r\n:]*([\u1797\u179C]+)", khmer_text)
    gender_match_en = re.search(r"(?:Sex)[^\S\r\n:]*([MF])", english_text, re.IGNORECASE)
    if gender_match_kh:
        gender_kh = gender_match_kh.group(1)
        if gender_kh == '\u1797' or 'ប្រុស' in gender_kh:
            data["gender"] = "Male"
        elif gender_kh == '\u179C' or 'ស្រី' in gender_kh:
            data["gender"] = "Female"
        logger.info(f"Extracted Khmer gender: {data['gender']}")
    elif gender_match_en:
        gender_en = gender_match_en.group(1).upper()
        data["gender"] = "Male" if gender_en == "M" else "Female"
        logger.info(f"Fallback English gender: {data['gender']}")
    else:
        logger.warning("Gender not found in OCR output.")

    return data