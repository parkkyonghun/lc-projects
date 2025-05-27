from schemas.ocr import CambodianIDCardOCRResult
from fastapi import UploadFile, HTTPException
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import re
import logging
import numpy as np
from typing import Optional, Dict

# Import our enhanced preprocessing utilities
from image_enhancement_utils import create_preprocessing_pipeline
from enhancement_config import get_config_by_name, KHMER_ID_CARD_CONFIG

# Import AI-powered enhancement
from ai_image_enhancement import create_ai_enhancer, assess_enhancement_potential
from ai_enhancement_config import get_config_by_name as get_ai_config, auto_select_config

# Import extreme enhancement and robust parsing
from extreme_enhancement import enhance_with_multiple_approaches, get_best_enhanced_image
from robust_ocr_parser import parse_ocr_robust

# Import Khmer language processing
from khmer_text_processor import create_khmer_processor
from khmer_language_integration import create_khmer_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KhmerIDOCR")

# Path to Tesseract executable (Windows only)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

async def get_active_model_info() -> Optional[Dict]:
    """
    Get information about the currently active deployed model.

    Returns:
        Dictionary with active model info or None if no active model
    """
    try:
        import json
        import os

        active_model_path = 'model_registry/active_model.json'
        if os.path.exists(active_model_path):
            with open(active_model_path, 'r') as f:
                active_model_info = json.load(f)

            # Add mock accuracy for demonstration (in production, this would come from model metadata)
            if 'accuracy' not in active_model_info:
                active_model_info['accuracy'] = 0.95  # Assume high accuracy for deployed models

            logger.info(f"Found active model: {active_model_info['model_id']}")
            return active_model_info
        else:
            logger.info("No active deployed model found, using default OCR settings")
            return None

    except Exception as e:
        logger.error(f"Failed to get active model info: {e}")
        return None

async def process_cambodian_id_ocr(
    file: UploadFile,
    use_enhanced_preprocessing: bool = True,
    use_ai_enhancement: bool = False,
    use_extreme_enhancement: bool = False,
    enhancement_mode: str = "auto",
    use_robust_parsing: bool = True
) -> CambodianIDCardOCRResult:
    """
    Process Cambodian ID card OCR with advanced AI-powered enhancement options.

    Args:
        file: Uploaded image file
        use_enhanced_preprocessing: Whether to use the enhanced preprocessing pipeline
        use_ai_enhancement: Whether to use AI-powered enhancement for ultra-low quality images
        use_extreme_enhancement: Whether to use extreme enhancement for severely damaged images
        enhancement_mode: Enhancement mode ('auto', 'ultra_low_quality', 'low_quality', 'khmer_optimized', 'high_performance')
        use_robust_parsing: Whether to use robust parsing for poor quality OCR text

    Returns:
        CambodianIDCardOCRResult with extracted data
    """
    try:
        # Check for active deployed model and use enhanced settings if available
        active_model_info = await get_active_model_info()
        if active_model_info:
            logger.info(f"Using active deployed model: {active_model_info['model_id']}")
            # Enhance processing parameters based on deployed model capabilities
            use_enhanced_preprocessing = True
            use_robust_parsing = True
            # If the deployed model has high accuracy, we can use more aggressive enhancement
            if active_model_info.get('accuracy', 0) > 0.9:
                use_ai_enhancement = True
                enhancement_mode = "khmer_optimized"

        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")

        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        logger.info(f"Processing image: {image.size}, mode: {image.mode}")

        # Choose preprocessing method based on parameters (priority order)
        if use_extreme_enhancement:
            # Use extreme enhancement for severely damaged images
            processed_image = await _apply_extreme_enhancement(image)
            logger.info("Applied extreme enhancement for severely damaged image")
        elif use_ai_enhancement:
            # Use AI-powered enhancement for ultra-low quality images
            processed_image = await _apply_ai_enhancement(image, enhancement_mode)
            logger.info(f"Applied AI enhancement with mode: {enhancement_mode}")
        elif use_enhanced_preprocessing:
            # Use enhanced preprocessing pipeline
            enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)
            processed_image = enhancer.enhance_for_ocr(image)
            logger.info("Applied enhanced preprocessing pipeline")
        else:
            # Use legacy preprocessing
            processed_image = preprocess_image(image, debug=False)
            logger.info("Applied legacy preprocessing")

        # Optimized Tesseract configuration for Khmer ID cards
        # PSM 11: Sparse text (good for ID cards with scattered text fields)
        # OEM 1: LSTM engine only (best for modern OCR)
        # preserve_interword_spaces: Important for Khmer script
        # DPI setting: Helps Tesseract understand image resolution
        tesseract_config = (
            '--oem 1 --psm 11 '
            '-c preserve_interword_spaces=1 '
            '--dpi 300 '
            '-c load_system_dawg=0 '
            '-c load_freq_dawg=0'
        )

        # Run Tesseract OCR with language-specific optimizations
        khmer_text = pytesseract.image_to_string(processed_image, lang='khm', config=tesseract_config)
        english_text = pytesseract.image_to_string(processed_image, lang='eng', config=tesseract_config)

        logger.info(f"OCR completed - Khmer: {len(khmer_text)} chars, English: {len(english_text)} chars")

        # Apply Khmer text processing and normalization
        khmer_processor = create_khmer_processor()

        # Process Khmer text
        if khmer_text.strip():
            # Normalize Khmer text
            normalized_khmer = khmer_processor.normalize_khmer_text(khmer_text)

            # Validate and correct OCR errors
            validation_result = khmer_processor.validate_khmer_text(normalized_khmer)
            correction_result = khmer_processor.correct_ocr_errors(normalized_khmer)

            # Use corrected text if confidence is high enough
            if correction_result['confidence'] > 0.7:
                khmer_text = correction_result['corrected_text']
                logger.info(f"Applied Khmer text corrections with confidence: {correction_result['confidence']:.3f}")
            else:
                khmer_text = normalized_khmer
                logger.info("Applied Khmer text normalization only")

            # Log validation results
            if validation_result['issues']:
                logger.warning(f"Khmer text validation issues: {validation_result['issues']}")

        logger.info(f"Khmer text processing completed - Final length: {len(khmer_text)} chars")

        # Parse Khmer and English text using appropriate parser
        if use_robust_parsing:
            parsed_data = parse_ocr_robust(khmer_text, english_text)
            logger.info("Used robust OCR parsing")
        else:
            parsed_data = parse_cambodian_id_ocr(khmer_text, english_text)
            logger.info("Used standard OCR parsing")

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


async def _apply_ai_enhancement(image: Image.Image, enhancement_mode: str) -> Image.Image:
    """
    Apply AI-powered enhancement to the image.

    Args:
        image: Input PIL Image
        enhancement_mode: Enhancement mode to use

    Returns:
        Enhanced PIL Image
    """
    try:
        # Assess image quality first
        assessment = assess_enhancement_potential(image)
        logger.info(f"Image quality assessment: score={assessment['quality_score']:.3f}, "
                   f"priority={assessment['enhancement_priority']}")

        # Select configuration based on mode
        if enhancement_mode == "auto":
            # Auto-select based on image characteristics
            config = auto_select_config(
                assessment['quality_score'],
                image.size,
                processing_priority='quality'
            )
            logger.info("Auto-selected AI enhancement configuration")
        else:
            # Use specified configuration
            config = get_ai_config(enhancement_mode)
            logger.info(f"Using AI enhancement configuration: {enhancement_mode}")

        # Create AI enhancer
        ai_enhancer = create_ai_enhancer(use_gpu=config.use_gpu)

        # Apply AI enhancement
        enhanced_image = ai_enhancer.enhance_ultra_low_quality(image)

        logger.info("AI enhancement completed successfully")
        return enhanced_image

    except Exception as e:
        logger.error(f"AI enhancement failed: {str(e)}")
        # Fallback to regular enhanced preprocessing
        logger.info("Falling back to regular enhanced preprocessing")
        enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)
        return enhancer.enhance_for_ocr(image)


async def _apply_extreme_enhancement(image: Image.Image) -> Image.Image:
    """
    Apply extreme enhancement for severely damaged images.

    Args:
        image: Input PIL Image

    Returns:
        Best enhanced PIL Image
    """
    try:
        logger.info("Applying extreme enhancement for severely damaged image")

        # Get the best enhanced version from multiple approaches
        enhanced_image = get_best_enhanced_image(image)

        logger.info("Extreme enhancement completed successfully")
        return enhanced_image

    except Exception as e:
        logger.error(f"Extreme enhancement failed: {str(e)}")
        # Fallback to AI enhancement
        logger.info("Falling back to AI enhancement")
        try:
            return await _apply_ai_enhancement(image, "ultra_low_quality")
        except:
            # Final fallback to regular enhanced preprocessing
            logger.info("Falling back to regular enhanced preprocessing")
            enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)
            return enhancer.enhance_for_ocr(image)


def preprocess_image(image: Image.Image, debug: bool = False) -> Image.Image:
    """
    Enhanced image preprocessing for optimal Tesseract OCR results.

    Args:
        image: Input PIL Image
        debug: If True, saves intermediate processing steps for debugging

    Returns:
        Preprocessed PIL Image optimized for OCR
    """
    import numpy as np
    import cv2
    from typing import Tuple

    # Step 1: Resize to optimal DPI (300 DPI is ideal for Tesseract)
    image = _resize_to_optimal_dpi(image)

    # Step 2: Convert to grayscale
    image_gray = image.convert("L")
    img_cv = np.array(image_gray)

    if debug:
        cv2.imwrite("debug_01_grayscale.png", img_cv)

    # Step 3: Detect and correct skew
    img_cv = _correct_skew(img_cv)

    if debug:
        cv2.imwrite("debug_02_deskewed.png", img_cv)

    # Step 4: Remove shadows and improve lighting
    img_cv = _remove_shadows(img_cv)

    if debug:
        cv2.imwrite("debug_03_shadow_removed.png", img_cv)

    # Step 5: Enhanced contrast and brightness adjustment
    img_cv = _enhance_contrast_adaptive(img_cv)

    if debug:
        cv2.imwrite("debug_04_contrast_enhanced.png", img_cv)

    # Step 6: Advanced noise reduction
    img_cv = _advanced_noise_reduction(img_cv)

    if debug:
        cv2.imwrite("debug_05_denoised.png", img_cv)

    # Step 7: Optimized binarization for text
    img_cv = _optimal_binarization(img_cv)

    if debug:
        cv2.imwrite("debug_06_binarized.png", img_cv)

    # Step 8: Text-specific morphological operations
    img_cv = _text_morphological_operations(img_cv)

    if debug:
        cv2.imwrite("debug_07_morphological.png", img_cv)

    # Step 9: Final quality enhancement
    img_cv = _final_quality_enhancement(img_cv)

    if debug:
        cv2.imwrite("debug_08_final.png", img_cv)

    # Convert back to PIL Image
    result_image = Image.fromarray(img_cv)
    result_image.info['dpi'] = (300, 300)

    return result_image


def _resize_to_optimal_dpi(image: Image.Image) -> Image.Image:
    """Resize image to optimal DPI for OCR (300 DPI)."""
    dpi = image.info.get('dpi', (72, 72))[0]

    # Calculate scale factor for 300 DPI
    target_dpi = 300
    scale_factor = target_dpi / dpi if dpi != target_dpi else 1

    # Ensure minimum size for small images
    min_width, min_height = 800, 600
    current_width, current_height = image.size

    # Calculate new size
    new_width = max(int(current_width * scale_factor), min_width)
    new_height = max(int(current_height * scale_factor), min_height)

    # Use high-quality resampling
    if scale_factor != 1 or new_width != current_width or new_height != current_height:
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)

    return image


def _correct_skew(img: np.ndarray) -> np.ndarray:
    """Detect and correct skew in the image."""
    import cv2

    # Create a copy for skew detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Use HoughLines to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Convert to skew angle (-45 to 45 degrees)
            if angle > 90:
                angle = angle - 180
            elif angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90

            # Only consider reasonable skew angles
            if -30 <= angle <= 30:
                angles.append(angle)

        if angles:
            # Use median angle to avoid outliers
            skew_angle = np.median(angles)

            # Only correct if skew is significant (> 0.5 degrees)
            if abs(skew_angle) > 0.5:
                # Get image center and rotation matrix
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

                # Apply rotation
                img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    return img


def _remove_shadows(img: np.ndarray) -> np.ndarray:
    """Remove shadows and improve lighting uniformity."""
    import cv2

    # Create a large kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    # Estimate background using morphological opening
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Smooth the background
    background = cv2.medianBlur(background, 19)

    # Subtract background to remove shadows
    # Convert to float to avoid underflow
    img_float = img.astype(np.float32)
    background_float = background.astype(np.float32)

    # Normalize by background
    normalized = img_float / (background_float + 1e-6) * 255

    # Clip values and convert back to uint8
    result = np.clip(normalized, 0, 255).astype(np.uint8)

    return result


def _enhance_contrast_adaptive(img: np.ndarray) -> np.ndarray:
    """Apply adaptive contrast enhancement optimized for text."""
    import cv2

    # Apply CLAHE with optimized parameters for text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Additional gamma correction for better text visibility
    gamma = 1.2
    gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                           for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, gamma_table)

    return enhanced


def _advanced_noise_reduction(img: np.ndarray) -> np.ndarray:
    """Apply advanced noise reduction while preserving text edges."""
    import cv2

    # Bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # Non-local means denoising for additional noise reduction
    denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)

    return denoised


def _optimal_binarization(img: np.ndarray) -> np.ndarray:
    """Apply optimal binarization techniques for text recognition."""
    import cv2

    # Try multiple binarization methods and combine results

    # Method 1: Adaptive Gaussian threshold
    binary1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Method 2: Adaptive Mean threshold
    binary2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Method 3: Otsu's threshold
    _, binary3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine methods using bitwise operations
    # Take the intersection of adaptive methods for cleaner text
    combined = cv2.bitwise_and(binary1, binary2)

    # Use Otsu as fallback for areas where adaptive methods disagree
    mask = cv2.bitwise_xor(binary1, binary2)
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(binary3, mask))

    return combined


def _text_morphological_operations(img: np.ndarray) -> np.ndarray:
    """Apply morphological operations optimized for text."""
    import cv2

    # Remove small noise with opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Connect broken text with closing
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

    # Strengthen text lines
    kernel_strengthen = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_strengthen, iterations=1)

    return cleaned


def _final_quality_enhancement(img: np.ndarray) -> np.ndarray:
    """Apply final quality enhancements for OCR."""
    import cv2

    # Ensure text is black on white background
    # Count black vs white pixels to determine if inversion is needed
    black_pixels = np.sum(img == 0)
    white_pixels = np.sum(img == 255)

    if black_pixels > white_pixels:
        # More black than white, likely inverted
        img = cv2.bitwise_not(img)

    # Final smoothing to reduce jagged edges
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img


def assess_image_quality(img: np.ndarray) -> Dict[str, float]:
    """
    Assess image quality metrics for OCR preprocessing validation.

    Returns:
        Dictionary with quality metrics
    """
    import cv2

    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(img)

    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = laplacian.var()

    # Calculate brightness (mean pixel value)
    brightness = np.mean(img)

    # Calculate noise level (using high-frequency content)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    noise = cv2.filter2D(img, -1, kernel)
    noise_level = np.std(noise)

    # Calculate text-to-background ratio
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text_pixels = np.sum(binary == 0)  # Assuming text is black
    background_pixels = np.sum(binary == 255)
    text_ratio = text_pixels / (text_pixels + background_pixels) if (text_pixels + background_pixels) > 0 else 0

    return {
        'contrast': float(contrast),
        'sharpness': float(sharpness),
        'brightness': float(brightness),
        'noise_level': float(noise_level),
        'text_ratio': float(text_ratio)
    }


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