import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import logging
from typing import List, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

# Set Tesseract path
tesseract_path = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Verify Tesseract exists
if not os.path.exists(tesseract_path):
    raise FileNotFoundError(f"Tesseract not found at {tesseract_path}. Please install Tesseract and update the path.")

# Set the Tesseract command for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Add Tesseract to system PATH if needed
tesseract_dir = os.path.dirname(tesseract_path)
if tesseract_dir not in os.environ['PATH'].lower():
    os.environ['PATH'] = f"{tesseract_dir};{os.environ['PATH']}"

# Verify Tesseract is accessible
try:
    import subprocess
    subprocess.run([tesseract_path, '--version'], capture_output=True, text=True, check=True)
    print("Tesseract is properly installed and accessible")
except Exception as e:
    print(f"Warning: Tesseract verification failed: {e}")

def preprocess_image(image_path: str) -> Image.Image:
    """
    Enhanced preprocessing for Khmer text in ID card images.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed PIL Image
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize if image is too large (better for OCR)
        height, width = img.shape[:2]
        if max(height, width) > 2000:
            scale = 2000.0 / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding with optimized parameters for Khmer script
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 7  # Increased block size and C value for better Khmer text
        )
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Remove small noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Apply slight blur to smooth the image
        smoothed = cv2.GaussianBlur(sure_bg, (3, 3), 0)
        
        # Convert back to PIL Image
        return Image.fromarray(smoothed)
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        return Image.open(image_path).convert('L')

def extract_text_with_tesseract(
    image_path: str,
    lang: str = 'khm+eng',
    config: str = ('--psm 6 --oem 3 -c preserve_interword_spaces=1 '  # PSM 6: Assume a single uniform block of text
                  '--dpi 300 -c tessedit_pageseg_mode=6 '  # PSM 6
                  '-c textord_min_linesize=2.5 '  # Better for small text
                  '-c textord_old_baselines=0 '  # Disable old baseline algorithm
                  '-c language_model_penalty_non_dict_word=0.5 '  # Be more lenient with non-dictionary words
                  '-c language_model_penalty_non_freq_dict_word=0.3 '  # Be more lenient with infrequent words
                  '-c load_system_dawg=0 '  # Don't load system dictionary (can help with Khmer)
                  '-c load_freq_dawg=0 '  # Don't load frequent words dictionary
                  '-c textord_min_xheight=8 '  # Minimum x-height (helps with small text)
                  '-c textord_tsv_medium_size_limit=10000 '  # Handle larger blocks of text
                  '-c textord_tabfind_show_blocks=0 '  # Disable debug output
                  '-c textord_min_linesize=2.5 '  # Minimum character height
                  '-c textord_oldbl_holed_low=1.5 '  # Better for Khmer script
                  '-c textord_oldbl_holed_size_limit=1000 '  # Better for Khmer script
                  '-c textord_oldbl_corr_mean_y=1 '  # Better for Khmer script
                  '-c textord_oldbl_jog_lim=8 '  # Better for Khmer script
                  '-c textord_oldbl_jog_lim=8 '  # Better for Khmer script
                  '-c textord_noise_sizelimit=0.25 '  # Better for Khmer script
                  '-c textord_noise_normratio=0.5 '  # Better for Khmer script
                  '-c textord_noise_sizefraction=0.5 '  # Better for Khmer script
                  '-c textord_noise_sp_ratio=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio2=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio3=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio4=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio5=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio6=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio7=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio8=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio9=0.5 '  # Better for Khmer script
                  '-c textord_noise_sn_ratio10=0.5')  # Better for Khmer script
) -> List[Tuple[str, float]]:
    """
    Extract text from image using Tesseract OCR with enhanced settings for ID cards.
    
    Args:
        image_path: Path to the image file
        lang: Language code(s) for OCR (default: 'khm+eng' for Khmer and English)
        config: Tesseract configuration options
        
    Returns:
        List of tuples containing (text, confidence) for each detected text block
    """
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        
        # Use Tesseract to extract text with confidence
        data = pytesseract.image_to_data(
            preprocessed_img,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and confidence for each detected text block
        results = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Only include results with confidence > 0
                results.append((
                    data['text'][i].strip(),
                    int(data['conf'][i]) / 100.0  # Convert to 0-1 range
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Tesseract OCR: {str(e)}")
        return []

def extract_text_with_tesseract_simple(
    image_path: str,
    lang: str = 'khm+eng',
    config: str = '--psm 6 --oem 3 -c preserve_interword_spaces=1 --dpi 300'
) -> str:
    """
    Extract text from image using Tesseract OCR (simpler version).
    
    Args:
        image_path: Path to the image file
        lang: Language code(s) for OCR (default: 'khm+eng' for Khmer and English)
        config: Tesseract configuration options
        
    Returns:
        Extracted text as a single string
    """
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(
            preprocessed_img,
            lang=lang,
            config=config
        )
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error in Tesseract OCR: {str(e)}")
        return ""
