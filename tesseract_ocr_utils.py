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
tesseract_path = '/usr/bin/tesseract'

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
        
        # The 'opening' image is now binarized and has small noise removed.
        # Further dilation and blurring (previously done on 'sure_bg') are generally
        # not ideal for OCR as they can thicken or blur text features.
        # The direct output of 'opening' should be better for OCR.
        
        # Convert back to PIL Image
        return Image.fromarray(opening)
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        # Ensure fallback returns a PIL image, as the function signature suggests.
        pil_img = Image.open(image_path)
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        return pil_img

def extract_text_with_tesseract(
    image_path: str,
    lang: str = 'khm+eng',
    config: str = (
        '--oem 1 --psm 11 -c preserve_interword_spaces=1 '  # OEM 1 for LSTM, PSM 11 for sparse text
        '--dpi 300 '
        '-c load_system_dawg=0 '  # Don't load system dictionary (can help with Khmer names)
        '-c load_freq_dawg=0 '    # Don't load frequent words dictionary
        # Removed legacy textord_ parameters and dictionary penalties as dawgs are off.
        # PSM 11: Sparse text. Find as much text as possible in no particular order. Good for ID cards.
        # OEM 1: Use LSTM OCR engine only.
        # preserve_interword_spaces=1: Crucial for Khmer.
        # load_system_dawg=0 / load_freq_dawg=0: Useful for proper nouns in IDs.
    )
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
    # Simplified config, recommend using the more detailed extract_text_with_tesseract for ID cards
    # Defaulting to PSM 11 and OEM 1 for general purpose if this simple one is used.
    config: str = '--oem 1 --psm 11 -c preserve_interword_spaces=1 --dpi 300'
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
