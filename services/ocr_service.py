import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple
import re
import logging
from datetime import datetime
import os

from core.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    """Service for OCR processing of ID cards and documents"""
    
    def __init__(self):
        # Configure Tesseract for Khmer language
        self.tesseract_config = '--oem 3 --psm 6 -l khm+eng'
        self.confidence_threshold = 30
        
        # Khmer ID card patterns
        self.id_patterns = {
            'id_number': r'[០-៩]{9}',  # 9 Khmer digits
            'phone': r'[០-៩]{8,10}',   # 8-10 Khmer digits
            'date': r'[០-៩]{1,2}/[០-៩]{1,2}/[០-៩]{4}',  # DD/MM/YYYY format
        }
        
        # Common Khmer words on ID cards
        self.khmer_keywords = {
            'name': ['ឈ្មោះ', 'នាម'],
            'id_number': ['លេខសម្គាល់', 'អត្តសញ្ញាណប័ណ្ណ'],
            'birth_date': ['ថ្ងៃខែឆ្នាំកំណើត', 'កំណើត'],
            'address': ['អាសយដ្ឋាន', 'ទីលំនៅ'],
            'nationality': ['សញ្ជាតិ'],
            'gender': ['ភេទ']
        }
    
    async def process_id_card(self, image_path: str) -> Dict[str, any]:
        """Process ID card image and extract information"""
        try:
            # Load and preprocess image
            processed_image = await self._preprocess_image(image_path)
            
            # Extract text using OCR
            extracted_text = await self._extract_text(processed_image)
            
            # Parse extracted information
            parsed_data = await self._parse_id_card_data(extracted_text)
            
            # Validate extracted data
            validation_result = await self._validate_extracted_data(parsed_data)
            
            return {
                'success': True,
                'extracted_data': parsed_data,
                'validation': validation_result,
                'raw_text': extracted_text,
                'confidence_score': validation_result.get('overall_confidence', 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'extracted_data': {},
                'validation': {},
                'raw_text': '',
                'confidence_score': 0
            }
    
    async def process_document(self, image_path: str, document_type: str = 'general') -> Dict[str, any]:
        """Process general document and extract text"""
        try:
            # Load and preprocess image
            processed_image = await self._preprocess_image(image_path)
            
            # Extract text using OCR
            extracted_text = await self._extract_text(processed_image)
            
            # Get confidence data
            confidence_data = await self._get_text_confidence(processed_image)
            
            return {
                'success': True,
                'extracted_text': extracted_text,
                'document_type': document_type,
                'confidence_data': confidence_data,
                'word_count': len(extracted_text.split()),
                'character_count': len(extracted_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'confidence_data': {},
                'word_count': 0,
                'character_count': 0
            }
    
    async def _preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format for preprocessing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(binary)
        
        # Additional PIL enhancements
        processed_image = processed_image.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)
        
        return processed_image
    
    async def _extract_text(self, image: Image.Image) -> str:
        """Extract text from preprocessed image using Tesseract"""
        try:
            # Configure Tesseract path if specified
            if hasattr(settings, 'TESSERACT_CMD') and settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            
            # Extract text
            text = pytesseract.image_to_string(
                image, 
                config=self.tesseract_config,
                lang='khm+eng'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    async def _get_text_confidence(self, image: Image.Image) -> Dict[str, any]:
        """Get confidence scores for extracted text"""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence statistics
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                return {
                    'average_confidence': sum(confidences) / len(confidences),
                    'min_confidence': min(confidences),
                    'max_confidence': max(confidences),
                    'total_words': len(confidences),
                    'high_confidence_words': len([c for c in confidences if c >= 80]),
                    'low_confidence_words': len([c for c in confidences if c < 50])
                }
            else:
                return {
                    'average_confidence': 0,
                    'min_confidence': 0,
                    'max_confidence': 0,
                    'total_words': 0,
                    'high_confidence_words': 0,
                    'low_confidence_words': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting confidence data: {str(e)}")
            return {}
    
    async def _parse_id_card_data(self, text: str) -> Dict[str, str]:
        """Parse extracted text to identify ID card fields"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        parsed_data = {
            'id_number': self._extract_id_number(lines),
            'name': self._extract_name_from_text(lines),
            'birth_date': self._extract_date_from_text(lines),
            'address': self._extract_address_from_text(lines),
            'phone': self._extract_phone_from_text(lines),
            'nationality': self._extract_nationality_from_text(lines),
            'gender': self._extract_gender_from_text(lines)
        }
        
        # Remove None values
        return {k: v for k, v in parsed_data.items() if v is not None}
    
    def _extract_id_number(self, lines: List[str]) -> Optional[str]:
        """Extract ID number from text lines"""
        for line in lines:
            # Look for Khmer digit patterns
            match = re.search(self.id_patterns['id_number'], line)
            if match:
                return match.group()
            
            # Also check for regular digits
            digit_match = re.search(r'\b\d{9}\b', line)
            if digit_match:
                return digit_match.group()
        
        return None
    
    def _extract_phone_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract phone number from text lines"""
        for line in lines:
            # Look for Khmer digit patterns
            match = re.search(self.id_patterns['phone'], line)
            if match:
                return match.group()
            
            # Also check for regular digits with common prefixes
            phone_match = re.search(r'\b(?:0|\+855)?[1-9]\d{7,9}\b', line)
            if phone_match:
                return phone_match.group()
        
        return None
    
    def _extract_date_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract date from text lines"""
        for line in lines:
            # Look for Khmer date patterns
            match = re.search(self.id_patterns['date'], line)
            if match:
                return match.group()
            
            # Also check for regular date formats
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
                r'\b\d{4}/\d{1,2}/\d{1,2}\b'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group()
        
        return None
    
    def _extract_nationality_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract nationality from text lines"""
        for line in lines:
            for keyword in self.khmer_keywords['nationality']:
                if keyword in line:
                    # Extract text after the keyword
                    parts = line.split(keyword)
                    if len(parts) > 1:
                        nationality = parts[1].strip()
                        if nationality:
                            return nationality
        
        return None
    
    def _extract_gender_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract gender from text lines"""
        gender_keywords = {
            'male': ['ប្រុស', 'Male', 'M'],
            'female': ['ស្រី', 'Female', 'F']
        }
        
        for line in lines:
            for gender, keywords in gender_keywords.items():
                for keyword in keywords:
                    if keyword in line:
                        return gender
        
        return None
    
    def _extract_name_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract name from text lines"""
        for line in lines:
            for keyword in self.khmer_keywords['name']:
                if keyword in line:
                    # Extract text after the keyword
                    name_part = line.replace(keyword, '').strip()
                    if name_part and len(name_part) > 2:
                        return name_part
        
        # If no keyword found, look for lines with Khmer characters that might be names
        for line in lines:
            if re.search(r'[ក-៹]', line):  # Contains Khmer characters
                # Skip lines that contain common keywords
                has_keywords = any(
                    any(kw in line for kw in keywords) 
                    for keywords in self.khmer_keywords.values()
                )
                if not has_keywords:
                    name_line = line.strip()
                    if name_line and len(name_line) > 2:
                        return name_line
        
        return None
    
    def _extract_address_from_text(self, lines: List[str]) -> Optional[str]:
        """Extract address from text lines"""
        address_lines = []
        found_address_keyword = False
        
        for line in lines:
            # Look for address keywords
            for keyword in self.khmer_keywords['address']:
                if keyword in line:
                    found_address_keyword = True
                    address_part = line.replace(keyword, '').strip()
                    if address_part:
                        address_lines.append(address_part)
                    break
            
            # If we found address keyword, collect subsequent lines
            elif found_address_keyword and line.strip():
                # Stop if we hit another keyword section
                is_other_section = any(
                    any(kw in line for kw in keywords) 
                    for keywords in self.khmer_keywords.values()
                )
                if is_other_section:
                    break
                address_lines.append(line.strip())
        
        return ' '.join(address_lines) if address_lines else None
    
    async def _validate_extracted_data(self, data: Dict[str, str]) -> Dict[str, any]:
        """Validate extracted data and calculate confidence scores"""
        validation = {
            'valid_fields': [],
            'invalid_fields': [],
            'missing_fields': [],
            'field_confidence': {},
            'overall_confidence': 0
        }
        
        required_fields = ['id_number', 'name']
        optional_fields = ['birth_date', 'address', 'phone', 'nationality', 'gender']
        
        # Check required fields
        for field in required_fields:
            if field in data and data[field]:
                validation['valid_fields'].append(field)
                validation['field_confidence'][field] = self._calculate_field_confidence(field, data[field])
            else:
                validation['missing_fields'].append(field)
                validation['field_confidence'][field] = 0
        
        # Check optional fields
        for field in optional_fields:
            if field in data and data[field]:
                validation['valid_fields'].append(field)
                validation['field_confidence'][field] = self._calculate_field_confidence(field, data[field])
            else:
                validation['field_confidence'][field] = 0
        
        # Calculate overall confidence
        if validation['field_confidence']:
            total_confidence = sum(validation['field_confidence'].values())
            field_count = len(validation['field_confidence'])
            validation['overall_confidence'] = total_confidence / field_count
        
        return validation
    
    def _calculate_field_confidence(self, field_type: str, value: str) -> float:
        """Calculate confidence score for a specific field"""
        if not value:
            return 0.0
        
        confidence = 50.0  # Base confidence
        
        if field_type == 'id_number':
            # ID numbers should be exactly 9 digits
            if re.match(r'^[០-៩]{9}$', value) or re.match(r'^\d{9}$', value):
                confidence = 95.0
            elif re.search(r'[០-៩]{8,10}', value) or re.search(r'\d{8,10}', value):
                confidence = 75.0
        
        elif field_type == 'phone':
            # Phone numbers should be 8-10 digits
            if re.match(r'^[០-៩]{8,10}$', value) or re.match(r'^\d{8,10}$', value):
                confidence = 90.0
        
        elif field_type == 'birth_date':
            # Dates should match expected patterns
            if re.match(r'^[០-៩]{1,2}/[០-៩]{1,2}/[០-៩]{4}$', value) or re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', value):
                confidence = 85.0
        
        elif field_type == 'name':
            # Names should contain Khmer characters and be reasonable length
            if re.search(r'[ក-៹]', value) and 2 <= len(value) <= 50:
                confidence = 80.0
        
        elif field_type in ['address', 'nationality', 'gender']:
            # Text fields should have reasonable length
            if 2 <= len(value) <= 100:
                confidence = 70.0
        
        return min(confidence, 100.0)

# Create global instance
ocr_service = OCRService()